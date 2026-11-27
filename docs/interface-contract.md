# 成员 1–5 接口契约

本文定义成员 1 控制面向成员 2–5 暴露的进程内接口。数据对象在
`zhixun_harness/contracts.py`，接口在 `zhixun_harness/pipeline.py`，占位实现
在 `zhixun_harness/fakes.py`。

## 一、所有权

| 角色 | 提供 | 不负责 |
| --- | --- | --- |
| 成员 1 | `StageBundle`、哈希、校验、预算、主循环、候选提交、状态保存 | 漂移算法、提示词、训练、指标 |
| 成员 2 | `Pipeline.diagnose()` | 训练、上线 |
| 成员 3 | `Pipeline.decide()`，通过 `ModelGateway` 调平台模型 | 训练、评测 |
| 成员 4 | `Pipeline.train_candidate()` | 评测、门控 |
| 成员 5 | `Evaluator.evaluate()`、`Pipeline.evaluate_and_commit()`、`Predictor.predict()` | 编排、artifact 写盘 |

## 二、输入：StageBundle

```text
bundle.task_id / stage_id / task_type
bundle.feature_schema_id / feature_count / sample_count
bundle.classes                 # 分类标签
bundle.ids                     # tuple[str], 与数组同序
bundle.features                # np.float64 (n, feature_count)
bundle.targets                 # int64 或 float64，hidden 时为 None
bundle.group_ids               # 样本级分组，用于泄漏检查
bundle.label_status            # available / delayed / hidden
bundle.source_sha256           # 原始文件哈希
bundle.canonical_sha256        # 规范化内容哈希
```

评测再给一份验证 bundle（`eval_bundle`），由 `/data/validation.jsonl` 或本地
stage 的 `labels.jsonl` 构造，成员 5 用它计算候选指标。训练 bundle 不等于评测
bundle。

对象是 frozen dataclass。成员 2–5 **不得原地修改**，需要派生数据时自行复制。

## 三、接口

```python
class Pipeline(Protocol):
    def diagnose(self, bundle, current_eval, budget) -> Diagnosis: ...
    def decide(self, diagnosis, history, budget) -> Decision: ...
    def train_candidate(self, bundle, decision, active, budget) -> CandidateModel | None: ...
    def evaluate_and_commit(self, bundle, candidate, active, gate, eval_bundle=None) -> Evaluation: ...

class Evaluator(Protocol):
    def evaluate(self, model, bundle) -> dict[str, float]: ...

class Predictor(Protocol):
    def predict(self, sample, model_dir) -> dict | list: ...

class ModelGateway(Protocol):
    def call(self, messages: list[dict], **options) -> ModelReceipt: ...
    @property
    def call_ids(self) -> Sequence[str]: ...
```

`train_candidate` 返回的 `CandidateModel.checkpoint_bytes` 是候选检查点字节；
本地 store 和官方 `submit_candidate` 都从这里取文件。返回 `None` 表示不产生
候选，编排器会记录失败并保留当前模型。

`evaluate_and_commit` 只返回 `Evaluation`，**不修改** active 指针；上线、保留
或回滚由编排器根据 `Evaluation.gate` 执行。

## 四、输出对象

| 对象 | 关键字段 |
| --- | --- |
| `Diagnosis` | `drift_score`、`drift_detected`、`drift_type`、`current_metrics`、`budget_remaining`、`recommendation` |
| `Decision` | `based_on`、`action`、`params`、`rationale`、`estimated_cost` |
| `CandidateModel` | `checkpoint_uri`、`checkpoint_sha256`、`parent_checkpoint_sha256`、`training_metrics`、`config`、`seed` |
| `Evaluation` | `candidate_id`、`baseline_metrics`、`candidate_metrics`、`metric_name`、`delta`、`gate` |

`action` 只能是 `no_update`、`adaptive_update`、`retrain_full`、
`retrain_recent`、`finetune`、`rollback`。`adaptive_update` 使用
`max_candidates`、`max_epochs` 和 `seed`，由成员4在当前任务的少量候选计划中
逐 epoch 选优。候选可以包含数据窗口和历史回放等通用变换；候选必须保持父检查
点的任务、特征路径、类别顺序和 MLP 宽度。平台要求基于官方 baseline 和父检查
点续训，不允许提交改变结构的新模型。其余参数由 `contracts.Decision` 严格校验。

`rollback` 的 `params` 必须包含 `target_artifact_id` 或
`target_checkpoint_sha256` 其中之一。目标必须出现在本轮 `rnd.artifacts`，
Runner 使用 `rnd.restore_artifact()` 下载并校验全部文件，再将该历史 artifact
提交为 `finish(selected_artifact_id=...)`。门控拒绝候选时产生的 rollback 仍表示
“不上线候选并保留父模型”，与主动恢复历史 artifact 是两种不同路径。

预算中 `max_candidates_per_stage` 限制最终提交候选数；
`max_training_trials_per_stage` 单独限制一次自适应搜索内部实际执行的训练试验数。

`gate` 只能是 `promote`、`keep`、`rollback`。

## 五、如何接入

1. 在 `pipeline.py` 的协议下实现自己的类，不要改编排器。
2. 在 `round_runner.run_round(..., pipeline=, evaluator=)` 或本地
   `orchestrator.main` 中注入。
3. 分类模型在 `Predictor` 中返回 `[{"label": ..., "score": ...}]`；回归返回
   `{"value": ...}`。SDK 负责逐行读写和校验。
4. 任何算法错误抛 `TrainingError`、`EvaluationError`、`DecisionError`，不要
   吞掉；编排器会记录为 `FAILED_RECOVERABLE` 并保留旧模型。
5. 不要直接写 artifact 目录或 `/previous`；只能通过 `Round` / `ArtifactStore`。

## 六、官方轮次映射

| 平台概念 | 内部概念 |
| --- | --- |
| `rnd.dataset_key` | `bundle.task_id` |
| `rnd.round_index` | `stage_id = round-XXXX` |
| `rnd.data.training_rows()` | 训练 `StageBundle` |
| `rnd.data.validation_rows()` | 评测 `StageBundle` |
| `rnd.parent.checkpoint_sha256` | `ActiveModel.checkpoint_sha256` |
| `rnd.model()` | `ModelGateway.call()` |
| `rnd.history()` | `history` 列表 |
| `rnd.submit_candidate()` | 候选提交 |
| `rnd.save_state()` | 跨轮状态 |
| `rnd.finish(selected_artifact_id)` | 上线 / 保留 / 回滚决策 |

正式运行不维护 `active-model.json`；它是平台的父产物。本地 store 只服务离线
两阶段开发和测试。
