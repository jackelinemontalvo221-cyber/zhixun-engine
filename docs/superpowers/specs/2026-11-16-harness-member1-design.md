# 成员 1 Harness 规格说明

- 日期：2026-11-16
- 状态：已按官方 workspace.v2 SDK 修订（2026-11-17），待评审
- 责任人：成员 1（数据入口、统一契约、主循环控制面、artifact 管理、Docker 交付）
- 关联材料：`项目总结.txt`、`README.md`、`cpu-a-guide-20261110-v1/`、`cpu-b-rul-20261110-v1/`、`cpu-c-guide-20261110-v1/`

---

## 0. 一句话

本规格定义一套**与具体任务无关**的 Harness 控制面：它接收平台声明的阶段数据，校验并规范化为统一契约，驱动「当前模型评估 → 诊断 → 决策 → 候选训练 → 门控上线/回滚 → 记录」闭环，并以原子、可复现、可审计的方式保存每一步证据。

---

## 1. 背景与目标

赛题要求提交的不是单个模型，而是一套无人干预的工业模型自进化系统。成员 1 负责让这条流水线"能转起来"：数据怎么进、契约长什么样、状态怎么推进、产物怎么写、失败怎么恢复、镜像怎么交付。

本规格只覆盖成员 1 的边界。成员 2–5 的算法与模型实现通过四个函数接口接入，不在本文范围内。

目标：

1. 在本地用 2 个阶段（batch1 → batch2）跑通完整闭环的空架子。
2. 核心逻辑不依赖 A/B/C 三个参考包，正式比赛换数据/换接口时只替换来源适配器。
3. 每一次训练、评估、决策都有唯一证据 ID 和可回放的 artifact。
4. 失败只影响当前阶段，已上线模型永不被静默覆盖。
5. 在 Linux、CPU-only、断网、只读输入的环境下可运行。

---

## 2. 范围

### 2.1 本文覆盖

- 数据契约与校验（`StageDescriptor`、`StageProfile`、`StageBundle`）
- 成员间 JSON 契约（`Diagnosis`、`Decision`、`CandidateModel`、`Evaluation`、`StageRecord`）
- 阶段来源抽象与本地实现
- 哈希、幂等与证据
- ArtifactStore 原子事务与 `active-model` 指针
- 主循环状态机、错误恢复、预算熔断
- 配置、CLI、日志与退出码
- 测试矩阵与交付物清单

### 2.2 本文不覆盖

- 漂移检测算法（成员 2）
- 大模型决策与提示词（成员 3）
- 工业模型训练与候选生成（成员 4）
- 评测指标、SwanLab 记录规范（成员 5）
- 正式平台 SDK 的具体实现（只留接口与待确认清单）
- 最终 Docker 镜像内容（只定义交付要求）
- 隐藏测试数据的任何读取方式

---

## 3. 关键约束与已定决策

| 编号 | 决策 | 理由 |
| --- | --- | --- |
| D-01 | 内部包为 `zhixun_harness/`，本地入口 `python -m zhixun_harness.orchestrator`；官方入口仍是 `/app/harness.py` | 包名不能叫 `harness`，否则会被官方入口脚本遮蔽 |
| D-02 | 采用「契约优先 + 进程内函数接口」路线 | 类型明确、可用 fake 测试、符合系统设计评分项 |
| D-03 | 三个 `cpu-*-20261110-v1` 包**只是开发参考数据**，核心不依赖包路径 | 正式比赛另有数据与接口 |
| D-04 | `task_id` 为不透明字符串；`task_type`、维度、类别、分组字段由阶段 `profile.json` 声明 | 不把 A/B/C 写死 |
| D-05 | 本地第一阶段只做 2 个阶段（batch1 → batch2） | 仓库没有 batch3–10；接口仍按 N 阶段设计 |
| D-06 | `StageBundle` 直接携带 numpy 数组 | 成员 2–5 拿到即可用 |
| D-07 | 本地 2 个阶段标签均为 `available`，同时保留 evaluator-only oracle 语义 | 先跑通闭环，后续可切 `hidden` |
| D-08 | 第一版 `Pipeline` 只接 fake 实现；真实训练适配器只定义接口 | 训练归成员 4 |
| D-09 | 数据文件中的事实以三个包为准，与总结/README 冲突时以包为准 | 用户明确指示 |

其他硬约束：

- 正式评测环境：无 GPU、断网、CPU-only、资源受限、输入只读。
- Python ≥ 3.10；开发环境可用 3.12 或更高。
- 文本统一 UTF-8；路径统一 `pathlib.Path`；LF 换行。
- 不使用 `shell=True`；不依赖 PowerShell/CMD。
- 只使用平台允许的数据；不读取隐藏标签。

---

## 4. 总体架构

```text
                    ┌──────────────────────────────┐
                    │        orchestrator          │
                    │  状态机 / 重试 / 信号 / 预算  │
                    └───────┬──────────────┬───────┘
                            │              │
                 ┌──────────▼───┐   ┌──────▼────────┐
                 │   sources    │   │    store      │
                 │ LocalManifest│   │ ArtifactStore │
                 │ Platform(stub)│  │ active-model  │
                 └──────┬───────┘   └──────┬────────┘
                        │                  │
                 ┌──────▼───────┐   ┌──────▼────────┐
                 │    codec     │   │   hashing     │
                 │ 通用校验+overlay│ │ source/canonical│
                 └──────┬───────┘   └───────────────┘
                        │
                 ┌──────▼───────┐
                 │  contracts   │
                 │ Bundle/Diag/ │
                 │ Decision/... │
                 └──────┬───────┘
                        │
                 ┌──────▼───────┐
                 │   pipeline   │
                 │ 4 个成员接口  │
                 └──────┬───────┘
                        │
              ┌─────────▼─────────┐
              │ fakes / 真实实现   │
              │ (成员 2–5 接入)    │
              └───────────────────┘
```

### 4.1 模块职责

| 模块 | 职责 | 不负责 |
| --- | --- | --- |
| `zhixun_harness/contracts.py` | 所有数据对象的定义、序列化、严格校验 | 不读文件、不训练 |
| `zhixun_harness/taskspec.py` | 从 `profile.json` 构造 `TaskSpec` | 不内置任务常量 |
| `zhixun_harness/codec.py` | 通用校验 + 可插拔 dev overlay，产出 `StageBundle` | 不做漂移或训练 |
| `zhixun_harness/hashing.py` | 文件哈希、canonical 哈希 | 不决定是否复用 |
| `zhixun_harness/sources.py` | 阶段发现与读取 | 不校验样本内容 |
| `zhixun_harness/store.py` | artifact 事务、原子提交、`active-model` | 不做业务决策 |
| `zhixun_harness/budget.py` | 资源计量与熔断 | 不执行训练 |
| `zhixun_harness/pipeline.py` | 定义四个成员接口 | 不实现算法 |
| `zhixun_harness/fakes.py` | 确定性假实现 | 不代表真实效果 |
| `zhixun_harness/orchestrator.py` | 状态机、主循环、CLI、信号、退出码 | 不含算法 |
| `zhixun_harness/errors.py` | 异常分类 | 不处理异常 |
| `zhixun_harness/config.py` | 配置加载与校验 | 不含默认业务阈值 |
| `zhixun_harness/platform.py` | `PlatformStageSource` 与官方适配器桩 | 未拿到官方契约前不实现细节 |

---

## 5. 数据契约

### 5.1 通用规则

- 所有对象有 `schema_version`，版本不匹配直接拒绝。
- JSON 严格解析：UTF-8、拒绝重复键、拒绝 `NaN`/`Infinity`、拒绝未知字段。
- 所有 ID：非空字符串，长度 ≤ 128，不允许控制字符。
- 时间戳：RFC 3339 UTC，例如 `2026-11-16T08:00:00+00:00`。
- 数值：有限浮点；回归目标 ≥ 0；分类标签必须属于声明的 `classes`。
- 对象在进程内为 frozen dataclass；跨进程只交换 JSON。
- 数组使用 `numpy.float64`（特征）与 `int64`/`float64`（目标）。

### 5.2 StageDescriptor

由 `StageSource` 产出，是"这一阶段在哪"的最小信息。

| 字段 | 类型 | 必填 | 约束 |
| --- | --- | --- | --- |
| `schema_version` | str | 是 | `"stage-descriptor.v1"` |
| `task_id` | str | 是 | 不透明；核心不解释 |
| `stage_id` | str | 是 | 不透明；核心不推断总数/顺序 |
| `stage_dir` | str | 是 | 相对 `stage_root`，禁止绝对路径与 `..` |
| `source` | str | 是 | `"local-manifest"` / `"platform"` |

### 5.3 StageProfile（阶段自带声明）

```text
<stage_dir>/profile.json
```

| 字段 | 类型 | 必填 | 约束 |
| --- | --- | --- | --- |
| `schema_version` | str | 是 | `"stage-profile.v1"` |
| `task_id` | str | 是 | 必须与 descriptor 一致 |
| `task_type` | str | 是 | `classification` / `regression` |
| `feature_schema_id` | str | 是 | 例如 `gas.uci.features128.v1` |
| `feature_count` | int | 是 | 1..4096；必须与实际展开后一致 |
| `feature_paths` | list[str] | 否 | 声明特征顺序；缺省时从首行推断并要求全阶段一致 |
| `classes` | list[str] | 分类必填 | ≥2、唯一、非空；回归必须为空 |
| `group_field` | str \| list[str] | 是 | 标签行里的分组来源；可为组合键，如 C 的 `["meta.source_subset","meta.unit"]` |
| `label_status` | str | 是 | `available` / `delayed` / `hidden` |
| `feature_uri` | str | 是 | 默认 `"features.jsonl"` |
| `label_uri` | str | 是 | 默认 `"labels.jsonl"` |
| `forbidden_input_fields` | list[str] | 否 | 归一化后匹配的禁用键名 |
| `group_split` | str | 否 | `disjoint` / `allow_overlap`，默认 `disjoint` |
| `limits` | object | 否 | `max_rows`、`max_features` 覆盖默认上限 |

规则：

- profile 是**阶段数据的一部分**，权威性高于任何 dev 配置。
- 核心不内置任何任务的维度、类别或禁用字段；A/B/C 的这些内容通过 dev overlay 叠加。
- `label_status=hidden` 时，`labels.jsonl` 不得出现在阶段目录内；目标只能由 evaluator 的 oracle 提供。
- `label_status=delayed` 时，首次读取允许 `targets=None`，评测前由 oracle 补齐。

### 5.4 StageBundle

统一对象，成员 2–5 只认它。

| 字段 | 类型 | 必填 | 说明 |
| --- | --- | --- | --- |
| `schema_version` | str | 是 | `"stage-bundle.v1"` |
| `task_id` | str | 是 | 不透明 |
| `stage_id` | str | 是 | 不透明 |
| `task_type` | str | 是 | 来自 profile |
| `feature_schema_id` | str | 是 | 来自 profile |
| `feature_uri` | str | 是 | 相对路径 |
| `label_uri` | str | 是 | 相对路径 |
| `label_status` | str | 是 | 来自 profile |
| `sample_count` | int | 是 | > 0 |
| `feature_count` | int | 是 | 与 profile 一致 |
| `group_ids` | list[str] | 是 | 样本级，与 `ids` 等长 |
| `classes` | list[str] | 分类必填 | 与 profile 一致 |
| `source_sha256` | str | 是 | 见 6.1 |
| `canonical_sha256` | str | 是 | 见 6.2 |
| `previous_stage_id` | str/null | 否 | 本地模拟可填 |
| `created_at` | str | 是 | RFC 3339 UTC |

运行时附带（不写入 `bundle.json`）：

```text
ids:        list[str]                  # 样本级，顺序与数组一致
features:   np.ndarray[float64]        # shape (n, feature_count)
targets:    np.ndarray | None          # int64 或 float64
```

`StageBundle.to_record()` 输出不含数组的 JSON；数组可用 `bundle.npz` 落盘（配置项 `embed_arrays`，默认 false）。

### 5.5 Diagnosis（成员 2 → 成员 3）

| 字段 | 类型 | 必填 | 说明 |
| --- | --- | --- | --- |
| `schema_version` | str | 是 | `"diagnosis.v1"` |
| `diagnosis_id` | str | 是 | `diag-<uuid4>` |
| `task_id` / `stage_id` / `attempt_id` | str | 是 | 定位 |
| `drift_score` | float | 是 | 有限、非负；归一化含义由成员 2 定义 |
| `drift_detected` | bool | 是 | |
| `drift_type` | str | 是 | `none` / `covariate` / `concept` / `mixed` / `unknown` |
| `current_metrics` | object | 是 | 指标名 → 有限数值 |
| `baseline_metrics` | object/null | 否 | 参考基线 |
| `budget_remaining` | object | 是 | `cpu_seconds`、`tokens`、`label_quota` |
| `recommendation` | str | 是 | `no_update` / `retrain_full` / `retrain_recent` / `finetune` / `rollback` / `investigate` |
| `evidence_ids` | list[str] | 是 | 引用的历史证据 |
| `notes` | str | 否 | ≤ 4096 字符 |
| `created_at` | str | 是 | RFC 3339 UTC |

### 5.6 Decision（成员 3 → 成员 4）

| 字段 | 类型 | 必填 | 说明 |
| --- | --- | --- | --- |
| `schema_version` | str | 是 | `"decision.v1"` |
| `decision_id` | str | 是 | `dec-<uuid4>` |
| `task_id` / `stage_id` / `attempt_id` | str | 是 | 定位 |
| `based_on` | str | 是 | `diagnosis_id` |
| `action` | str | 是 | `no_update` / `retrain_full` / `retrain_recent` / `finetune` / `rollback` |
| `params` | object | 是 | 按动作校验，见下表 |
| `rationale` | str | 是 | ≤ 4096 字符 |
| `estimated_cost` | object | 是 | `cpu_seconds`、`tokens`、`label_quota` |
| `created_at` | str | 是 | RFC 3339 UTC |

`params` 允许项：

| action | 允许参数 |
| --- | --- |
| `no_update` | `{}` |
| `retrain_full` | `epochs`, `learning_rate`, `batch_size`, `seed` |
| `retrain_recent` | 上一行 + `window_ratio` (0,1] |
| `finetune` | `epochs`, `learning_rate`, `batch_size`, `seed` |
| `rollback` | `target_artifact_id` 或 `target_checkpoint_sha256` 之一，必须对应当前 `rnd.artifacts` |

非法参数、未知动作、预算为负：拒绝并记 `FAILED_RECOVERABLE`。

### 5.7 CandidateModel（成员 4 → 成员 5）

| 字段 | 类型 | 必填 | 说明 |
| --- | --- | --- | --- |
| `schema_version` | str | 是 | `"candidate-model.v1"` |
| `candidate_id` | str | 是 | `cand-<uuid4>` |
| `task_id` / `stage_id` / `attempt_id` | str | 是 | 定位 |
| `checkpoint_uri` | str | 是 | 相对当前阶段 artifact 目录；禁止绝对路径 |
| `checkpoint_sha256` | str | 是 | `sha256:<hex>` |
| `parent_checkpoint_sha256` | str | 是 | 训练起点 |
| `training_metrics` | object | 是 | 指标名 → 有限数值 |
| `config` | object | 是 | 实际使用的超参数 |
| `seed` | int | 是 | 0..2^32-1 |
| `created_at` | str | 是 | RFC 3339 UTC |

候选模型必须写入新文件，不得覆盖 active 模型。

### 5.8 Evaluation（成员 5 产出并决定门控）

| 字段 | 类型 | 必填 | 说明 |
| --- | --- | --- | --- |
| `schema_version` | str | 是 | `"evaluation.v1"` |
| `evaluation_id` | str | 是 | `eval-<uuid4>` |
| `task_id` / `stage_id` / `attempt_id` | str | 是 | 定位 |
| `candidate_id` | str | 是 | 被评估候选 |
| `baseline_metrics` | object | 是 | active 模型指标 |
| `candidate_metrics` | object | 是 | 候选指标 |
| `metric_name` | str | 是 | 主指标名 |
| `delta` | float | 是 | 已按指标方向归一：`maximize` 为候选−基线，`minimize` 为基线−候选；正值表示更好 |
| `gate` | str | 是 | `promote` / `keep` / `rollback` |
| `reasons` | list[str] | 是 | 每条 ≤ 512 字符 |
| `cost` | object | 是 | `cpu_seconds`、`tokens`、`label_quota` |
| `created_at` | str | 是 | RFC 3339 UTC |

### 5.9 StageRecord（本轮完整记录）

| 字段 | 类型 | 必填 | 说明 |
| --- | --- | --- | --- |
| `schema_version` | str | 是 | `"stage-record.v1"` |
| `record_id` | str | 是 | `rec-<uuid4>` |
| `run_id` / `task_id` / `stage_id` / `attempt_id` | str | 是 | 定位 |
| `bundle_sha256` | str | 是 | canonical 哈希 |
| `diagnosis_id` / `decision_id` | str | 是 | |
| `candidate_id` / `evaluation_id` | str/null | 否 | 无候选时为空 |
| `final_action` | str | 是 | `no_update` / `promote` / `keep` / `rollback` / `failed` |
| `active_before_sha256` | str | 是 | 阶段开始前的 active 哈希 |
| `active_after_sha256` | str | 是 | 阶段结束后的 active 哈希 |
| `timings` | object | 是 | 各步耗时（秒） |
| `events_count` | int | 是 | 本阶段事件数 |
| `created_at` | str | 是 | RFC 3339 UTC |

### 5.10 ActiveModel

`artifacts/active-model.json`：

| 字段 | 类型 | 必填 | 说明 |
| --- | --- | --- | --- |
| `schema_version` | str | 是 | `"active-model.v1"` |
| `task_id` | str | 是 | |
| `checkpoint_uri` | str | 是 | 相对 `artifact_root` |
| `checkpoint_sha256` | str | 是 | |
| `stage_id` | str | 是 | 首次激活时的阶段 |
| `activated_at` | str | 是 | RFC 3339 UTC |
| `reason` | str | 是 | `initial` / `promote` / `rollback` |
| `previous` | object/null | 否 | `{checkpoint_uri, checkpoint_sha256, stage_id}` |

回滚只修改指针，不删除任何历史阶段目录。

---

## 6. 哈希与幂等

### 6.1 source_sha256

对阶段目录内参与加载的文件（`profile.json`、`features.jsonl`、`labels.jsonl`，若存在）计算：

```text
按相对路径字节序排序，逐行写入：
  <relpath> \0 <文件字节的 sha256 hex> \n
再对整个字节流取 sha256
```

结果格式：`sha256:<hex>`。

### 6.2 canonical_sha256

步骤：

1. 拒绝任何 `NaN`/`Inf`；`-0.0` 归一为 `0.0`。
2. 按 `id` 的 UTF-8 字节序排序。
3. 逐样本写入：
   ```text
   id UTF-8 \x1f
   [每个特征按 schema 顺序 struct.pack(">d", value)] \x1f
   target：存在则 struct.pack(">d", value)，否则 b"-" \x1f
   group_id UTF-8 \x1e
   ```
4. 前缀为元数据的紧凑 JSON（键排序、UTF-8、无空格）：
   `{schema_version, task_id, stage_id, task_type, feature_schema_id, feature_count, classes}`
5. 对整个字节流取 sha256，格式 `sha256:<hex>`。

性质：与行序、JSON 键序、空白、路径无关；同内容必同哈希。

### 6.3 幂等键与复用

- 阶段内幂等键：`(task_id, stage_id)`。
- 已有正式阶段目录时：
  - `canonical_sha256` 相同 → 复用，不重训、不重复提交。
  - 不同 → `FatalContractError`，立即停止（exit 2）。
- 跨 run 不做自动复用；调用方通过新的 `run_id` 显式开始新一轮。
- 证据 ID（`diag-`/`dec-`/`cand-`/`eval-`/`rec-`）使用 UUID4，不参与幂等判断。

---

## 7. 阶段来源与本地模拟

### 7.1 StageSource 协议

```python
class StageSource(Protocol):
    def next(self) -> StageDescriptor | None: ...
    def read(self, descriptor: StageDescriptor) -> mapping[str, bytes]: ...
    def ack(self, stage_id: str, status: str) -> None: ...
```

- `next()` 返回 `None` 表示没有更多阶段（正常结束）。
- `read()` 返回相对路径 → 字节；不解析业务字段。
- `ack()` 在平台要求时确认；本地实现默认 no-op 或写 `acks.jsonl`。
- 实现必须幂等：重复 `next()` 不跳阶段；重复 `ack()` 不报错。

### 7.2 LocalManifestSource

```text
data/stages/<task>/
  manifest.json
  stage-01/
    profile.json
    features.jsonl
    labels.jsonl
  stage-02/
    ...
```

`manifest.json`：

```json
{
  "schema_version": "stage-manifest.v1",
  "task_id": "A",
  "stages": [
    {"stage_id": "stage-01", "dir": "stage-01"},
    {"stage_id": "stage-02", "dir": "stage-02"}
  ]
}
```

规则：

- `stages` 有序，按顺序返回。
- `stage_id` 唯一；重复直接拒绝。
- `dir` 必须 resolve 到 `stage_root` 内，禁止绝对路径、`..`、符号链接。
- 文件必须为普通文件；单文件 ≤ 128 MiB；总行数受 `profile.limits.max_rows` 限制。

### 7.3 WorkspaceV2Source（官方适配器）

官方 `contest_harness 1.0.4` 已明确 `contest.harness.workspace.v2` 协议，因此正式的
平台适配器是 `zhixun_harness/sources.py::WorkspaceV2Source`：

- 平台通过 stdin 的 `start` 帧驱动一轮，提供 `prompt`、`dataset_key`、`round_index`、
  `task_contract`、父检查点摘要和可写 `/workspace`。
- 训练投影为 `/data/train.jsonl`，公开验证投影为 `/data/validation.jsonl`；
  `WorkspaceV2Source` 把两者分别转成内部训练 bundle 和评测 bundle。
- 父产物来自 `/previous`，状态来自 `/state`，跨轮持久化只能通过
  `rnd.save_state()`；`/workspace` 不跨轮保留。
- 候选通过 `rnd.submit_candidate()` 提交，选择结果通过
  `rnd.finish(selected_artifact_id=...)` 交回平台；平台持有 active model，
  本地不再维护 `active-model.json` 作为权威。
- 每轮至少一次真实 `rnd.model()` 调用，`model_call_ids` 必须是真实回执号。

`PlatformStageSource` 只保留为未来平台协议的显式桩，未实现时直接抛
`NotImplementedError`，避免调用方误以为已经支持。

### 7.4 本地构建脚本

`scripts/build_stages_a.py`：

```bash
python scripts/build_stages_a.py \
  --package cpu-a-guide-20261110-v1 \
  --output data/stages/a \
  [--limit 6]
```

- stage-01 ← `public/train.jsonl`（batch1）
- stage-02 ← `public/validation.jsonl`（batch2）
- 输出 `features.jsonl`（仅 `id` + `inputs`）、`labels.jsonl`（`id` + `answer` + `group`）、`profile.json`、`manifest.json`。
- `--limit N` 用于生成提交入库的极小夹具。
- 该脚本是开发工具，不在运行时路径。

---

## 8. 编解码与校验

### 8.1 处理顺序（fail-closed）

1. 路径与文件检查（普通文件、大小、UTF-8）。
2. JSON 严格解析（重复键、非有限数、深度 ≤16、特征数 ≤4096）。
3. 逐行结构检查：`id`、`inputs`、`answer`（若标签可用）。
4. 特征展开并核对 `feature_schema_id`/`feature_count`。
5. 禁用字段检查（profile + dev overlay）。
6. 标签类型与取值检查。
7. 分组检查：组信息存在；`group_split=disjoint` 时训练/评测组不重叠。
8. 计算 `source_sha256` 与 `canonical_sha256`。
9. 构造 `StageBundle`。

任一步失败：抛出 `StageDataError`，附带行号、字段、原因；不产生部分 bundle，不训练。

### 8.2 通用校验（与任务无关）

- `id` 非空、唯一。
- `inputs` 只含有限数值；禁止布尔、字符串、嵌套深度超限。
- 特征路径序列在阶段内一致。
- 维度等于 `feature_count`。
- 分类标签属于 `classes`；回归标签为有限非负数。
- `features.jsonl` 与 `labels.jsonl` 的 `id` 集合一致。
- 分组字段存在且非空；组合键按声明顺序拼接。
- 同一阶段内若按组切分训练/评测，`group_split=disjoint` 要求两组交集为空；跨阶段历史回放允许组重复。
- `label_status=hidden` 时不得读到标签。

### 8.3 Dev overlay

A/B/C 的维度、类别、禁用字段放在 `configs/dev/*.json`，只在开发期与 profile 合并：

```json
{
  "task_id": "A",
  "package_dir": "cpu-a-guide-20261110-v1",
  "profile": "baselines/configs/A.json",
  "initial_checkpoint": "checkpoint.npz",
  "feature_count": 128,
  "forbidden_input_fields": ["id", "source_batch", "source_index"],
  "group_field": "meta.source_batch"
}
```

合并规则：overlay 只补充 profile 未声明或本地需要的字段；**冲突时以 profile 为准**。正式运行不加载 overlay。

参考包的 `group_field`：A 用 `"meta.source_batch"`；B 用 `"meta.bearing"`；C 用 `["meta.source_subset","meta.unit"]`。

### 8.4 已知任务的禁用输入字段（dev）

| 任务 | 禁用字段（归一化后匹配） |
| --- | --- |
| A | `id`、`source_batch`、`source_index` |
| B | `bearing`、`snapshot`、`totalsnapshots`、`rul`、`remainingusefullife` |
| C | `unit`、`cycle`、`source_subset`、`rul`、`remainingusefullife` |

通用默认还包含：`answer`、`target`、`label`、`truerul`、`normalizedrul`。

---

## 9. ArtifactStore

### 9.1 目录布局

```text
artifacts/
  active-model.json
  initial/
    checkpoint.npz                 # 首次运行复制进来，不可变
    source.json                    # 记录来源
  runs/
    <run_id>/
      run.json
      events.jsonl
      .stage-<hash16>.tmp-<uuid>/  # 写入中
      stage-<hash16>/              # 已提交，只读
        manifest.json              # descriptor + bundle 元数据 + 哈希
        bundle.json                # StageBundle 元数据（无数组）
        bundle.npz                 # 可选，embed_arrays=true 时
        diagnosis.json
        decision.json
        evaluation.json
        record.json
        events.jsonl
        candidate/
          checkpoint.npz
          candidate.json
      stage-<hash16>/...
```

- `stage-<hash16>` 中的 hash 为 `sha256(stage_id)[:16]`，避免任意 stage_id 直接进入文件名。
- 正式阶段目录一旦存在，永不修改。
- 本次运行开始时若 `artifacts/initial/checkpoint.npz` 不存在，则按优先级解析初始检查点：dev overlay 的 `initial_checkpoint`（相对 `package_dir`）> `harness.json` 的 `initial_checkpoint`（路径）> 报错 `ConfigError`。复制后校验哈希并写入 `initial/source.json`（`origin`、`sha256`、`copied_at`）。

### 9.2 事务 API

```python
store = ArtifactStore(root=artifact_root, run_id=run_id)
txn = store.begin_stage(descriptor, bundle)
txn.write_json("diagnosis.json", diagnosis.to_dict())
txn.write_json("decision.json", decision.to_dict())
txn.write_candidate(candidate, checkpoint_bytes)
txn.write_json("evaluation.json", evaluation.to_dict())
txn.write_json("record.json", record.to_dict())
txn.append_event(event)
txn.commit()          # 原子提交
txn.abort()           # 删除临时目录
```

### 9.3 提交协议

1. 所有文件写入 `.tmp-<uuid>` 目录，使用 `"xb"` 防覆盖。
2. 每个文件 `flush()` + `os.fsync()`。
3. `os.fsync()` 临时目录。
4. `os.replace(tmp, final)`（同一文件系统内原子）。
5. `os.fsync()` 父目录。
6. 若 `final` 已存在：canonical 相同 → 删除 tmp 并复用；不同 → `FatalContractError`。

### 9.4 active-model 更新

1. 仅在阶段 `COMMITTED` 之后调用。
2. 写 `active-model.json.tmp-<uuid>`，fsync。
3. `os.replace()` 到 `active-model.json`，fsync 父目录。
4. 同时写入 `previous`，保留上一版本指针。
5. 回滚 = 写一个新的 active-model，`reason="rollback"`，指向 `previous`；不删除候选或失败阶段。

### 9.5 并发与崩溃

- 同一 `artifact_root` 只允许一个写进程；用 `filelock`（若可用）或独占锁文件；冲突时 exit 2。
- 启动时清理/忽略遗留 `.tmp-*` 目录；不自动删除，记录到 `run.json`。
- 进程被 SIGKILL 后，最多丢失未提交的临时阶段；已提交阶段和 active-model 保持有效。

---

## 10. 主循环与状态机

### 10.1 状态

| 状态 | 含义 |
| --- | --- |
| `WAITING_STAGE` | 等待下一个阶段 |
| `STAGE_READY` | 已取得 descriptor 与原始文件 |
| `VALIDATED` | 校验通过，bundle 已生成 |
| `CURRENT_MODEL_EVALUATED` | active 模型在本阶段上的指标已得到 |
| `DIAGNOSED` | 诊断报告已产出 |
| `EXPERIMENTING` | 候选训练中 |
| `CANDIDATE_EVALUATED` | 候选指标已得到 |
| `DECIDED` | 门控决定已产出 |
| `COMMITTED` | 阶段 artifact 已原子提交 |
| `FAILED_RECOVERABLE` | 当前阶段失败，系统可继续 |
| `FAILED_FATAL` | 契约/配置致命错误，停止 |
| `FINISHED` | 正常结束或预算耗尽/中断 |

### 10.2 主流程

```text
while True:
    descriptor = source.next()
    if descriptor is None: FINISHED; break
    if max_stages reached: FINISHED(budget); break

    raw = source.read(descriptor)                 # 重试
    bundle = codec.normalize(raw, profile)        # 校验
    if store.has_committed(bundle):               # 幂等
        source.ack(stage_id, "reused"); continue

    current = evaluator.evaluate(active, bundle)
    diagnosis = pipeline.diagnose(bundle, current, budget)
    decision = pipeline.decide(diagnosis, history, budget)

    if decision.action == "no_update":
        record no_update; commit; ack; continue

    if decision.action == "rollback":
        store.rollback_active(...); commit; ack; continue

    candidate = pipeline.train_candidate(bundle, decision, active)
    evaluation = pipeline.evaluate_and_commit(bundle, candidate, active, gate)
    if evaluation.gate == "promote": store.activate(candidate)
    elif evaluation.gate == "rollback": store.rollback_active(...)
    else: keep active

    record = build_record(...)
    txn.commit(); source.ack(stage_id, "done")
```

### 10.3 转移与异常

| 当前 | 事件 | 下一状态 |
| --- | --- | --- |
| `WAITING_STAGE` | 取得 descriptor | `STAGE_READY` |
| `STAGE_READY` | 读取成功 | `VALIDATED` |
| `STAGE_READY` | 读取失败（重试耗尽） | `FAILED_RECOVERABLE` |
| `VALIDATED` | 当前模型评估完成 | `CURRENT_MODEL_EVALUATED` |
| `CURRENT_MODEL_EVALUATED` | 诊断完成 | `DIAGNOSED` |
| `DIAGNOSED` | 决策 `no_update`/`rollback` | `DECIDED` |
| `DIAGNOSED` | 决策需要训练 | `EXPERIMENTING` |
| `EXPERIMENTING` | 候选完成 | `CANDIDATE_EVALUATED` |
| `EXPERIMENTING` | 训练失败/超时 | `FAILED_RECOVERABLE` |
| `CANDIDATE_EVALUATED` | 门控完成 | `DECIDED` |
| `DECIDED` | artifact 提交成功 | `COMMITTED` |
| `DECIDED` | 提交失败 | `FAILED_RECOVERABLE` |
| `COMMITTED` | 下一阶段存在 | `WAITING_STAGE` |
| `COMMITTED` | 无下一阶段/预算耗尽 | `FINISHED` |
| 任意 | 哈希冲突/非法配置/路径越界 | `FAILED_FATAL` |
| 任意 | SIGTERM/SIGINT | `FINISHED(interrupted)` |

### 10.4 阶段失败后的行为

- 默认 `on_stage_error="continue"`：记录失败，保留 active 模型，继续下一阶段。
- 可配置 `"stop"`：遇到 `FAILED_RECOVERABLE` 即结束。
- `FAILED_FATAL` 永远停止。

### 10.5 信号与退出码

| 退出码 | 含义 |
| --- | --- |
| 0 | 正常结束、预算耗尽、中断（已安全落盘） |
| 2 | 配置/契约致命错误 |
| 3 | 预算熔断 |
| 4 | 未预期异常（保留 traceback） |

SIGTERM/SIGINT：设置停止标志，只在阶段边界生效；若正在写临时事务则 `abort()`；不打断已开始的 `os.replace`。

---

## 11. 成员接口（Pipeline）

```python
class Pipeline(Protocol):
    def diagnose(self, bundle: StageBundle, current_eval: dict, budget: BudgetState) -> Diagnosis: ...
    def decide(self, diagnosis: Diagnosis, history: list[StageRecord], budget: BudgetState) -> Decision: ...
    def train_candidate(self, bundle: StageBundle, decision: Decision,
                        active: ActiveModel, budget: BudgetState) -> CandidateModel | None: ...
    def evaluate_and_commit(self, bundle: StageBundle, candidate: CandidateModel | None,
                            active: ActiveModel, gate: GateConfig) -> Evaluation: ...


class Evaluator(Protocol):
    # 当前模型在阶段开始时的评估；诊断前调用，属于成员 5 的评测实现
    def evaluate(self, model: ActiveModel, bundle: StageBundle) -> dict[str, float]: ...
```

说明：

- `evaluate_and_commit` 只返回 `Evaluation`，**不修改** artifact 或 active 指针；上线/回滚由 orchestrator 根据 `Evaluation.gate` 调用 `ArtifactStore` 完成。
- `Evaluator.evaluate` 用于 `CURRENT_MODEL_EVALUATED` 和候选对比；隐藏标签场景由 oracle 实现，Harness 不直接读。

不变量：

- 输入对象只读，成员不得原地修改。
- 返回值必须是已通过契约校验的结构化对象。
- 错误显式抛出 `TrainingError` / `EvaluationError` / `DecisionError`，不得吞掉。
- 不得直接修改 active 模型或 artifact 目录；只能通过 `ArtifactStore`。
- 每个成员实现必须幂等：相同 `(task_id, stage_id, attempt_id)` 重放不产生额外副作用。
- 证据 ID 唯一；历史记录只追加。

---

## 12. 预算与安全护栏

### 12.1 预算字段

```json
{
  "cpu_seconds": 3600,
  "tokens": 0,
  "label_quota": 0,
  "max_stages": 60,
  "max_candidates_per_stage": 2,
  "eval_timeout_seconds": 300,
  "phase_wallclock_seconds": 1800
}
```

- 训练/评估前检查；超限抛 `BudgetExceeded`，干净收尾。
- 计数在阶段内累加，跨阶段累计。
- 所有上限来自配置；代码只校验上限合法（≥0、≤硬上限）。
- `null` 表示不限制；`0` 表示不允许消耗（本地 fake 阶段用 `tokens=0`、`label_quota=0` 即禁止外部调用和标注）。
- `cpu_seconds` 用 `time.process_time()` 计量；`phase_wallclock_seconds` 用 `time.monotonic()` 计量。

### 12.2 安全护栏

- 输入目录只读语义；不写、不删、不改名。
- 只有 `artifact_root` 可写；路径必须 resolve 在根内。
- 拒绝符号链接、绝对路径、`..`、空路径。
- 单文件 ≤ 128 MiB；行数 ≤ `limits.max_rows`；特征 ≤ 4096；嵌套 ≤ 16。
- 不创建网络连接；不读环境变量中的外部 API Key。
- 子进程训练：固定 argv、`shell=False`、超时后终止进程组。
- 所有异常写结构化事件；不把 traceback 写入 `record.json`。

---

## 13. 配置与 CLI

### 13.1 平台中立配置 `configs/harness.json`

```json
{
  "schema_version": "harness-config.v1",
  "pipeline": "fakes",
  "gate": {
    "metric_name": "macro_f1",
    "direction": "maximize",
    "min_delta": 0.0,
    "max_regression": 0.02,
    "min_samples": 2
  },
  "budget": {
    "cpu_seconds": 3600,
    "tokens": 0,
    "label_quota": 0,
    "max_stages": 60,
    "max_candidates_per_stage": 2,
    "eval_timeout_seconds": 300,
    "phase_wallclock_seconds": 1800
  },
  "retry": {
    "read_attempts": 3,
    "backoff_seconds": 0.5,
    "on_stage_error": "continue"
  },
  "artifact": {
    "embed_arrays": false,
    "cleanup_tmp_on_start": false
  },
  "initial_checkpoint": null
}
```

### 13.2 开发 overlay `configs/dev/a.json`

```json
{
  "schema_version": "dev-profile.v1",
  "task_id": "A",
  "package_dir": "cpu-a-guide-20261110-v1",
  "profile": "baselines/configs/A.json",
  "initial_checkpoint": "checkpoint.npz",
  "feature_count": 128,
  "group_field": "meta.source_batch",
  "forbidden_input_fields": ["id", "source_batch", "source_index"]
}
```

### 13.3 CLI

```bash
python -m zhixun_harness.orchestrator \
  --stage-root data/stages/a \
  --artifact-root artifacts \
  --run-id run-001 \
  [--config configs/harness.json] \
  [--dev-profile configs/dev/a.json] \
  [--max-stages 2] \
  [--log-level INFO]
```

- `--stage-root`、`--artifact-root`、`--run-id` 必填。
- `--task` 不再必填；`task_id` 来自 manifest。
- 配置校验失败：exit 2，不创建 artifact。

---

## 14. 日志与可观测性

- `events.jsonl` 每行一个事件：

```json
{
  "ts": "2026-11-16T08:00:00.123456+00:00",
  "run_id": "run-001",
  "task_id": "A",
  "stage_id": "stage-01",
  "attempt_id": "a1",
  "state": "VALIDATED",
  "event": "bundle_ready",
  "level": "info",
  "data": {"sample_count": 445, "canonical_sha256": "sha256:..."}
}
```

- `append` 后立即 flush；提交时 fsync。
- 人类日志走 stderr，使用标准 `logging`；不写 stdout，避免污染管道。
- `run.json` 记录：启动参数、配置哈希、版本、初始 active、最终 active、退出码、汇总计数。
- 不记录原始特征、标签、API Key 或隐藏数据。

---

## 15. 错误分类与恢复矩阵

| 异常 | 分类 | 处理 | 状态 |
| --- | --- | --- | --- |
| 文件不存在/被占用/读超时 | transient | 重试 3 次，退避 0.5s | `FAILED_RECOVERABLE` |
| 文件过大/非普通文件/符号链接 | terminal | 拒绝 | `FAILED_FATAL` |
| JSON 非法/重复键/非有限数 | terminal | 当前阶段失败，不训练 | `FAILED_RECOVERABLE` |
| 维度/标签/禁用字段/组泄漏 | terminal | 当前阶段失败，不训练 | `FAILED_RECOVERABLE` |
| 同 stage 哈希冲突 | terminal | 立即停止 | `FAILED_FATAL` |
| 训练失败/超时 | terminal（非幂等，不自动重试） | 无候选，保留 active | `FAILED_RECOVERABLE` |
| 评估失败/超时 | transient | 可在超时预算内重试一次；仍失败则禁止上线 | `FAILED_RECOVERABLE` |
| 预算耗尽 | terminal | 干净收尾 | `FINISHED` |
| 配置非法/路径越界 | terminal | 启动前拒绝 | exit 2 |
| SIGTERM/SIGINT | signal | abort tmp，保留已提交 | `FINISHED` |

"已开始、效果未知、非幂等"的操作（例如训练已启动但未拿到候选）**不盲目重试**；记录为失败，保留旧模型。

---

## 16. 测试矩阵

### 16.1 单元测试

| 文件 | 覆盖 |
| --- | --- |
| `tests/test_contracts.py` | 字段缺失、未知字段、非法枚举、类型错误、ID 约束 |
| `tests/test_hashing.py` | 行序/键序/空白无关；-0.0 归一；NaN 拒绝；同内容同哈希 |
| `tests/test_codec.py` | 正常、空文件、重复 id、维度错、NaN/Inf、禁用字段、组泄漏、hidden 标签 |
| `tests/test_sources.py` | manifest 顺序、重复 stage_id、路径越界、符号链接、结束信号 |
| `tests/test_store.py` | 原子提交、幂等复用、哈希冲突、abort 无残留、active-model 原子替换、崩溃恢复 |
| `tests/test_budget.py` | 各项预算累加与熔断 |
| `tests/test_orchestrator.py` | 2 阶段 happy path、no_update、训练失败、门控拒绝、rollback、预算熔断、SIGTERM |
| `tests/test_build_stages_a.py` | 从参考包生成 2 阶段并校验 |
| `tests/test_dev_integration.py` | 参考包存在时跑通 2 阶段闭环；不存在时 `skip` |

### 16.2 属性测试（Hypothesis，建议）

- 任意行序的同一数据 → 同一 canonical 哈希。
- 任意 JSON 键序/空白 → 同一 canonical 哈希。
- 任何 NaN/Inf → 校验必然失败。
- 任意重复 id → 校验必然失败。

### 16.3 夹具

```text
tests/fixtures/stages_a/     # 每阶段 6 条，提交入库
tests/fixtures/stages_b_tiny/
tests/fixtures/stages_c_tiny/
```

- 核心测试只用夹具，不读三个参考包。
- `tests/test_dev_integration.py` 才读参考包，且在包不存在时 `skip`。

### 16.4 运行方式

```bash
python -m pytest -q
```

目标环境：WSL Ubuntu 或 Linux 容器；最终在 `linux/amd64`、CPU、断网、只读输入下验收。

---

## 17. 参考实现与取舍（成员 1）

| 项目 | 借鉴点 | 是否引入依赖 |
| --- | --- | --- |
| Temporal | workflow/activity 分离、每步 journal、稳定幂等键、重试分类、信号 | 否，只借语义 |
| DVC | 内容寻址、stage 缓存、哈希指针 | 否 |
| MLflow Model Registry | alias-first、`previous` 指针、确定性回滚 | 否 |
| Pandera / Great Expectations | 声明式校验、一次报所有错、fail-closed | 否，手写校验器 |
| Kedro / Prefect | 数据目录抽象、任务状态、retry policy | 否 |
| `python-statemachine` | 显式状态转移表 | 否 |
| `eventsourcing` | append-only 事件日志、乐观并发 | 否 |
| `filelock` | 跨平台文件锁 | 可选 |
| `tenacity` | 退避重试 | 可选 |
| `structlog` | 结构化 JSON 日志 | 可选；标准库可实现 |
| `Hypothesis` | 属性测试 | 是（dev） |
| `distroless` / `hadolint` / `dive` | 最小离线镜像、Dockerfile 检查、体积分析 | 交付阶段 |

原则：正式镜像保持最小依赖（numpy + 标准库）；其余只借设计。

---

## 18. 待组委会确认（TBD）

官方 workspace.v2 SDK 已解决大部分问题，剩余项不再阻塞显式适配器：

| 原问题 | 现状 |
| --- | --- |
| 阶段目录路径 | `/data/train.jsonl`、`/data/validation.jsonl`、`/previous`、`/state`、`/workspace`、`/model` |
| 阶段清单与 schema | 由 SDK `start` 帧的 `task_contract` 和 `public_projection` 提供 |
| `stage_id` 来源 | `dataset_key` + `round_index`；内部使用 `round-<index>` |
| 数据完整性 | 平台写入投影文件；SDK 逐行读取 |
| 标签可见性 | 公开训练/验证集带 `answer`；隐藏集不挂载 |
| 下一阶段通知 | 平台再次调用 `train(rnd)`；没有本地循环 |
| 结束信号 | `rnd.finish()` + `round_ack` |
| 超时/重试 | 由平台控制；SDK 侧只做本地校验 |
| 结果写回 | `submit_candidate` + `save_state` + `finish` |
| 资源预算 | `rnd.max_output_tokens`、`rnd.limits`；SDK 1.0.4 单候选/状态最多 64 文件、512 MiB |
| ack / 回执 | `finish` 返回 `round_ack`；候选返回 `artifact_id` 和验证摘要 |

仍需向平台确认：`rnd.task` 中完整字段名、`rnd.limits` 的具体字节上限、真实宿主的
超时行为。这些只影响适配器细节，不改变内部契约。

---

## 19. 交付物清单

```text
docs/superpowers/specs/2026-11-16-harness-member1-design.md   # 本文
docs/interface-contract.md                                    # 成员 2–5 接口
docs/references-member1.md                                    # 参考实现取舍
zhixun_harness/__init__.py
zhixun_harness/errors.py
zhixun_harness/contracts.py
zhixun_harness/taskspec.py
zhixun_harness/codec.py
zhixun_harness/hashing.py
zhixun_harness/sources.py
zhixun_harness/store.py
zhixun_harness/budget.py
zhixun_harness/pipeline.py
zhixun_harness/fakes.py
zhixun_harness/orchestrator.py
zhixun_harness/round_runner.py          # workspace.v2 一轮适配
zhixun_harness/platform.py
zhixun_harness/config.py
contest_submission.py                    # 官方 train/predict 入口
app/harness.py                           # 复制到镜像 /app/harness.py
pyproject.toml
Dockerfile
contest-harness-sdk-1.0.4.tar.gz
scripts/build_stages_a.py
configs/harness.json
configs/dev/a.json
configs/dev/b.json
configs/dev/c.json
tests/...（见第 16 节；含 test_round_protocol.py）
docs/interface-contract.md
docs/references-member1.md
README.md（更新目录结构与运行命令）
.gitignore（增加 artifacts/）
```

---

## 20. 决策记录

| 编号 | 决策 | 状态 |
| --- | --- | --- |
| D-01 | 顶层 `zhixun_harness/` 包 | 已定 |
| D-02 | 契约优先 + 进程内接口 | 已定 |
| D-03 | 三个包仅为参考数据 | 已定 |
| D-04 | 任务由 profile 声明，核心不认 A/B/C | 已定 |
| D-05 | 本地先做 2 阶段 | 已定 |
| D-06 | StageBundle 携带 numpy 数组 | 已定 |
| D-07 | 本地标签 available，保留 oracle 语义 | 已定 |
| D-08 | 第一版只接 fake Pipeline | 已定 |
| D-09 | 包与总结冲突以包为准 | 已定 |
| D-10 | artifact 根下 `runs/<run_id>/`，active-model 在根 | 待评审 |
| D-11 | 阶段目录名用 `stage-<sha256(stage_id)[:16]>` | 待评审 |
| D-12 | 幂等键 `(task_id, stage_id)`，跨 run 不复用 | 待评审 |

---

## 21. 验收标准

1. `python -m pytest -q` 全绿。
2. `python scripts/build_stages_a.py --limit 6` 生成夹具，`test_build_stages_a` 通过。
3. `python -m zhixun_harness.orchestrator --stage-root tests/fixtures/stages_a --artifact-root /tmp/art --run-id demo --max-stages 2` 正常结束。
4. 两个阶段都产生 `bundle.json`、`diagnosis.json`、`decision.json`、`evaluation.json`、`record.json`、`events.jsonl`。
5. 重复运行同一 `stage_id` 且内容相同 → 复用，不重复训练。
6. 篡改同一 `stage_id` 的内容 → exit 2，且不修改 active-model。
7. 模拟训练失败 → active-model 不变，失败被记录。
8. 发送 SIGTERM → 无半提交阶段，exit 0。
9. 输入目录只读、断网、非 root 下运行通过。
10. 所有路径处理使用 `pathlib.Path`，无 Windows 专属依赖。

---

## 22. workspace.v2 对齐修订（2026-11-17）

本节保留首次接入时的历史背景；SDK 1.0.4 的分块上传变更见第 23 节。

官方 `contest_harness 1.0.2` 下载包与本规格的差异，以及本次落地的处理：

| 原设计 | workspace.v2 事实 | 修订 |
| --- | --- | --- |
| 自建阶段循环 `source.next()` | 平台调用 `train(rnd)`，每轮一个 start 帧 | 保留本地 `LocalManifestSource` 供离线开发；正式路径改走 `WorkspaceV2Source` + `round_runner` |
| `artifacts/active-model.json` 是权威 | 平台持有父检查点，`finish(selected_artifact_id)` 决定选择 | 本地 store 仅用于离线开发；正式运行以平台父产物为准 |
| 跨轮 artifact 目录常驻 | `/workspace` 本轮可写，跨轮只剩 `/state` 和平台 history | 证据写 `save_state`、`finish` 记录和 stderr/SwanLab；候选受 64 文件 / 64 MiB 限制 |
| `harness/` 包名 | 镜像必须存在 `/app/harness.py` | 内部包改名 `zhixun_harness/` |
| 自建 `predict` 循环 | SDK `serve_predictions` 逐行读写并 flush | `contest_submission.predict` 只返回结果，由 SDK 负责协议 |
| 每轮不需要模型调用 | `finish` 要求真实 `model_call_ids` | 假决策通过 `RoundModelGateway` 调一次 `rnd.model()`；正式决策由成员 3 替换 |
| `PlatformStageSource` 预留 | 官方协议已明确 | 改由 `WorkspaceV2Source` 实现；`PlatformStageSource` 仅保留为未来桩 |

已落地的接口与验证：

- `zhixun_harness/round_runner.py`：一轮 `train(rnd)` 的完整适配。
- `tests/test_round_protocol.py`：用真实 `contest_harness` 协议帧跑通 `train`。
- `app/harness.py` + `contest_submission.py`：官方入口与选手模块。
- `Dockerfile`：Python 3.12 + 官方 SDK + 本包，非 root 运行。

## 23. SDK 1.0.4 分块上传修订（2026-11-22）

- 镜像安装 `contest-harness-sdk-1.0.4.tar.gz`，继续使用
  `contest.harness.workspace.v2` 协议。
- 宿主必须在 `start.resource_limits.artifact_upload_protocol` 中声明
  `chunked-v1`；SDK 不再退回旧版单条消息内联上传。
- 候选和状态分别通过 `upload_begin_request`、`upload_chunk_request`、
  `upload_finish_request` 上传，完成后再发送 `candidate_request` 或
  `state_request`。
- 每个候选或状态最多 64 个文件、合计 512 MiB；每轮候选累计 1 GiB、状态
  累计 512 MiB，具体仍以 `rnd.limits` 为准。
- `tests/test_round_protocol.py` 的内存宿主和子进程宿主均验证分块 offset、
  文件大小和 SHA-256。
