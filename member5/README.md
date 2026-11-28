# 成员 5：评测、上线门控与实验记录

## 已提供

- `member5.evaluator:EvaluationGate`：校验候选检查点及父哈希，在独立验证集上
  比较当前模型和候选模型。
- `compare_metrics()`：统一处理最大化/最小化指标，检查主指标提升和辅助指标
  最大退化。
- `member5.record:write_round_record()`：每轮生成不可覆盖的本地 JSON 记录，
  默认离线；设置 `SWANLAB_MODE=online` 时尝试上传 SwanLab。
- `member5.robustness`：按分组确定性拆分验证集，分别比较父模型和候选模型。
  `worst_group` 是最多退化的组（回归审计），`worst_absolute_group` 是候选下
  绝对指标最差的组（平台 worst 口径）。它只记录稳健性风险，不擅自改变现有
  `promote / rollback` 结论。
- `member5.metrics`：生成分类逐类 precision/recall/F1、Macro 指标和混淆矩阵；
  回归额外记录最差批次 RMSE/MAE；C 还按成员4同口径计算 NASA 非对称分数
  （`exp(-d/13)-1` / `exp(d/10)-1`，指数裁剪到 ±50 后取平均）。NASA 作为
  辅助指标参与门控，默认相对容差 2%（`gate.max_regression`）；成员4 在逐
  epoch/跨计划选择时也优先 gate-safe 候选。其余详细指标保存在独立审计文件中。
- `member5.audit`：检查父/候选哈希、真实调用 ID、指标方向、门控理由和
  自适应候选试验，并生成 Markdown、CSV、JSON 三种验收汇总。
- `member5.offline_acceptance`：分离断网运行状态和 SwanLab 公开证据，
  无需人工输入即可给出 `pass / fail`。
- `member5.swanlab_export`：比赛运行结束后，在联网机器上把本地 JSON 单独
  导出为公开实验，不进入比赛运行流程。SwanLab config 和 metrics 同时带上
  成员4的训练诊断：选中的计划、窗口/加权/SWA/anchor、剪枝 `keep`、训练行数、
  实际 epoch、最差组变化和选择分数。
- `scripts/run_local_harness_e2e.py`：无网络运行 A/B/C 本地全系统试跑。
  第 1 轮可触发真实成员 4 候选，后续轮次验证 `no_update`。记录标记为
  `local-simulated`，SwanLab 导出器会拒绝发布。
- `member5.evidence_bundle`：把多个证据目录合并为 A/B/C 总表，检查重复记录、
  来源类型和 Git 提交是否一致。
- `member5.swanlab_reconcile`：SwanLab 上传完成后，把公开链接回填到本地 JSON
  并重新计算验收状态。

门控不会修改当前模型。候选失败或未达到门控要求时返回 `rollback`，编排器继续
选择父产物。

## 配置

`configs/harness.full.json` 已注册成员二诊断、成员三决策、成员四训练和本模块
门控。`configs/harness.member5.json` 保留为评测模块的独立调试入口。
成员 5 的调试配置同时启用两折稳健性审计。稳健性审计失败只写入门控理由和
`members/evaluation/`，不会代替正式门控规则。

自适应候选的成员 4 应在 `candidate.config` 中保留：

- `selected_plan`
- `trials`
- 每个 trial 的 `best_epoch`、`training_rows`、`best_metrics`
- 剪枝迁移信息、历史回放行数、跨轮策略统计和失败原因

成员 5 验收工具会把这些字段整理进总表；字段缺失会标记为错误或警告。

## 必需测试

```bash
python -m pytest -q tests/test_member5_gate.py tests/test_member5_records.py
```

测试使用真实 A 基线检查点验证加载，使用注入的轻量评测函数覆盖提升、退化和
检查点损坏分支，不访问外网。

新增验收和稳健性测试：

```bash
python -m pytest -q tests/test_member5_audit.py tests/test_member5_robustness.py
```

## 生成验收报告

平台轮次结束后，把 `/workspace/records` 目录导出到开发机，然后运行：

```bash
python scripts/member5_report.py \
  --records /workspace/records \
  --output /workspace/member5-report
```

如果 Windows 虚拟环境不能直接执行脚本文件，可改用：

```powershell
python -X utf8 -m member5.report_cli --records .\records --output .\member5-report
```

输出：

```text
round-summary.json
round-summary.md
round-summary.csv
```

启用了 `detailed_metrics` 后，每轮还会生成：

```text
/workspace/records/member5-metrics/metrics-<task>-<round>.json
```

该文件包含父模型和候选模型的逐类 recall、precision、F1、混淆矩阵或回归
最差批次。汇总 Markdown 会自动附上分类混淆矩阵。

如果用于提交前门禁，可增加 `--fail-on-error`。该选项只根据证据完整性和
标准字段判断，不会把尚未运行的平台实验当成已通过。

## 离线无人干预验收

镜像已包含 SwanLab 依赖。比赛运行默认保持离线；平台需要在线记录时设置
`SWANLAB_MODE=online` 并提供 `SWANLAB_API_KEY`。运行结束后执行：

```bash
python -m member5.offline_acceptance \
  --records /workspace/records \
  --output /workspace/member5-report \
  --min-rounds 10 \
  --require-action no_update \
  --require-action promote \
  --require-action reject
```

`--require-action` 既可以写决策动作（`adaptive_update`、`rollback` …），也可以写最终结果。
最终结果（`final_action`）的含义：

| 值 | 含义 |
| --- | --- |
| `no_update` | 本轮不训练，保留父模型 |
| `promote` | 候选通过门控并上线 |
| `keep` | 候选不比父模型差，但没有足够提升；保留父模型（门控 `keep`） |
| `reject` | 候选更差或辅助指标退化，被拒绝；保留父模型（门控 `rollback`） |
| `rollback` | 只指恢复一个曾经上线的历史模型（门控在当前验证集上复核过） |
| `failed` | 流程异常，保留父模型并照常 finish |

注意 `/workspace` 每轮都会被平台清空，`/workspace/records` 只在本地运行时可用。
平台上的持久证据是：每轮 `finish` 的 `configuration.evidence`，以及状态里的
`datasets/<slot>/round-log.json`（最近 30 轮的精简记录）和 `lineage.json`（上线过的模型）。
状态整体有 64 个文件的上限，所以成员 5 写进状态的逐轮审计文件只保留最新一份。

`runtime_status` 检查断网运行本身；`evidence_status` 检查公开 SwanLab 材料。
平台刚结束时 `evidence_status=pending` 是正常状态。导出 SwanLab 后，再增加
`--strict-evidence` 作为最终材料门禁。

## 运行后导出 SwanLab

先在联网机器上 dry-run：

```bash
python -m member5.swanlab_export \
  --records /workspace/records \
  --project zhixun-engine \
  --output /workspace/member5-report \
  --dry-run
```

确认后去掉 `--dry-run`。结果写入
`member5-report/swanlab-export.json`，不会重新训练模型或改变本地轮次。

## 证据完整性

轮次记录会优先从以下位置解析代码提交号：

```text
ZHIXUN_GIT_COMMIT
GIT_COMMIT
SOURCE_VERSION
/app/GIT_COMMIT
/workspace/GIT_COMMIT
```

正式镜像应由成员 1 在构建时写入提交号。如果 `git_commit` 为空，验收报告会
标记 `GIT_COMMIT_MISSING`，该轮不能作为正式实验证据。

`platform_ack` 应包含：

```json
{
  "received": true,
  "accepted": true,
  "ack_id": "platform-ack-id",
  "source": "rnd.finish"
}
```

缺少该字段时运行状态为 `pending`，不能声称真实平台轮次已经验收完成。

## 本地全系统试跑

```bash
python scripts/run_local_harness_e2e.py \
  --datasets A C \
  --rounds 10 \
  --output artifacts/local-simulated-ac10
```

本地运行器使用脚本化模型回复，但会真实执行成员 2–5 的流水线、官方
checkpoint、成员 4 训练和成员 5 门控。记录来源为：

```json
{"kind": "local-simulated"}
```

这类记录用于验证控制面和稳健性，不允许导出到 SwanLab，也不能作为比赛成绩。

## 合并多个证据目录

```bash
python -m member5.evidence_bundle \
  --source B-live=artifacts/live-qwen-b10/B/work/records \
  --source A-C-local=artifacts/local-simulated-ac10 \
  --output artifacts/evidence-bundle \
  --expected-commit <最终提交号>
```

最终平台材料检查时增加 `--require-platform`，此时 `legacy`、
`local-live-qwen` 和 `local-simulated` 均不能作为平台证据。

## SwanLab 链接回填

上传完成后：

```bash
python -m member5.swanlab_reconcile \
  --records artifacts/live-qwen-b10/B/work/records \
  --manifest artifacts/live-qwen-b10/B/reaudit/swanlab-export.json
```

该步骤把 URL 写回对应 `record_id` 的本地 JSON，并重新计算
`evidence_status`。
