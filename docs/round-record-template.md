# 标准轮次记录模板

每个数据集、每一轮都必须产生一份不可覆盖的 JSON 记录。成员 5 负责生成本地
`records/<dataset>-<round>-<run_id>.json`，并在开发阶段可选择导出 SwanLab。

## 必填字段

| 字段 | 含义 |
| --- | --- |
| `record_id` | 本轮记录唯一编号 |
| `run_id` / `dataset_key` / `round_index` | 运行和轮次定位 |
| `bundle` 哈希 | 训练、验证数据及规范化内容版本 |
| `active_before` | 上线前的父检查点及其哈希 |
| `diagnosis` | 漂移分数、类型、当前指标和预算 |
| `decision` | 动作、参数、理由和预计成本 |
| `candidate` | 候选检查点、父检查点、种子和训练指标 |
| `evaluation` | 同一验证集上的旧模型/候选指标和门控理由 |
| `final_action` | `promote`、`keep`、`rollback` 或失败状态 |
| `selected_artifact_id` | 平台最终选择的模型产物 |
| `resources` | CPU、Token、标注预算及本轮评测成本 |
| `code.git_commit` | 可追溯代码版本 |
| `swanlab` | 本地或在线记录状态；在线时包含公开链接 |
| `platform_ack` | 平台 `round_ack` 是否收到、接受及来源 |
| `provenance.kind` | `contest`、`local-live-qwen`、`local-simulated` 或 `legacy` |
| `rollback` | 主动恢复历史 artifact 时的目标、恢复哈希和来源信息 |
| `acceptance` | 自动证据检查结果、警告和错误列表 |

主动恢复历史的轮次会在 `rollback` 中记录：

```json
{
  "artifact_id": "model-old",
  "checkpoint_path": "checkpoint.npz",
  "checkpoint_sha256": "sha256:<restored digest>",
  "previous_checkpoint_sha256": "sha256:<previous digest>",
  "restore_dir": "rollback-<random>"
}
```

`restore_dir` 只保存工作区内的目录名，不保存开发机绝对路径。

`platform_ack` 示例：

```json
{
  "received": true,
  "accepted": true,
  "ack_id": "platform-ack-id",
  "source": "rnd.finish"
}
```

本地模拟运行必须标记：

```json
{"kind": "local-simulated"}
```

本地真实 Qwen 运行标记：

```json
{"kind": "local-live-qwen"}
```

两种记录都不会被视为官方平台证据。

详细分类/回归指标不塞进门控合同，而是写入：

```text
records/member5-metrics/metrics-<task>-<round>.json
```

其中分类包含逐类 precision/recall/F1 和混淆矩阵，回归包含逐批及最差批次
RMSE/MAE。`member5_report.py` 会把它们合并到 Markdown 报告。

## 最小示例

```json
{
  "schema_version": "round-record.v1",
  "dataset_key": "A",
  "round_index": 2,
  "active_before": {
    "checkpoint_sha256": "sha256:<64 lowercase hex>"
  },
  "decision": {
    "action": "finetune",
    "params": {"epochs": 5, "learning_rate": 0.001, "seed": 42},
    "rationale": "检测到输入漂移，先低成本续训"
  },
  "metrics": {
    "metric_name": "macro_f1",
    "baseline": {"macro_f1": 0.72, "accuracy": 0.88},
    "candidate": {"macro_f1": 0.74, "accuracy": 0.89},
    "delta": 0.02
  },
  "final_action": "promote",
  "swanlab": {
    "mode": "online",
    "status": "submitted",
    "url": "<public experiment URL>"
  }
}
```

候选未达到主指标提升门槛，或任一受保护辅助指标退化超过 `max_regression`
时，门控返回 `rollback`，父模型继续作为当前模型。记录不得删除失败轮次。

## 本地与在线模式

评测环境默认使用：

```bash
export SWANLAB_MODE=offline
```

开发实验使用：

```bash
export SWANLAB_MODE=online
export ZHIXUN_SWANLAB_PROJECT=zhixun-engine
export ZHIXUN_GIT_COMMIT="$(git rev-parse HEAD)"
```

在线上传失败不会阻塞比赛轮次，但本地记录中的 `swanlab.status` 必须显示
`failed`，并保留错误原因。
