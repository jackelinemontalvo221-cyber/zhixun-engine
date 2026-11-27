# 工程问题修复（2026-11-27）

本文件对应组员提出的三个工程问题。

## C 数据准备与 wall-clock

- `WorkspaceV2Source` 现在把 SDK 已解析的行直接交给规范化器，不再先生成
  JSONL 字符串再解析。
- C 单轮准备实测从约 `5.06s / 1.08GB` 降到约 `2.78s / 0.58GB`（本机）。
- `RoundContext.started_monotonic` 在读取数据之前创建，因此数据准备时间
  也纳入本轮 wall-clock。

## rows.json 状态上限

- 行状态改为固定大小的 Bloom filter（`row-bloom-v1`），不再保存无界的
  全部哈希字符串。
- C 单轮 `rows.json` 约 `5.33MB`，重复相同轮次不会继续增长。
- `enforce_state_budget()` 现在同时检查文件数量和总字节数。

Bloom filter 有极小的假阳性概率，因此新行识别是保守的；它不会保存验证集
或隐藏集内容。

## SwanLab

- 合并时（成员 1）把 `swanlab` 改回可选依赖（`pip install .[swanlab]`），镜像不再安装：
  赛题规定正式评测在“统一、离线的运行环境”中进行，且“不得进行未授权的外部网络访问”，
  容器内在线记录既做不到也不合规；不装还能让镜像更小。代码里 `SWANLAB_MODE=online`
  的可选路径保留（仅本地实验用）。平台运行改为事后用
  `scripts/upload_platform_runs.py` 补记到 SwanLab。
- `evidence_status=pending` 时，离线验收整体状态为 `warning`，不会被误当成
  完整最终证据；需要完整证据时必须使用 `--strict-evidence` 或完成 SwanLab
  回填。
