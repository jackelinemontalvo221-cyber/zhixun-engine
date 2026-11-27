# 真实 Qwen 本地全系统验证（2026-11-26）

## 目的

在不消耗平台试运行和正式评测次数的前提下，使用本机已登录的 PyTRIO
账户调用真实 `Qwen/Qwen3.8-27B`，由它驱动成员3决策，再执行真实的成员4
工业模型训练、逐 epoch 选优、成员5门控和跨轮状态保存。

这不是平台官方结果。平台宿主、隐藏测试数据、平台 artifact 持久存储和
SwanLab 在线证词仍需平台验收。

> 2026-11-27 更新：平台反馈要求候选必须基于官方 baseline 和父检查点续训。
> 本文中 B 任务的结构化剪枝结果仅为历史记录；代码已移除该候选，所有候选
> 现在都保持父模型结构。

## 前置条件

```bash
cd /Users/zhangjingcheng/Desktop/zhixun-member4
source .venv/bin/activate
python -m pip install -e ".[live]"
trio login
```

API Key 从 `PYTRIO_API_KEY` 或现有 `~/.pytrio/config.toml` 读取，不写入
仓库、输出目录、命令行参数或镜像。

## 复现命令

```bash
python scripts/run_live_qwen_e2e.py \
  --datasets B \
  --rounds 10 \
  --max-tokens 512
```

A/C 单轮快速检查：

```bash
python scripts/run_live_qwen_e2e.py \
  --datasets A C \
  --rounds 1 \
  --max-tokens 512
```

脚本会创建新的 `artifacts/live-qwen-*` 目录，保存每轮记录、策略、真实
Qwen `call_id`、父/候选哈希、`round_ack` 和最终父checkpoint摘要。

## 已观察结果

B 任务最终十轮验证：

- 第1轮真实Qwen选择`adaptive_update`。
- 成员4选择结构化剪枝候选（该候选已因平台要求移除，见上方更新）。
- RMSE从`681.160`降至`463.439`。
- MAE从`563.106`降至`386.666`。
- 最差批次RMSE从`1064.973`降至`701.148`。
- 成员5返回`promote`，候选成为后续父模型。
- 第2–10轮真实Qwen均选择`no_update`。
- 父checkpoint摘要跨轮保持一致。
- 每轮保存真实`call_id`、历史引用、`round_ack`、详细指标和稳健性结果。

成员5离线验收：

```text
record_count: 10
runtime_status: pass
status: pass
evidence_status: pending
```

唯一剩余的系统级待办是运行后的 SwanLab公开导出。

A任务单轮：

- 真实Qwen选择`adaptive_update`。
- Macro-F1从`0.7140`提升到`0.7258`。
- 成员5返回`promote`。

C任务单轮：

- 真实Qwen选择`no_update`。
- 保留父模型。

## 已知限制

- 本地十轮重复使用同一公开投影，没有复刻平台的真实逐轮漂移数据流。
- 本地宿主模拟`history`和`round_ack`，不等同于平台持久artifact服务。
- A/C只完成真实Qwen单轮，双任务十轮仍需补齐。
- B剪枝仍需按轴承分组做稳定性和最差轴承审计。
- PyTRIO调用产生个人账户费用，因此脚本为显式opt-in，不进入默认pytest。
