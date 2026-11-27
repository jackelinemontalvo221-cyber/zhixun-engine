# 成员 4：工业模型训练与候选生成

本模块实现成员 4 的训练接口：

```python
train_candidate(bundle, decision, active, budget) -> CandidateModel | None
```

它从成员 1 验证过的当前父检查点继续训练，产生候选工业模型并返回真实
`checkpoint.npz` 字节。模块不负责漂移诊断、Qwen 调用、评测、上线或最终状态切换。

## 启用的动作

| 动作 | 数据与训练方式 |
| --- | --- |
| `no_update` | 不产生候选 |
| `adaptive_update` | 在预算内训练少量任务级候选，逐 epoch 验证并返回最佳 checkpoint |
| `finetune` | 从当前父检查点使用当前训练批次续训 |
| `retrain_full` | 从当前父检查点使用当前完整训练批次续训 |
| `retrain_recent` | 从当前父检查点使用当前批次末尾 `window_ratio` 比例的数据续训 |

`retrain_full` 不是重新初始化模型。所有训练动作都必须继承平台提供的父检查点。

`adaptive_update` 的决策参数为：

```json
{"max_candidates": 2, "max_epochs": 3, "seed": 42}
```

成员4根据数据包生成最多 `max_candidates` 个低成本计划，每个计划都从同一个
父检查点重新开始。每个训练 epoch 后都在公开验证集上评测，只保留该计划中
指标最好的 checkpoint，再在不同计划之间选择最终候选。候选仍必须经过成员5
的非回归门控才能上线。

搜索器会折叠同一父模型、同一数据变换下已被长候选覆盖的短候选；长时间
没有验证提升的计划会提前停止。这样可以避免重复训练和无收益 epoch。

当前默认计划：

| 任务 | 候选计划 | 选择指标 |
| --- | --- | --- |
| A | 全量最多20 epoch，逐 epoch 选优 | Macro-F1 |
| B | 最近30%（按轴承）最多3 epoch；全量1 epoch；可用时加入近期历史回放 | RMSE |
| C | 全量最多8 epoch，使用训练集重新拟合输入标准化；可用时加入历史回放 | RMSE |

所有候选都从同一个父检查点继续训练，保持父模型的任务、特征路径、类别和 MLP
宽度；平台要求基于官方 baseline 和父检查点续训，不允许提交改变结构的新模型。

训练状态目录还会保存：

| 文件 | 内容 |
| --- | --- |
| `members/training/replay.npz` | 有界、仅训练集、按类别/RUL分层采样的历史回放缓冲区 |
| `members/training/strategy_stats.json` | 各候选策略的历史平均提升和胜出次数 |
| `members/training/latest.json` | 最近一轮候选、逐 epoch 指标和选择结果 |
| `members/training/audit.jsonl` | 不可覆盖的历史候选审计 |

可选候选会根据历史策略统计重新排序；已证明长期无效的候选会降低优先级。

### Champion–challenger 与预算

- 无新训练数据且指标没有退化时，默认不训练；
- B 连续稳定达到配置间隔且 CPU 预算充足时，允许 `1 candidate x 1 epoch`；
- C 默认关闭周期探测，避免放大 NASA 非对称误差风险；
- C 的全量候选只用当前训练集重新拟合输入标准化统计量，最多 8 epoch；
- 无退化但有新数据时，使用受限 idle search；明显漂移才展开完整搜索；
- 所有候选仍经过验证集和成员5门控，不会因为“探测”自动上线。

公开B验证集的初步对照：

| 策略 | 数据窗口 | epochs | RMSE |
| --- | --- | ---: | ---: |
| 父模型 | - | - | 681.16 |
| 普通近期更新 | 最近30% | 3 | 631.88 |

该表为历史对照。结构化剪枝候选已移除；平台反馈明确要求候选必须基于父检查点
续训，不能改变模型结构。候选始终要经过成员5门控。

## 安全约束

- 校验任务类型、输入维度、分类类别顺序和 active 检查点身份。
- 没有训练标签时拒绝训练。
- 续训必须沿用父检查点的随机种子。
- 不修改、不覆盖父检查点。
- 候选文件在临时目录生成，通过 `CandidateModel.checkpoint_bytes` 交给 Harness。
- 训练失败抛出 `TrainingError`，由外层保留当前模型。
- 每轮的候选计划、逐 epoch 指标、选中策略和失败原因写入
  `members/training/latest.json` 与 `audit.jsonl`。

## 接入

使用配置：

```bash
CONTEST_CONFIG=configs/harness.full.json
```

全系统配置中的训练组件为：

```json
{
  "train_candidate": "member4.trainer:CandidateTrainer"
}
```

`configs/harness.member4.json` 保留为成员四调试入口；它与正式总配置使用同一个
训练组件。

## 测试

```bash
python -m pytest -q tests/test_member4_trainer.py
```

测试覆盖真实 A/B/C 官方检查点、不更新、近期窗口、父检查点保护、哈希校验、
无标签、任务/维度/类别不匹配、seed 不一致和 CPU 预算不足。

## 当前限制

- 训练成本估算仍是保守开发值，后续需要用 2 核、4 GiB 沙箱下的 A/B/C 真实
  运行数据重新标定。
- `retrain_full` 使用当前训练 bundle；平台是否向当前容器提供更早历史训练数据，
  应以正式数据合同为准。
- 集成分支已切换到 SDK 1.0.5，并使用 `chunked-v1` 分块上传测试宿主。
