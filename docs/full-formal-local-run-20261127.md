# 本地正式流程演练（2026-11-27）

这不是官方平台成绩。运行使用真实 Qwen、完整 Harness、真实训练/门控和 SwanLab
online 记录，只缺少官方平台的隐藏测试和正式评分。

## 运行

- 代码提交：`cb3c2f3`
- Qwen：`Qwen/Qwen3.8-27B`
- 数据集：A、B、C 各 10 轮
- 输出目录：`artifacts/full-formal-local-clean-20261127T131903Z`
- 30 轮均有真实 `pytrio-qwen-*` 调用号
- 30 轮均有 `round_complete` 和本地 `round_ack`
- 30 轮 SwanLab 状态均为 `submitted`

## 首轮结果

| 任务 | 父模型 | 首轮候选 | 结果 |
| --- | ---: | ---: | --- |
| A | Macro-F1 0.71403 | Macro-F1 0.72637 | promote |
| B | RMSE 681.15983 | RMSE 657.64542 | promote |
| C | RMSE 52.86857；NASA 252654.75 | RMSE 51.77710；NASA 152428.42 | promote |

C 实际执行的是 `input-refit-selected`，最佳 epoch 为 7，两个验证折均改善。

## 发现的问题

1. A 的验证投影只有 1 个 group，因此两折稳健性无法计算，状态为
   `insufficient`；这不是训练失败，但 A 的稳健性证据不足。
2. B 首轮总体 RMSE 改善，但 `Bearing1_4` 分组退化约 32.49，门控仍选择
   `promote`，只记 warning。是否应改成 warning 即拒绝，需要团队明确策略。
3. C refit8 总体 RMSE/MAE/NASA 均改善，两个验证折也改善，但逐 unit
   门控有部分 FD001 分组退化，最差 group 约 -10.41；当前同样只记 warning。
4. 本次 A/B/C 都使用重复的公开投影，不是真实多轮漂移流；后续更新轮次
   多数为 `no_update` 是合理行为，不能据此评价漂移适应能力。
5. 本地记录 provenance 为 `local-live-qwen`，严格验收按设计返回
   `evidence_status=pending`，不能冒充官方平台证据。
6. 一次 SwanLab 上传出现 HTTP 502，SDK 自动重试后恢复；最终状态仍为
   `submitted`。

## SwanLab

项目：<https://swanlab.cn/@gkdzjc/zhixun-engine>

首轮示例：

- A：<https://swanlab.cn/@gkdzjc/zhixun-engine/runs/qtklq9lt>
- B：<https://swanlab.cn/@gkdzjc/zhixun-engine/runs/usmbg04x>
- C：<https://swanlab.cn/@gkdzjc/zhixun-engine/runs/mzl176hl>

完整记录和统一证据包位于本地 `artifacts/full-formal-local-clean-20261127T131903Z`。
