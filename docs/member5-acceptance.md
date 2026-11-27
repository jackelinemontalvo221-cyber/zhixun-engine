# 成员 5 验收流程

成员 5 不负责训练候选或设计 Qwen 提示词。成员 5 负责检查候选是否有资格
上线，并把每一轮的父模型、候选、指标、成本和证据整理为可复核记录。

## 一轮验收顺序

1. 检查真实模型调用 ID、历史引用和代码提交号是否存在。
2. 检查父模型哈希和候选父哈希一致。
3. 检查候选哈希、artifact ID 和本地记录的候选一致。
4. 检查父模型与候选模型使用同一验证集和同一指标方向。
5. 检查 `promote / rollback` 理由与最终 artifact 一致。
6. 对自适应候选检查以下训练证据：
   - `selected_plan`
   - `trials`
   - `best_epoch`
   - `training_rows`
   - `migration`
   - `replay_rows_used`
   - 失败 trial 的错误原因
7. 检查两折稳健性审计。整体提升但某一折退化的候选标记为警告。
8. 检查 SwanLab 是否公开、是否成功提交，并保留链接。
9. 在总表中将本轮标为 `pass`、`warning` 或 `fail`。

验收状态分为两层：

```text
runtime_status   断网运行、调用、数据和门控是否有效
evidence_status  公开 SwanLab 等最终材料是否齐全
```

比赛运行阶段允许 `SWANLAB_MODE=offline`。此时 `evidence_status=pending`
不代表运行失败；运行结束后单独导出 SwanLab。

## 自适应候选的特殊风险

成员 4 在验证集上选择最佳计划和最佳 epoch，成员 5 又使用同一验证集做上线
比较。因此整体验证集提升不能单独证明候选在新数据上稳定。

两折稳健性结果用于报告和复核，不自动改变门控阈值。只有在真实平台轮次积累
足够证据后，才决定是否把某一折退化升级为硬性拒绝条件。

## 指标口径

分类任务：

- 主指标：Macro-F1，最大化。
- 辅助指标：Accuracy。
- 报告项：每类 precision、recall、F1，混淆矩阵和最差类别。

回归任务：

- 主指标：RMSE，最小化。
- 辅助指标：MAE。
- 报告项：平均 RMSE、最差批次 RMSE。

## 平台轮次材料

向成员 1、3、4 收集：

- 镜像标签、镜像摘要、代码提交号
- 平台 run ID、轮次号和 artifact ID
- 真实 Qwen 调用 ID、原始回复和是否回退
- 候选试验、父/候选哈希、CPU 和 Token
- SwanLab 公开链接
- 本地轮次 JSON 和 `index.jsonl`
- 详细指标文件 `records/member5-metrics/metrics-*.json`

没有机器可读证据的截图不能代替轮次 JSON。

平台确认 `round_ack` 后，轮次记录中的 `platform_ack` 至少应包含：

```json
{"received": true, "accepted": true, "source": "rnd.finish"}
```
