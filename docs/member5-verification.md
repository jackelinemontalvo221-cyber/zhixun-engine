# 成员 5 验收记录

日期：2026-11-20

## 已验证功能

| 功能 | 证据 |
| --- | --- |
| 候选模型与当前模型比较 | A/B/C 官方检查点在相同验证集上完成评测；105 项测试全部通过 |
| 上线门控 | 集成测试验证候选提升时返回 `promote`，提交候选产物并选择 `model-2` |
| 回滚门控 | 集成测试验证候选退化、相同、损坏或父哈希不符时返回 `rollback`，继续选择父产物 |
| 本地轮次记录 | 每轮生成不可覆盖的 JSON 和 `index.jsonl`；断网时仍可写入 |
| SwanLab 在线记录 | 已创建公开冒烟实验，在线上传 22 条记录并正常结束 |
| Docker 环境 | 4 CPU、8 GB、`--network none`、非 root 用户下启动成功 |
| 模型权重合规 | 镜像中未发现 `.npz`、`.pt`、`.pth` 或 `.safetensors` 文件 |

## 公开 SwanLab 冒烟实验

- 项目：`zhixun-engine`
- 实验：`SMOKE-MEMBER5-round-0001-no_update-round-re`
- 链接：https://swanlab.cn/@Bay/zhixun-engine/runs/tbtm6can
- 状态：Finished

该记录只验证在线写入，不是比赛训练成绩。

## 镜像

- 镜像名：`zhixun-harness:member5`
- 镜像 ID：`sha256:b5c187df16139c73e1aee3e7bbbbaf7812cfc1b5a422683b39ab39ac862b525f`
- 镜像大小：约 328 MB
- 运行用户：`65534:65534`
- 入口：`python3 /app/harness.py`

## 当前依赖边界

成员 4 的真实 `train_candidate` 已合入。`tests/test_full_system.py` 现在使用
真实成员二诊断、成员三决策校验、成员四 A 包续训和成员五门控跑通一轮：
Macro-F1 从 0.7140 提升到 0.7257，候选被采纳。该结果来自本地 CPU 测试宿主，
不是平台正式评分；正式 Linux/amd64 镜像和平台试运行仍需完成。

## 2026-11-25 新增验收能力

新增代码和离线测试如下，尚未声称完成任何平台轮次：

- `member5/robustness.py`：按 `group_ids` 确定性拆成两折，分别比较父模型和
  候选模型。结果状态为 `robust`、`warning` 或 `insufficient`。
- `member5/evaluator.py`：配置启用后，把稳健性摘要加入门控理由，并把完整
  分折结果保存为 `members/evaluation/robustness-*.json`；门控结论保持原规则。
- `member5/audit.py`：检查真实调用回执、历史引用、Git 提交号、数据哈希、
  父/候选绑定、指标方向、门控理由、SwanLab 状态和自适应训练试验。
- `member5/metrics.py`：为分类任务输出逐类 precision/recall/F1、Macro
  precision/recall 和混淆矩阵；为回归任务输出最差批次 RMSE/MAE。
- `scripts/member5_report.py`：从轮次 JSON 生成 Markdown、CSV、JSON 总表。
- `member5/offline_acceptance.py`：分离断网运行状态和 SwanLab 公开证据，
  自动生成无人干预验收结论。
- `member5/swanlab_export.py`：运行结束后独立导出 SwanLab，不作为比赛运行依赖。
- `tests/test_member5_audit.py` 与 `tests/test_member5_robustness.py`：覆盖
  自适应试验摘要、证据缺失、分组分折和门控兼容性。

这些功能可以基于原 `feature/member5-evaluation-logging` 分支独立评审。
成员 2–4 的新分支现已加入团队集成基线；自适应试验的真实运行数据仍需等
平台试运行完成后填入。

当前 Python 3.12 验收环境：

```text
Windows 11
Python 3.12.10
pytest 9.2.0
NumPy 1.26.4
PyTRIO 0.2.10
contest_harness 1.0.4
SwanLab 0.10.5
```

Windows 完整测试结果：

```text
196 passed, 1 skipped, 1 deselected
```

跳过项是 Windows 无符号链接权限；排除项是会向测试进程自身发送 SIGTERM 的用例。
这两项已在 Ubuntu 24.04 + Docker `--network none` 中分别单独执行并通过。

该结果证明代码、A/B/C 检查点和符号链接负例在当前提交上通过，不代表真实平台
轮次已经通过。

## 离线与无人干预补充

- 比赛运行默认保持 `SWANLAB_MODE=offline`，本地 JSON 是权威来源。
- SwanLab 作为运行后独立导出，不再阻塞断网运行。
- 轮次记录新增 `platform_ack`；缺失时运行状态为 `pending`。
- `member5.offline_acceptance` 默认允许公开证据待导出，使用
  `--strict-evidence` 才把它升级为最终材料失败。
- 新增 `scripts/run_local_harness_e2e.py`：无网络运行脚本化决策、真实成员 4
  候选和真实成员 5 门控。
- 轮次记录新增 `provenance.kind`；`local-simulated` 和 `local-live-qwen`
  不会被当成官方平台证据，本地模拟记录禁止导出 SwanLab。

本地 A/C 各 10 轮试跑结果：

```text
record_count: 20
runtime_status: pass
evidence_status: pending
failures: []
A 第 1 轮：Macro-F1 0.7140297 -> 0.7256954，promote，稳健性 robust
C 第 1 轮：RMSE 52.8685726 -> 51.8300631，promote，稳健性 robust
后续轮次：no_update，父检查点保持不变
```

本地模拟记录不会被 SwanLab 导出器发布。

新增多证据合并与 SwanLab 回填工具：

- `member5.evidence_bundle`：检查提交号、来源类型、重复记录并生成 A/B/C 总表。
- `member5.swanlab_reconcile`：将公开 SwanLab URL 回填到本地轮次 JSON。
- PyTRIO HTTP 错误现在保存响应正文和 `retry-after`。
- Live Qwen 失败会写 `failure.json`，避免把半成品误认为成功。
- Qwen 可选择平台本帧明确提供的历史 artifact；Runner 通过 SDK
  `restore_artifact` 下载并校验文件，再把目标 artifact 提交为回滚结果。
- 轮次 JSON 在 `finish` 前先写入草稿；收到确认后再回填 `platform_ack` 和
  SwanLab 状态，避免平台已接受但本地记录缺失。
- 回滚和证据包只保存可移植目录名或相对路径。
- `max_training_trials_per_stage` 独立限制自适应搜索内部训练试验数。
