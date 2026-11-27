# 成员3：本地验证与平台验收交接

日期：2026-11-24。代码来源：GitHub main，下载时页面提交号 `d74e832becacd04d740f5e0399ef8f3ca60ef6b5`。
本地验证于 2026-11-24 完成；2026-11-25 整理到 `member3/decision-audit-20261124` 分支供 PR 审核，是否合并以 GitHub 状态为准。

## 已完成

- 检查提示词、动作及参数校验、JSON 解析、规则回退、预算保护与决策日志。
- 修复超大整数引发 OverflowError、规则诊断字段形状异常导致回退失败的问题；增强深层 JSON 异常处理。
- 缺失/全空白 call_id：记录已有 Token 用量与 invalid_receipt 日志，再明确失败；不补造回执，不完成成功轮次。
- Token 用完强制 no_update 时，日志标记 budget_override，并保存 override_reason，而非继续标为模型决定。
- 日志增加任务、阶段、尝试、诊断编号及 system 提示词 SHA-256；调用失败时也保存这些上下文。
- 新增 8 个回归测试，其中修改前 7 个失败、1 个通过。

集成成员4的 `adaptive_update` 后，提示词版本升级为
`member3-decision-v4`，system 文本 SHA-256：
`93d01444428e02074351ee10bfb5e16e9b26445af92ee7d590fe4ca066d3af6e`。
测试固定该指纹；有意改变提示词时应同步修改版本和锁定值。动态诊断、预算、历史仍逐轮变化，完整消息保存在审计日志。

## 验证与限制

本地补充环境：Windows、Python 3.13.15、NumPy 2.5.4、pytest 9.2.0、官方 SDK 1.0.4。
使用已有本地包建立离线环境，未联网调用模型。未修改 pyproject.toml 中的 NumPy 1.26.4 依赖。
完整回归：127 passed, 1 skipped；跳过 Windows 无权限创建符号链接的测试。
describe 正常返回 SDK 1.0.4、classification / regression；member3.demo 的五种模拟场景完成。

这不是团队要求的 Python 3.12 + NumPy 1.26.4 空白环境验收，也不代表 Linux 容器或真实平台调用成功。
最初的临时测试方式有两个子进程找不到 SDK 的环境错误；将 SDK 放入独立离线环境后，全套回归通过。
新增测试不是八个真实调用，所有回执均为测试模拟值。

## 队长如何集成

1. 检出本 PR 的 `member3/decision-audit-20261124` 分支，确认没有未保存修改。
2. 审阅 member3/decider.py、tests/test_member3_audit_edges.py 和本文档的差异。
3. 如果 main 后续已有变化，先解决差异并重新验证。
4. 在 Python 3.12 新虚拟环境执行：

```text
python -m pip install -e ".[dev]"
python -m pip install ./contest-harness-sdk-1.0.4.tar.gz
python -m pytest -q
python -m member3.demo
```

5. 验证通过后由队长审核合并本 PR，并重新构建镜像。代码修复不会自动进入现有平台镜像。

## 平台首次试运行：待验收

用户先前确认已有可用镜像；随后实际查看当前登录账号的作品提交页面，显示“尚未上传镜像”“还没有提交版本”，试运行 0/10，正式评测 0/3。尚未启动任何运行。需核对镜像是否仅存在本地，或是否在其他队伍账号；取得已上传镜像后先核对代码提交号，首次只验证一轮，不直接启动十轮。

- 由队长控制平台运行次数，使用正式 configs/harness.full.json。
- 取得真实 call_id、模型名称、原始回复和消息；确认 rnd.prompt 没有重复附加。
- 对照 latest.json / audit.jsonl 中的提示词版本、指纹、Token 来源与最终 decision。
- 验证父检查点哈希、history 读取、状态上传、finish/round_ack 和合法预测输出。
- 如果必须指定 no_update，应使用明确的一轮试运行配置或平台支持方式，不能把任意 Qwen 输出假称 no_update。
- 模型调用失败或 call_id 缺失时必须失败，不把回退策略当成完成真实调用的替代品。
- 将运行 ID、代码提交号、数据版本、父哈希、call_id、SwanLab 链接一起留存。

## EX-007 固定规则与 Qwen：待真实数据

为 A/B/C 分别执行配对实验。两组固定数据顺序和划分、初始父检查点、代码、种子、工业模型、漂移阈值、可选动作、参数范围、门控和预算；只改变动作决策来源。
各策略按自身被平台采纳的产物继续后续轮次，不互相借用未来结果。

重要：平台每轮要求真实模型调用，规则组也不能省略真实回执。现有 member2.rules.RuleDecider 会做一次简短模型调用，但预算和审计逻辑与 LLMDecider 不完全相同；未经统一校验、计量和日志处理，不应直接把两者结果称为严格单变量对照。
可选设计：两组使用相同的模型请求，规则组只把模型回复作为旁观记录，动作来自固定规则；这样比较决策质量，Token 成本不体现省略模型调用的收益。若采用现有短请求规则组，应明确提示词/Token 也改变，属于实际系统成本对比。

每轮记录：任务、轮次、策略、运行 ID、call_id、提示词版本、父/候选哈希、决定、参数、是否回退、失败原因、Token、CPU 秒、门控采纳/拒绝、指标及 SwanLab 链接。
分类记录 Macro-F1、Accuracy、混淆矩阵；回归记录 RMSE（不要填分类 Accuracy）。
分类最差批次取最低 F1，回归最差批次取最高 RMSE。均值和失败率写清分母；模型调用失败、非法回复回退、训练失败分别统计，不能把回退率直接当整轮失败率。
所有实验结果待填，不能用 demo 的模拟回复证明 Qwen 优于规则。

## 团队范围外发现

main ZIP 实际含 cpu-a/b/c 官方数据和 checkpoint，另有多个 :Zone.Identifier 文件名，Windows 解压会跳过这些附加标识条目。
这与任务清单“仓库不含官方数据/checkpoint”不一致。请队长统一处理仓库和镜像打包范围，本次没有删除这些团队文件。
