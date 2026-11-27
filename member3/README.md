# 成员三：大模型决策与接入

基于成员二 PR #1 的 `15751c3a816681e6ff62f6ce5c006f947b5bf663`。
此版本实现成员三，以及接入成员二所缺的配置、上下文、跨轮状态接口。

## 已实现

- `LLMDecider.bind_round(context)` / `decide(diagnosis, history, budget)`。
- 支持 `adaptive_update`：Qwen 选择候选数量、最大 epoch 和预算，成员4在范围内
  逐 epoch 选优；普通单项训练动作仍保留。
- Qwen提示词会收到成员4的候选策略历史统计，用于减少重复无效搜索。
- 解析成员二标准 Diagnosis 与 notes JSON；模型决定动作与参数，程序严格校验。
- 单轮最多一次模型调用，使用平台 `ModelGateway`；输出上限遵循平台限制。
- 拒绝重复 JSON 键、NaN/Inf、额外字段、非法动作、参数越界、不可执行训练和超预算成本。
- 无法解析或不合法的模型回复：使用成员二规则提出备选，再经过相同校验；备选也失败则 no_update。
- 模型调用失败：记录保守消耗，抛 DecisionError；不伪造回执或成功轮次。
- 真实调用回执、提示词版本、消息、原回复、实际/估计 Token 消耗、兜底原因及最终决定保存到状态。
- 正式入口读取 CONTEST_CONFIG，按配置加载组件，并向组件传入每轮上下文。
- 通过 SDK 验证父检查点并复制只读状态到可写工作区；按数据集隔离新行记录、参考成绩和预算。
- 提供官方基线的只读评测、预测适配，三种任务共用官方 checkpoint 格式。
- 修复官方入口数组特征的字符串排序问题，保持官方顺序 features[2] 在 features[10] 前。

## 当前边界（务必阅读）

全系统正式配置是 `configs/harness.full.json`，已同时注册成员二的漂移诊断、
成员三的模型决策、成员四的工业模型训练和成员五的评测门控。镜像默认使用
该配置。

`configs/harness.member3.json` 只保留为成员三的独立调试配置；没有训练和
门控组件时，可用动作只有 `no_update`。`configs/harness.json` 保留成员一
原有的 fake 模式，只用于旧测试和离线演示。

## 本地安装与测试（无付费调用）

在项目根目录，用 Python 3.10+（建议 3.12）创建独立虚拟环境，然后：

```powershell
python -m pip install -e ".[dev]"
python -m pip install ./contest-harness-sdk-1.0.5.tar.gz
python -X utf8 -m pytest -q --basetemp=.test-tmp
python -X utf8 -m member3.demo
```

所有自动测试与 demo 使用模拟模型/宿主，不访问外网，不消耗 PyTRiO 额度。
SDK 联调测试使用真正的官方 SDK、公开 A 数据与父 checkpoint，并模拟宿主响应。
Windows 账户无符号链接权限时，仅符号链接测试跳过，其他测试照常执行。
如果路径过长，请将项目放入较短路径，例如 Documents/Codex/zhixun。

## 正式入口

```powershell
$env:CONTEST_CONFIG = "configs/harness.full.json"
python app/harness.py describe
```

train 需要平台注入 start 帧，不要直接在空终端运行并等待它完成。
密钥由平台持有，本模块不需要把 API Key 写入配置或镜像。Docker 镜像默认设置
`CONTEST_CONFIG=/app/configs/harness.full.json`。正式提交前仍需在比赛环境
复核一次性调用、CPU/Token 预算和 Linux/amd64 镜像。

本地 orchestrator 使用 FakeModelGateway，仅用于离线接线验证，日志中的 fake-call 不是真实回执。
原本地 manifest 夹具没有独立验证集；不把训练成绩冒充当前验证成绩，缺少证据时保持 no_update。

## 成员四、五接口

`configs/harness.full.json` 已登记团队的 `train_candidate` 和
`evaluate_and_commit` 实现。训练类接口如下：

```python
supported_actions = ("no_update", "finetune", "retrain_full", "retrain_recent")

def bind_round(self, context):
    self.context = context

def estimate_training_cost(self, action, params):
    # 返回经过你们测量/标定的 CPU 秒估算，不可直接让模型编造。
    ...

def train_candidate(self, bundle, decision, active, budget):
    # 使用 context.checkpoint_path 续训，返回 CandidateModel。
    ...
```

门控类提供 `evaluate_and_commit(bundle, candidate, active, gate, eval_bundle=None)`，返回 Evaluation。
只有训练和门控均注册后，决策器才开放训练动作。
全量/近期重训均指从当前父模型续训；禁止重新初始化冒充续训。
模型选择范围：epochs 1–20、学习率 1e-6–0.1、batch_size 1–4096、seed uint32，近期窗口比例 (0,1]。
如成员四支持范围更小，需同步收紧 decider.py 的提示词与校验，并添加对应测试。
`rollback` 只能选择本轮 `rnd.artifacts` 中明确提供的历史 artifact；
`params` 使用 `target_artifact_id` 或 `target_checkpoint_sha256`。Runner 通过
SDK `restore_artifact` 下载并校验文件，再把目标 artifact 作为本轮选择提交。
当前候选提交只写 checkpoint.npz；如果训练依赖额外代码/资产，须与成员一扩展候选交付协议。

## 预算与记录

示例本地预算：每数据集累计 3600 CPU 秒、120000 Token、60 轮；它们不是比赛官方额度。
实际使用时按团队和平台额度修改。Token 为 0 表示禁止调用，不表示无限。
当本地 Token 预算无法支持必需的模型调用时，模块明确失败，不跳过调用冒充完成。
有 usage.total_tokens 时记真实回执值；没有时记 UTF-8 输入字节数 + 256 封装预留 + max_tokens。
这是本地保守估算，包含不了宿主追加官方提示的实际开销，最终计量以平台为准。
平台若返回超过预算的真实用量，仍完整记账并选择 no_update。
CPU 计量为当前 Python 进程 CPU 时间，外部训练进程/远端耗时需由对应训练模块和平台记录。

状态路径（根目录由 SDK 管理）：

- datasets/<任务哈希>/members/drift：成员二参考成绩与历史。
- datasets/<任务哈希>/members/decision/latest.json：最新模型调用和决定。
- datasets/<任务哈希>/members/decision/audit.jsonl：各轮调用记录，供成员五导入 SwanLab。
- datasets/<任务哈希>/rows.json：训练/验证分离的样本内容摘要，不存原始特征。
- datasets/<任务哈希>/reference.json、budget.json、latest-round.json。

## 复核与实验

先运行全部测试，再比较：固定规则 vs 大模型决策。固定数据划分、父 checkpoint、随机种子、
训练实现和评测门槛，比较 F1/RMSE、CPU、Token、失败次数及更新次数。
测试通过仅证明接口与保护逻辑，不证明大模型优于规则。真实收益需有完整实验与 SwanLab 记录。
