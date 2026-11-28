# 成员 2：漂移分析与预算

| 文件 | 内容 |
|---|---|
| `drift.py` | `DriftDiagnoser`，实现 `Pipeline.diagnose`，输出诊断报告 `Diagnosis` |
| `rules.py` | `RuleDecider`，实现 `Pipeline.decide`。这是临时决策规则，等成员 3 的大模型决策接入后替换；`decision_from_diagnosis()` 可以留给成员 3 作兜底 |
| `signals.py` | 各种漂移信号的计算，都是纯 NumPy 函数 |
| `simulate.py`、`simulate_rul.py` | 开发用：按真实接口重放漂移流，前者用 UCI 气体数据（A 任务），后者用 C-MAPSS（C 任务）。`results/` 里是三个种子的结果 |

## 怎么启用

在配置文件里登记（`configs/harness.member2.json` 就是现成的例子）：

```json
"pipeline": {
  "diagnose": "member2.drift:DriftDiagnoser",
  "decide":   "member2.rules:RuleDecider"
}
```

- 正式运行：设置环境变量 `CONTEST_CONFIG=/app/configs/harness.member2.json`，或者把这两行合并进 `harness.json`。
- 本地运行：`python -m zhixun_harness.orchestrator --config configs/harness.member2.json …`。

## 每轮回答三个问题

### 1. 当前模型变差了没有？——这是唯一的触发条件

每批数据都带标签，所以最直接的办法就是比较当前模型的**现在成绩**和它的**参考成绩**：

- 分类任务看 Macro-F1 掉了多少：$\Delta = m_{\text{ref}} - m_{\text{now}}$；
- 回归任务看 RMSE 涨了百分之几：$\Delta = (m_{\text{now}} - m_{\text{ref}}) / m_{\text{ref}}$。

| 严重程度 | 分类（Macro-F1 绝对下降） | 回归（RMSE 相对上升） | 建议 `recommendation` | 建议训练轮数 |
|---|---|---|---|---|
| `severe` | $> 0.15$ | $> 30\%$ | `retrain_full` | 20 |
| `mild` | $0.05 \sim 0.15$ | $10\% \sim 30\%$ | `finetune` | 20 |
| `none` | $\le 0.05$ | $\le 10\%$ | 见下面三条 | 5 或 0 |
| `unknown` | 算不出来 | 算不出来 | `investigate`（临时规则会把它当作不更新） | 0 |

`none` 的时候：

- 输入超出训练范围的比例**创下新高**，或者出现训练时没见过的运行工况：`finetune` 5 轮；
- 否则，CPU 预算还剩一半以上：照样 `finetune` 5 轮（开关是 `IDLE_LIGHT_UPDATE`）。有验证门控把关，更差的候选不会被换上。这在 C 任务上让 RMSE 和「每轮都更新」持平，在 A 任务上没有影响，见下文实验；
- 预算紧张：`no_update`。

回归阈值是在 C-MAPSS 漂移流上标定的。不同批发动机之间，即使没有漂移，RMSE 也会差 5–15%，所以 5% 太容易误触发。

成绩的来源：

- **现在成绩**：当前模型在本轮验证集上的成绩。能加载模型时由成员 2 自己算，加载不了时用成员 5 给的 `current_eval`。
- **参考成绩**，按优先级依次取：
  1. 模型被换上那一轮测得的成绩（`at_activation`，平台或成员 5 测的）；
  2. 成员 2 第一次见到这个模型时测的成绩（存在 `first_seen.json` 里）；
  3. 模型在它训练过的数据上的成绩。这个值偏乐观，只在第一轮、实在没有别的参考时才用。

CPU 预算用完时，一律给 `no_update`。

### 2. 是哪里变了？——用来归因，不触发更新

比较的是「以前见过的行」和「这一轮新来的行」，靠 `context.new_rows` 区分。第一轮还没有新旧之分，就比较训练集和验证集。

| 信号 | 含义 |
|---|---|
| `psi_mean`、`psi_max`、`psi_frac_gt_0.25` | 各特征分布变化的大小（PSI，按参考数据的十分位分箱） |
| `ks_frac_sig` | KS 检验显著（$\alpha = 0.01$）的特征比例 |
| `top_features` | 变化最大的 5 个特征，带上「平移了几个标准差」。A 任务会翻译成「传感器 N·特征类型」 |
| `worst_sensors` | A 任务：变化最大的 3 个传感器 |
| `regime_tv`、`unseen_regime_share` | C 任务：各运行工况的占比变了多少，以及有多少新数据落在训练时没见过的工况里 |
| `clip_rate`、`clip_increase` | 新数据里超出模型标准化范围（±10）的比例，以及它比以往各轮最高值多出多少。标准化参数在第一次训练时就固定了，比例高可能是长期状态，只有创下新高才算新情况 |
| `conf_drop` / `pred_psi` | 模型预测置信度的下降（分类），或预测值分布的变化（回归） |
| `label_shift` | 标签或目标值分布的变化 |

**C 任务按工况分组**：用取整后的 `settings` 把数据分成 6 种运行工况，每个传感器在**各自的工况内**做标准化后再比较。否则只要各工况的占比一变，所有传感器都会显得在漂移。实测「设备状态不变、只改工况占比」时，不分组的 PSI 均值是 2.25，分组后降到 0.375；`regime_tv = 0.73` 准确指出了原因是工况占比变了。模型的 RMSE 前后是 46.6 和 46.8，没有变化，所以结论是不更新。

`drift_type` 怎么定：

| 成绩掉了 | 输入有变化 | `drift_type` |
|---|---|---|
| 是 | 是 | `mixed` |
| 是 | 否 | `concept` |
| 否 | 是 | `covariate`（只记录，不触发更新） |
| 否 | 否 | `none` |
| 什么都算不出 | | `unknown` |

### 3. 还剩多少预算？

`budget_remaining` 直接取成员 1 的预算对象。没设上限的项填 $10^{15}$，因为合同要求这里必须是数字。

## 交给成员 3 的诊断报告

`Diagnosis` 的标准字段：

- `drift_score`：$\max(0, \Delta) / 0.15$，大于 1 就是严重；
- `drift_detected`、`drift_type`、`recommendation`；
- `current_metrics`：成员 5 的成绩，加上 `member2_<指标>` 和 `new_rows_<指标>`；
- `baseline_metrics`：参考成绩；
- `budget_remaining`；
- `evidence_ids`：最近 10 条历史记录的 id。

其余细节都放在 `notes` 里，是一段压缩的 JSON，不超过 4096 字符。成员 3 写提示词时可以直接用：

```json
{"v":"member2-drift-v1","metric":"macro_f1","current":0.714,"current_src":"member2_validation",
 "reference":0.9558,"reference_src":"in_sample_training_rows","perf_drop":0.2417,
 "new_rows_score":0.714,"new_rows_drop":0.2417,"severity":"severe","split":"old_vs_new_train",
 "rows":{"total":1689,"new":1244},
 "signals":{"psi_mean":0.7293,"ks_frac_sig":0.9844,"worst_sensors":[9,10,5],"clip_rate":0.0012,
            "conf_drop":-0.0066,"label_shift":0.3388},
 "top_features":[{"name":"传感器5·上升EMA0.01","psi":1.731,"shift_sd":1.65}],
 "rule":"成绩明显下降（超过严重阈值），用全部数据续训","epochs_hint":20,
 "trend":[{"round":2,"severity":"severe","perf_drop":0.2417,"recommendation":"retrain_full"}],
 "budget_warnings":[],"model":"loaded","cpu_s":0.05}
```

（这是官方 A 包上的真实输出，有删节。其中 0.714 和官方 `training.json` 里的验证成绩一致。）

从 `member2-drift-v2` 起，`notes` 里还有三项：

| 字段 | 含义 |
|---|---|
| `model_seed` | 当前父检查点的随机种子。成员 4 只接受这个种子续训，规则兜底直接用它 |
| `new_fraction` | 训练数据里本轮新行的占比（按行 id 判定；没有新行标记时为 `null`） |
| `update_hint` | 建议的 `adaptive_update` 搜索规模：有明显下降时用成员 4 的任务默认规模；没下降但预算充足时只搜 1 个候选、1 个 epoch；否则 `no_update`。训练器不支持 `adaptive_update` 时为 `null` |

## 跨轮状态（`members/drift/`）

| 文件 | 内容 |
|---|---|
| `history.json` | 最近 30 轮的严重程度、成绩下降、PSI、超范围比例和建议；报告里的 `trend` 和 `clip_increase` 都取自这里 |
| `first_seen.json` | 每个检查点第一次被成员 2 见到时的成绩，作为它以后的参考成绩 |

## 临时决策规则（`RuleDecider`）

把 `recommendation` 换成一个合法的 `Decision`：

| `recommendation` | `action` | `params` |
|---|---|---|
| `retrain_full` | `retrain_full` | `{"epochs": 20, "seed": 1}` |
| `finetune` | `finetune` | `{"epochs": 20 或 5, "seed": 1}` |
| `retrain_recent` | `retrain_recent` | 加上 `window_ratio: 0.5` |
| `rollback` | `rollback` | `{}` |
| `no_update`、`investigate` | `no_update` | `{}` |

- 平台要求每轮至少真实调用一次大模型，所以这个规则会向平台模型问一句「这个建议是否合理」，并把回答写进 `rationale`。拍板的仍然是规则。
- 调用时 `max_tokens` 不会超过平台给的单次上限 `rnd.max_output_tokens`，超过的话 SDK 会直接拒绝这次调用。

## 实验结果

两条模拟流都走真实接口：官方数据路径、新数据标记、状态目录、本诊断、本规则、官方 MLP 续训，再加一道验证门控，候选比当前模型好才换上。每条流都跑了 3 个随机种子。

### A 任务：UCI 气体漂移

UCI 第 2–10 批依次作为第 1–9 轮。每批随机对半分，一半公开（再按 70/30 分成训练和验证），一半当作隐藏测试集，只用来打分。分数是隐藏集上的 Macro-F1，$G$ 的参考上界用「每轮都更新」近似。

| 策略 | 平均 F1 | 每个种子 | 最差一轮* | 平均 $G$ | 训练轮数 |
|---|---|---|---|---|---|
| 永不更新 | 0.433 | 0.437 / 0.429 / 0.434 | 0.191 | 0 | 0 |
| 每轮都更新 | 0.952 | 0.975 / 0.950 / 0.932 | 0.850 | 1 | 180 |
| **成员 2** | **0.952** | 0.975 / 0.948 / 0.933 | **0.856** | **0.996** | **165** |

\* 三个种子各自最差一轮的平均值。

- A 任务几乎每批都有严重漂移，所以能省的训练量不多（8%）。
- 调参记录：第一版规则对轻度下降只续训 10 轮。结果在种子 2 上，3 次轻度下降里有 2 次候选被门控挡下，平均 F1 只有 0.937。改成 20 轮后才是上表的结果。
- 一次诊断用时 0.03–0.13 秒。

### C 任务：C-MAPSS 涡扇发动机

官方 C 检查点是用 FD001–FD004 全部数据训练的，对它来说没有漂移。所以这里从一个**只在一半 FD001 发动机上训练**的模型出发（官方结构和超参数），依次收到：

FD001 剩下的一半 → FD003 分 2 段 → FD002 分 3 段 → FD004 分 3 段，共 9 轮。

FD003 多了一种故障模式，FD002 有 6 种运行工况，FD004 两者都有。各子集的第 2、3 段属于「没有新漂移」的轮次。所有划分都**按发动机**进行：同一台发动机的数据前后相关，按行划分会把答案泄漏给测试集。分数是隐藏集上的 RMSE，越低越好。

| 策略 | 平均 RMSE | 每个种子 | 最差一轮 | 平均 $G$ | 训练轮数 |
|---|---|---|---|---|---|
| 永不更新 | 76.10 | 74.36 / 79.59 / 74.37 | 107.6 | 0 | 0 |
| 每轮都更新 | 63.46 | 60.97 / 67.26 / 62.16 | 81.9 | 1 | 135 |
| **成员 2（阈值 10%，空闲时轻量续训）** | **63.45** | 60.93 / 66.60 / 62.83 | **81.8** | 0.958 | **88**（−35%） |
| 成员 2（阈值 5%，空闲时轻量续训） | 63.39 | 61.06 / 66.60 / 62.50 | 82.1 | 0.970 | 98 |
| 成员 2（阈值 10%，空闲时不更新） | 64.30 | 61.72 / 66.61 / 64.57 | 81.7 | 0.846 | 57（−58%） |

- 三个真正的漂移点都抓到了：
  - FD003 进来：RMSE 涨了 93%；
  - FD002 进来：RMSE 涨了 151%，85% 的新数据落在没见过的工况里；
  - FD004 进来：RMSE 涨了 23%。
- 空闲时轻量续训的两种做法对比：开启时 RMSE 和每轮都更新持平；关闭时训练再省一半，代价是 RMSE 高 1.3%。默认开启。CPU 预算紧张时，规则会自动停掉它。
- 在官方 C 包上（训练集加验证集共 13.6 万行）：一次诊断 0.31 秒，峰值内存 311 MB（包括读数据）。算出的当前 RMSE 是 52.87，和官方基线 52.9 一致。
- 过程中修掉的问题：标准化参数在 FD001 上就固定了，从 FD002 起约 65% 的输入一直超出范围。第一版规则因此每轮都触发轻量续训。现在只有超范围比例创下新高才算新情况。

完整的逐轮数据在 `results/` 里：

- `uci_a_3seeds.json`
- `cmapss_c_3seeds.json`
- `cmapss_c_3seeds_no_idle_update.json`

复现命令：

```sh
python -m member2.simulate --raw-dir "<…>/gas+sensor+array+drift+dataset/Dataset" \
       --package cpu-a-guide-20261110-v1 --seeds 0 1 2
python -m member2.simulate_rul --package cpu-c-guide-20261110-v1 --seeds 0 1 2 --mild 0.05 0.1
```

## 与成员 4 的 `adaptive_update` 配合：策略对照实验（11/25）

工具：`member2/compare.py`。每轮走真实组件：成员 2 诊断 → 决策 → 成员 4 `CandidateTrainer` → 成员 5 `EvaluationGate`，只有 Qwen 调用换成固定策略。数据流和前面标定用的一样（A：UCI 第 2–10 批；C：FD001→FD003→FD002→FD004），3 个种子，在隐藏的一半上计分。

```bash
python -m member2.compare --task A --raw-dir <UCI Dataset> --package cpu-a-guide-20261110-v1 --seeds 0 1 2
python -m member2.compare --task C --package cpu-c-guide-20261110-v1 --seeds 0 1 2
```

结果在 `member2/results/compare_*.json`。主要数字（3 个种子的平均；A 是 Macro-F1，越高越好；C 是 RMSE，越低越好；CPU 是 9 轮训练和门控合计的秒数）：

| 策略 | A 平均 | A 最差 | A CPU | C 平均 | C 最差 | C CPU |
|---|---:|---:|---:|---:|---:|---:|
| 从不更新 | 0.433 | 0.191 | 0 | 76.10 | 107.6 | 0 |
| 每轮 adaptive（成员 4 现有计划） | 0.926 | 0.812 | 1.9 | 64.63 | 82.4 | 1.3 |
| 每轮 finetune（A 20 轮，C 15 轮） | 0.951 | 0.852 | 2.8 | **63.46** | 81.9 | 4.9 |
| 成员 2 触发 + adaptive | 0.923 | 0.802 | 1.8 | 64.81 | 83.4 | 1.0 |
| 成员 2 规则兜底（finetune/retrain_full） | **0.952** | 0.852 | 2.4 | 63.93 | **80.3** | 3.8 |
| 每轮 adaptive，计划放宽到 A 20 / C 15 个 epoch | 0.940 | 0.804 | 3.4 | 63.83 | 82.2 | 3.3 |
| 成员 2 触发 + 放宽后的 adaptive | 0.940 | 0.804 | 3.2 | 63.84 | 82.2 | 2.4 |

结论：

1. **成员 4 现有计划的 epoch 上限太低。** A 最多 5 个 epoch，C 最多 2 个。A 的三个种子都比固定 20 轮 finetune 低（平均 0.926 对 0.951）。把上限放宽到 A 20、C 15（实验里临时改，`--plan-epochs`），差距大部分消失。
2. **训练很便宜。** 9 轮合计最多 5 CPU 秒，配置预算是 3600 秒。所以"少训几轮省算力"几乎换不来资源分，反而丢成绩。
3. **阈值不用改。** C 上 5%/10%/15% 几乎没有差别。A 用 10% 变差（0.912），3% 和 5% 一样。保持 A 5%、C 10%、空闲轻量更新开启。
4. **没下降的轮次做轻量搜索就够了。** "成员 2 触发 + 放宽后的 adaptive"在没下降的轮次只搜 1 个候选、1 个 epoch，成绩和每轮全量搜索一样（A 0.940 对 0.940，C 63.84 对 63.83），C 上 CPU 少约 27%，A 上差不多。这就是 `update_hint` 的依据。关掉空闲更新，C 上 CPU 再少约 40%，但 RMSE 高 0.2–0.5。
5. **门控用同一份验证集的偏乐观程度不大。** 上线的候选，验证集提升和隐藏集提升接近：A 0.257 对 0.222，C 14.4 对 13.6。但 C 上 adaptive 每 9 轮有 1–2 次"验证集变好、隐藏集变差"的上线。种子 0 的逐轮记录里，9 次中有 8 次来自"最近 20% 数据"的计划：这个计划上线 35 次，8 次在隐藏集上变差；"全量 1 个 epoch"上线 18 次，只有 1 次变差。把 `min_delta` 提到 0.3 或 1.0 也没有改善（只在实验中试过，结果没有保存进仓库）。
6. **修了一个兜底的 bug。** `decision_from_diagnosis` 原来固定 `seed: 1`，成员 4 要求和父检查点一致（官方都是 42），所以 Qwen 回复不合法、走规则兜底时，每轮训练都会报错。C 流上 9 轮全部失败、模型从不更新。现在从 `notes.model_seed` 取，缺省 42。

## 给本地真实 Qwen 测试接上漂移数据流（11/26）

队长的 `scripts/run_live_qwen_e2e.py` 每轮都用同一份公开数据，所以第 2–10 轮没有东西可反应。`member2/drift_stream.py` 在**不改这个脚本**的前提下，每轮开始前把这一轮的 `public_projection` 换成新的漂移批次：

- **A**：官方检查点（用 UCI 第 1 批训练），然后依次是 UCI 第 2–10 批。
- **C**：从半个 FD001 训练的模型出发，依次 FD001→FD003×2→FD002×3→FD004×3（按发动机切分）。官方 C 检查点已经见过全部四个子集，没有漂移可测。
- **第 10 轮是「安静轮」**：没有新训练数据，验证集和第 9 轮相同，用来检查系统会不会选择不训练。

每批数据随机分成公开训练、公开验证和隐藏三部分；训练数据逐轮累积，新行接在末尾。每轮结束后，在这一轮的隐藏数据上给当前父模型打分，并和一直不更新的初始模型对比。

```bash
# 真实 Qwen（PyTRIO，每轮 1 次调用）
python -m member2.drift_stream --datasets A C --rounds 10 --uci-raw-dir <UCI Dataset 文件夹>
# 离线演练：用脚本模拟的模型，照成员 2 的 update_hint 作答，不联网、不花钱
python -m member2.drift_stream --datasets A C --rounds 10 --scripted-qwen --uci-raw-dir <UCI Dataset 文件夹>
```

输出在 `artifacts/drift-stream-<时间>/`：脚本自己的轮次记录，加上 `drift_stream.json`（每轮动作、门控结果、隐藏集成绩）。下面这次离线演练的结果在 `member2/results/drift_stream_scripted_seed0.json`。

离线演练结果（种子 0，完整走 SDK 轮次协议 → 成员 2 → 模拟模型 → 成员 4 → 成员 5）：

| | 隐藏集平均 | 不更新时 | 最差一轮 | 10 轮结果 |
|---|---:|---:|---:|---|
| A（Macro-F1） | 0.939 | 0.431 | 0.823 | 上线 9、不更新 1 |
| C（RMSE） | 65.15 | 76.53 | 88.92 | 上线 6、回退 3、不更新 1 |

这次也改了一条规则：**没有新训练数据、成绩也没下降时，不在每轮做轻量更新**。
连续稳定 4 轮后允许一次 1 候选、1 epoch 的有限探测；只有实际进入训练的
探测才计入失败次数，连续 2 次未提升后冷却 8 轮。本轮一旦出现退化或新增
训练数据，就优先执行本轮训练，不再套用历史稳定探测。

## 局限

- **B 任务（轴承）没有单独标定。** 官方数据只有 12 个轴承、3 种工况，按轴承划分公开集和隐藏集后，每段只剩一两个轴承，标定结果不可信。B 和 C 同属回归，暂时沿用 C 的阈值。
- A 任务几乎每批都有严重漂移，能省的算力不多，只省了 8%。
- 第一轮只能用训练集上的成绩作参考，偏乐观，第一轮更容易判为严重。
- 回滚建议目前不由成员 2 给出：判断「上一个版本是否更好」需要把它取回来重新评估，这一步交给成员 4、5 的门控更合适。

## 测试

- `tests/test_member2_adaptive.py`：9 项，包括安静轮不训练、模拟模型的作答规则、逐轮累积的数据投影，以及用离线模拟模型驱动队长的 live 脚本跑 3 轮。原有 5 项覆盖规则兜底用父检查点的种子、`notes` 新字段、`update_hint` 的规模、对照工具的策略选择，以及用官方 C 检查点跑两轮真实"训练器 + 门控"。
- `tests/test_member2.py`：9 项，覆盖这几种情况：
  - 没有漂移；
  - 标签关系变了；
  - 只有输入变化；
  - 工况占比变化；
  - 没有模型也没有参考；
  - 预算用完；
  - 跨轮状态；
  - 规则只调用一次模型；
  - 官方 A 包上的结果和官方成绩对得上。
- `tests/test_round_protocol.py::test_round_with_member2_registered`：用成员 2 的配置完整跑一轮协议。
- `tests/test_image.py::test_member2_round_in_image`：在镜像里，断网、只读、非 root，挂上真实检查点完整跑一轮。

## 单独提交说明（不改成员 1 的文件）

这次提交只新增文件：`member2/`、`tests/test_member2.py`、`tests/test_member2_standalone.py`、`configs/harness.member2.json`，没有修改成员 1 的任何代码。

在现在的 `xhc` 分支上：

- `member2` 可以导入，`DriftDiagnoser` 和 `RuleDecider` 满足 `Pipeline` 协议，输出能通过 `contracts.Diagnosis` / `Decision` 的校验（见 `tests/test_member2_standalone.py`）。
- 但流水线还拿不到当前模型、跨轮状态和新数据标记，诊断会退化为 `investigate`，临时规则把它当作不更新。`tests/test_member2.py` 会整体跳过，等成员 1 提供下面的接口后自动启用。
- `notes` 控制在 3900 字节以内，因为它会被当作 SDK 的 `hypothesis` 提交（上限 4000 字节）。

需要成员 1 提供的接口（`member2` 全部用 `getattr` 读取，缺哪项就跳过哪项）：

| 接口 | 用途 |
|---|---|
| 每轮调用 `pipeline.bind_round(context)` | 把本轮上下文交给成员 2 |
| `context.checkpoint_path` | 当前模型存档，用来算置信度、截断比例等 |
| `context.eval_bundle` | 本轮验证集，用来算"现在成绩" |
| `context.new_rows.new_mask` | 训练数据中哪些行是本轮新来的 |
| `context.member_dir("drift")` | 成员 2 的跨轮状态目录（参考成绩、历史） |
| `context.reference`、`context.active`、`context.round_index`、`context.dataset_key`、`context.gate`、`context.gateway` | 参考成绩、当前模型身份、轮次、任务、门控规则、平台模型调用 |
| 按配置加载流水线 | `configs/harness.member2.json` 里的 `"pipeline": {"diagnose": "member2.drift:DriftDiagnoser", "decide": "member2.rules:RuleDecider"}` |

`simulate.py`、`simulate_rul.py`、`compare.py` 是开发用的重放脚本。`simulate.py` 原来依赖仓库里没有的 `scripts/build_stages_a_uci.py`，现在改用 `compare.read_uci_batch`。
官方 `baselines` 代码：优先用成员 1 的 `zhixun_harness.official`；没有时，按 `vendor/baselines` 或环境变量 `ZHIXUN_BASELINES` 指向的目录查找。
