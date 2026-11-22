# 旅游租车套餐改价后的真实增量与旧价权利

各渠道将报价与预约交给定价台复盘。成本按实际消耗的车时和采购接驳计。

## 资料和运行环境

`data/` 提供本地合成业务资料。各业务文件为 UTF-8 JSON 数组，键、单位和关联见 [资料字典](data/README.md)。`format.json` 记录列集合和基础引用约束。

使用 Python 3.11，依赖锁定在 `requirements.txt`（numpy、scipy）：

```sh
python3 -m pip install -r requirements.txt
```

检查资料与全部业务口径：

```sh
PYTHONPATH=. python3 -m unittest discover -s tests -v
```

`tests/test_input_contract.py` 覆盖文件可读性、字段与明确单列引用；其余测试覆盖
定价试算工具的业务不变量。业务原始记录允许保留冲突、未决关系及尚未到达的观察，
工具只报告、不改写。

## 定价试算工具 `pricing_lab/`

一个可在项目内反复调用的 Python 包，也提供命令行入口。从资料加载到结论的完整链路：

```
io 资料读取/校验/时间快照
 └─ evidence 证据分层（旅程单位、seen/unseen、诊断隔离）
     └─ estimator 可用性条件 logit + 时段整批 bootstrap + 历史留出
     └─ support 共同支持诊断（超范围只作试探）
     └─ margin 单位毛利（车时/接驳/优惠随套餐替换而变）
     └─ rights 旧价权利台账（多版本状态、车位与补差预留、释放）
     └─ feasibility 联合可行性（同一批抽样守容量、适用范围、补差预算）
     └─ commitments 已发承诺冻结与人工队列
     └─ reconcile 留出期试算与实际对账
```

### 快速开始

```sh
# 端到端演示（证据分层、两套阶梯、权利台账、留出对账、人工事项）
PYTHONPATH=. python3 scripts/run_trial_demo.py data

# 完整试算，报告与人工队列写入 runs/
PYTHONPATH=. python3 -m pricing_lab.cli --data data trial --mode 试算
PYTHONPATH=. python3 -m pricing_lab.cli --data data trial --mode 定案

# 共同支持、权利台账、留出对账、人工队列
PYTHONPATH=. python3 -m pricing_lab.cli --data data support --policy ladder-b
PYTHONPATH=. python3 -m pricing_lab.cli --data data rights --policy ladder-a
PYTHONPATH=. python3 -m pricing_lab.cli --data data reconcile --policy base --period P00
PYTHONPATH=. python3 -m pricing_lab.cli --data data queue
```

Python 调用：

```python
from pricing_lab import load_bundle, PricingTrial
from pricing_lab.feasibility import TrialMode

bundle = load_bundle("data")
trial = PricingTrial(bundle)
trial.fit(n_bootstrap=1000)
report = trial.run(mode=TrialMode.FINALIZE, n_draws=1000)
```

## 方法与口径

### 1. 分析单位：旅程，不是触达

一次企图预订由旅程标识认领。同一旅程在多个渠道询价（如直订 + 酒店）只形成一条
证据，套餐替换不会被计成新客。`evidence.py` 以旅程为单位聚合全部触达与履约。

### 2. 「没看到」与「看到没选」是两种证据

- 套餐出现在随机轮换触达的可购买选项里，才记为 `seen`，旅客选择其他套餐才构成
  对该套餐的拒绝证据；
- 从未出现的套餐记为 `unseen`，只代表缺失证据，不进入选择集，不被当成拒绝。
- 需求模型为**可用性条件 logit**：每条旅程的选择集严格取其实际看到的选项，
  外部选项（未付款）效用固定为 0，净增需求由外部选项的转化识别。

### 3. 随机轮换的单位是时段内整批客群

- 只有随机时段（`random=true`）且全程未出现 `old_choice` 触达的旅程进入因果比较；
- 渠道临时关闭时段（P09、P22）与可自行在旧价/新价间选择的旅程仅作**适用性诊断**，
  不并入随机比较，避免把自选行为当成新价效果；
- 不确定区间用**时段整批 cluster bootstrap**（重抽样整个时段而非拆散旅程）；
- 历史留出按时段整批划分折外集，报告 log-loss、平均绝对误差与选择准确率。

### 4. 共同支持：超范围只作试探

随机轮换中实际观察到的价格档与客群构成支持域。拟刊价落在观察网格内才算有证据；
超出共同支持的价表（如 `ladder-b` 在港城的 580/700/380 元档）整体标记为
「超支持范围仅可试探」，并逐条列出缺证的价格档与客群，定案模式下一律受阻。

### 5. 毛利与随套餐替换而变的成本

单位毛利 = 实收报价 − 车时成本（车时 × 车时单价）− 接驳采购 − 运营方承担优惠。
半日、整日、直通景点的车时、接驳、优惠不同，套餐替换的成本变化逐项可见。

### 6. 旧价权利：容量与价差的联合保护

- 同一报价的多版本状态按 `change_at` 形成时间线，快照取最新可见版本；
  到期或明确放弃即释放，行权后转为已兑现；
- 一份有效权利只占**一个并行车位**（不按等价替代数重复占量），并按等价替代范围内
  「发布价 − 旧报价」的最大正差预留补差；
- 容量与价差落在**同一批回来行权的旅程**上：可行性检验的每次模拟共用同一组
  bootstrap 参数、旅程选择与权利行权抽样，不能分别挑两组最有利预测。

### 7. 定案三约束与试算模式

定案须在价表 `support_level`（默认 90%）下同时守住：时段可售车位数、
旧价适用套餐范围、每时段补差预算。试算模式允许暴露较高价差风险并照常给区间，
但输出明确标注不得据此定案。

### 8. 承诺冻结与人工接住

渠道实收报价一旦确认即按当时条件核兑并冻结；后续补回的曝光/履约可以重新比较、
使可采用范围升降，但不能覆盖已发承诺，冲突须经显式人工变更并留痕。
实收与所见价不符、购买未见套餐、收齐后仍缺履约等异常全部进入人工队列。

### 9. 时间快照与回查

所有记录带可见时间：触达按 `seen_at`、履约对账按 `received_day` 日终、
权利按 `change_at`。`--as-of` 可复现任意历史时点的结论；晚到数据在到达前不进分析。
`reconcile` 按旧报价与定价会次回查预留/兑现/释放义务，并核对预测区间与实际的偏差。

## 已知口径选择（合成数据上的显式假设）

- `fleet_hours` 与真实履约件数同量纲（实际占用车时约为其 5 倍且所有时段如此），
  工具按**同时段可并行服务的车位数（可售量）**处理容量约束；车时仍作为成本与
  对账指标，按套餐车时核算。
- 同一旅程多条触达价格不一致时，选择模型取该旅程最后一次随机轮换触达价格，
  并在诊断中保留矛盾标记。
- 聚合选择数据无法识别逐对套餐替代流向，工具只报告可识别的各套餐需求变化与
  来自未转化旅客的净增量，不虚构逐对流向。
