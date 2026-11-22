# 资料字典

时钟使用北京时间，日期型可见时间按当日日终。端点相同的时段按排定时段标识认领。没收到付款记录的旅程在收齐日前仍未闭合。定价台的价差按行权套餐在相应发布价表与旧报价之间需要企业承担的正差计。

## `periods.json`

报价方案安排及对应用量。

| 字段 | 含义 |
| --- | --- |
| `period` | 轮换时段编号 |
| `day` | 服务日期 |
| `area` | 分析区域 |
| `window` | 服务段 |
| `arm` | 原分配方案 |
| `random` | 是否按公布的整批随机轮换 |
| `fleet_hours` | 可支配车时 |
| `closed_at` | 业务数据收齐日期 |

## `journeys.json`

同一旅程可以有不止一次曝光。

| 字段 | 含义 |
| --- | --- |
| `journey` | 匿名旅程标识 |
| `period` | 报价轮换时段 |
| `requested_at` | 首次请求时间 |
| `length_hours` | 最初请求时长 |
| `party` | 同行人数 |
| `needs` | 车位条件 |

## `exposures.json`

套餐代码和所见报价按数组位置对应。

| 字段 | 含义 |
| --- | --- |
| `exposure` | 触达编号 |
| `journey` | 旅程标识 |
| `channel` | 触达渠道 |
| `seen_at` | 可见时间 |
| `options` | 可购买的套餐代码 |
| `prices` | 对应人民币元 |
| `old_choice` | 是否可自行在旧价与新价间选择 |

## `fulfilment.json`

付款与履约核对记录。

| 字段 | 含义 |
| --- | --- |
| `journey` | 旅程标识 |
| `plan` | 实购套餐 |
| `paid` | 实收元 |
| `use` | 履约状态 |
| `paid_at` | 付款时间 |
| `received_day` | 对账记录可见日期 |

## `catalog.json`

套餐所耗共用运力。

| 字段 | 含义 |
| --- | --- |
| `plan` | 套餐代码 |
| `hours` | 车时 |
| `transfer` | 采购接驳元 |
| `operator_coupon` | 运营方承担优惠元 |
| `hour_cost` | 车时单价元 |

## `quote_rights.json`

酒店已发可行权旧报价与处理进度。

| 字段 | 含义 |
| --- | --- |
| `quote` | 报价编号 |
| `period` | 约定时段 |
| `origin` | 原套餐 |
| `price` | 有效报价元 |
| `alternatives` | 可等价替换套餐 |
| `until` | 到期时刻 |
| `state` | 有效、行权或放弃 |
| `change_at` | 状态记录可见时刻 |

## `policies.json`

每个价表按当前确认范围比较，时段预算以实际书面额度为限。

| 字段 | 含义 |
| --- | --- |
| `version` | 价表标识 |
| `area` | 提案范围 |
| `plans` | 分套餐拟刊价格 |
| `credit_ceiling` | 每时段补差上限元 |
| `support_level` | 置信度下限 |
