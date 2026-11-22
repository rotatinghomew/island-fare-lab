"""资料读取、结构校验与时间快照。

快照口径（字典：日期型可见时间按当日日终）：
- 触达按 ``seen_at`` 可见；
- 履约对账按 ``received_day`` 日终可见（晚到记录在到达前不进分析）；
- 旧价权利按 ``change_at`` 可见的状态版本，同一报价保留多版本时间线；
- 时段、套餐目录、价表为静态主数据。
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, date, time
from pathlib import Path

from .models import (
    CatalogItem,
    DataIssue,
    Exposure,
    Fulfilment,
    Journey,
    Period,
    Policy,
    QuoteRight,
    parse_day,
    parse_dt,
)

TABLES = (
    "periods",
    "journeys",
    "exposures",
    "fulfilment",
    "catalog",
    "quote_rights",
    "policies",
)


@dataclass
class DataBundle:
    """一次加载后的全部主数据与索引。"""

    periods: dict[str, Period]
    journeys: dict[str, Journey]
    exposures: list[Exposure]
    fulfilment: list[Fulfilment]
    catalog: dict[str, CatalogItem]
    quote_rights: list[QuoteRight]
    policies: dict[str, Policy]
    issues: list[DataIssue] = field(default_factory=list)

    # ---- 索引 ----
    def exposures_of(self, journey_id: str) -> list[Exposure]:
        return self._exposure_index[journey_id]

    def fulfilment_of(self, journey_id: str) -> Fulfilment | None:
        return self._fulfilment_index.get(journey_id)

    def rights_of(self, period: str) -> list[QuoteRight]:
        return [r for r in self.quote_rights if r.period == period]

    def period_of_journey(self, journey_id: str) -> Period:
        return self.periods[self.journeys[journey_id].period]

    def __post_init__(self):
        self._exposure_index: dict[str, list[Exposure]] = {}
        for exp in self.exposures:
            self._exposure_index.setdefault(exp.journey, []).append(exp)
        for rows in self._exposure_index.values():
            rows.sort(key=lambda e: e.seen_at)
        self._fulfilment_index = {f.journey: f for f in self.fulfilment}

    # ---- 时间快照 ----
    def snapshot(self, as_of: datetime) -> "DataBundle":
        """返回截至 ``as_of`` 已可见记录组成的新 bundle（主数据原样保留）。"""
        exposures = [e for e in self.exposures if e.seen_at <= as_of]
        cutoff_day = as_of.date()
        fulfilment = [f for f in self.fulfilment if f.received_day <= cutoff_day]
        rights = [r for r in self.quote_rights if r.change_at <= as_of]
        return DataBundle(
            periods=self.periods,
            journeys=self.journeys,
            exposures=exposures,
            fulfilment=fulfilment,
            catalog=self.catalog,
            quote_rights=rights,
            policies=self.policies,
            issues=list(self.issues),
        )

    # ---- 业务事实体检 ----
    def detect_issues(self) -> list[DataIssue]:
        """识别需要人工接住的资料异常（只报告，不改写）。"""
        issues: list[DataIssue] = []
        for f in self.fulfilment:
            seen: dict[str, float] = {}
            for exp in self.exposures_of(f.journey):
                seen.update(exp.price_map)
            if f.plan in seen and abs(seen[f.plan] - f.paid) > 1e-9:
                issues.append(
                    DataIssue(
                        code="paid_price_mismatch",
                        message=(
                            f"旅程 {f.journey} 实收 {f.paid:.0f} 元与任何触达所见 "
                            f"{f.plan} 价格 {seen[f.plan]:.0f} 元均不符，承诺需人工核对"
                        ),
                        refs={"journey": f.journey, "plan": f.plan, "paid": f.paid},
                    )
                )
            if f.plan not in seen:
                issues.append(
                    DataIssue(
                        code="bought_unseen_plan",
                        message=(
                            f"旅程 {f.journey} 购买了从未在触达中出现的套餐 {f.plan}"
                        ),
                        refs={"journey": f.journey, "plan": f.plan},
                    )
                )
        for journey_id, journey in self.journeys.items():
            period = self.periods[journey.period]
            if not self.exposures_of(journey_id):
                issues.append(
                    DataIssue(
                        code="journey_without_exposure",
                        message=f"旅程 {journey_id} 没有任何触达记录",
                        refs={"journey": journey_id},
                    )
                )
            f = self.fulfilment_of(journey_id)
            if f is None and period.closed_at < date.today():
                issues.append(
                    DataIssue(
                        code="closed_period_missing_fulfilment",
                        message=(
                            f"时段 {journey.period} 已过收齐日 {period.closed_at}，"
                            f"旅程 {journey_id} 仍无付款/履约记录，按未转化处理但需复核"
                        ),
                        refs={"journey": journey_id, "period": journey.period},
                    )
                )
        return issues


def load_bundle(data_dir: str | Path) -> DataBundle:
    """读取 ``data_dir`` 下全部 JSON 资料并完成结构、外键与一致性校验。"""
    data_dir = Path(data_dir)
    raw = {name: json.loads((data_dir / f"{name}.json").read_text(encoding="utf-8"))
           for name in TABLES}
    _validate_columns(data_dir, raw)

    periods = {
        r["period"]: Period(
            period=r["period"], day=parse_day(r["day"]), area=r["area"],
            window=r["window"], arm=r["arm"], random=bool(r["random"]),
            fleet_hours=float(r["fleet_hours"]), closed_at=parse_day(r["closed_at"]),
        )
        for r in raw["periods"]
    }
    journeys = {
        r["journey"]: Journey(
            journey=r["journey"], period=r["period"],
            requested_at=parse_dt(r["requested_at"]),
            length_hours=float(r["length_hours"]), party=int(r["party"]),
            needs=r["needs"],
        )
        for r in raw["journeys"]
    }
    exposures = [
        Exposure(
            exposure=r["exposure"], journey=r["journey"], channel=r["channel"],
            seen_at=parse_dt(r["seen_at"]), options=tuple(r["options"]),
            prices=tuple(float(x) for x in r["prices"]), old_choice=bool(r["old_choice"]),
        )
        for r in raw["exposures"]
    ]
    fulfilment = [
        Fulfilment(
            journey=r["journey"], plan=r["plan"], paid=float(r["paid"]), use=r["use"],
            paid_at=parse_dt(r["paid_at"]), received_day=parse_day(r["received_day"]),
        )
        for r in raw["fulfilment"]
    ]
    catalog = {
        r["plan"]: CatalogItem(
            plan=r["plan"], hours=float(r["hours"]), transfer=float(r["transfer"]),
            operator_coupon=float(r["operator_coupon"]),
            hour_cost=float(r["hour_cost"]),
        )
        for r in raw["catalog"]
    }
    quote_rights = [
        QuoteRight(
            quote=r["quote"], period=r["period"], origin=r["origin"],
            price=float(r["price"]), alternatives=tuple(r["alternatives"]),
            until=parse_dt(r["until"]), state=r["state"],
            change_at=parse_dt(r["change_at"]),
        )
        for r in raw["quote_rights"]
    ]
    policies = {
        r["version"]: Policy(
            version=r["version"], area=r["area"], plans=dict(r["plans"]),
            credit_ceiling=float(r["credit_ceiling"]),
            support_level=float(r["support_level"]),
        )
        for r in raw["policies"]
    }

    _validate_references(raw, periods, journeys, catalog)
    bundle = DataBundle(
        periods=periods, journeys=journeys, exposures=exposures,
        fulfilment=fulfilment, catalog=catalog, quote_rights=quote_rights,
        policies=policies,
    )
    bundle.issues = bundle.detect_issues()
    return bundle


def _validate_columns(data_dir: Path, raw: dict) -> None:
    contract = json.loads((data_dir / "format.json").read_text(encoding="utf-8"))
    for name, columns in contract["fields"].items():
        if name not in raw or not isinstance(raw[name], list) or not raw[name]:
            raise ValueError(f"资料表 {name} 缺失或为空")
        for row in raw[name]:
            if set(row) != set(columns):
                raise ValueError(f"资料表 {name} 字段集合与 format.json 不符：{row}")
    for exp in raw["exposures"]:
        if len(exp["options"]) != len(exp["prices"]):
            raise ValueError(f"触达 {exp['exposure']} 的套餐与价格数组长度不一致")


def _validate_references(raw: dict, periods, journeys, catalog) -> None:
    for r in raw["journeys"]:
        if r["period"] not in periods:
            raise ValueError(f"旅程 {r['journey']} 引用未知时段 {r['period']}")
    for r in raw["exposures"]:
        if r["journey"] not in journeys:
            raise ValueError(f"触达 {r['exposure']} 引用未知旅程 {r['journey']}")
    for r in raw["fulfilment"]:
        if r["journey"] not in journeys:
            raise ValueError(f"履约记录引用未知旅程 {r['journey']}")
        if r["plan"] not in catalog:
            raise ValueError(f"履约记录引用未知套餐 {r['plan']}")
    for r in raw["quote_rights"]:
        if r["period"] not in periods:
            raise ValueError(f"旧价权利 {r['quote']} 引用未知时段 {r['period']}")
        for plan in r["alternatives"]:
            if plan not in catalog:
                raise ValueError(f"旧价权利 {r['quote']} 引用未知替代套餐 {plan}")
    if len({f["journey"] for f in raw["fulfilment"]}) != len(raw["fulfilment"]):
        raise ValueError("同一旅程出现多条履约记录，工具不支持自动合并")


def snapshot(bundle: DataBundle, as_of: datetime) -> DataBundle:
    """模块级快照入口；日期参数按当日日终处理。"""
    if isinstance(as_of, date) and not isinstance(as_of, datetime):
        as_of = datetime.combine(as_of, time(23, 59, 59))
    return bundle.snapshot(as_of)
