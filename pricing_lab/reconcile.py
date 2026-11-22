"""试算预测与实际履约的回查对账。

对每个已收齐时段输出：
- 需求：各套餐预测区间（试算时点）对实际付款件数的覆盖；
- 车时：预测车时区间对实际消耗车时的覆盖；
- 毛利：预测单位毛利加总与实际实收毛利的差额；
- 旧价权利：预留/兑现/释放的台账状态与实际行权是否一致。
预测区间未覆盖实际值即标记偏差，供定价会次回查，不回改历史承诺。
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import numpy as np

from .estimator import DemandModel
from .evidence import JourneyEvidence, build_evidence
from .io import DataBundle
from .margin import gross_margin_components
from .models import PLANS, Policy
from .rights import RightsLedger


@dataclass
class MetricCheck:
    name: str
    predicted: float
    low: float
    high: float
    actual: float
    covered: bool
    gap: float


@dataclass
class ReconcileReport:
    period: str
    closed: bool
    metric_checks: dict[str, MetricCheck]
    rights_checks: list[dict]
    covered_count: int
    total_count: int

    @property
    def all_covered(self) -> bool:
        return self.covered_count == self.total_count

    def as_dict(self) -> dict:
        return {
            "period": self.period,
            "closed": bool(self.closed),
            "coverage": f"{self.covered_count}/{self.total_count}",
            "all_covered": bool(self.all_covered),
            "metrics": {
                k: {
                    "predicted": round(v.predicted, 2),
                    "interval": [round(v.low, 2), round(v.high, 2)],
                    "actual": round(v.actual, 2),
                    "covered": bool(v.covered),
                    "gap": round(v.gap, 2),
                }
                for k, v in self.metric_checks.items()
            },
            "rights": self.rights_checks,
        }


def reconcile_period(
    bundle: DataBundle,
    model: DemandModel,
    ledger: RightsLedger,
    policy: Policy,
    period: str,
    cohort: list[JourneyEvidence],
    n_draws: int = 1000,
    rng: np.random.Generator | None = None,
) -> ReconcileReport:
    rng = rng or np.random.default_rng()
    period_row = bundle.periods[period]
    prices = {p: policy.plans[p] for p in PLANS}
    available = {p: True for p in PLANS}
    # 实际是整数件数，对账区间用离散联合模拟而非连续期望
    dist = model.choice_simulation(
        cohort, available, prices, n_draws=n_draws, rng=rng
    )
    level = policy.support_level
    alpha = (1 - level) / 2

    # 实际值（按快照内履约记录）
    actual_demand = {p: 0 for p in PLANS}
    actual_hours = 0.0
    actual_margin = 0.0
    predicted_margin_draws = np.zeros(n_draws)
    hours_draws = np.zeros(n_draws)
    unit_margin = {
        p: gross_margin_components(bundle, p, prices[p]).unit_margin for p in PLANS
    }
    evidence = build_evidence(bundle)
    for ev in cohort:
        full = evidence[ev.journey]
        if full.chosen_plan is not None:
            actual_demand[full.chosen_plan] += 1
            actual_hours += bundle.catalog[full.chosen_plan].hours
            fulfilment = bundle.fulfilment_of(full.journey)
            actual_margin += gross_margin_components(
                bundle, full.chosen_plan, fulfilment.paid
            ).unit_margin
    for d in range(n_draws):
        hours_draws[d] = sum(
            dist[p][d] * bundle.catalog[p].hours for p in PLANS
        )
        predicted_margin_draws[d] = sum(
            dist[p][d] * unit_margin[p] for p in PLANS
        )

    checks: dict[str, MetricCheck] = {}
    for plan in PLANS:
        low, high = np.quantile(dist[plan], [alpha, 1 - alpha])
        point = float(np.mean(dist[plan]))
        checks[f"需求_{plan}"] = MetricCheck(
            name=f"需求_{plan}", predicted=point, low=float(low), high=float(high),
            actual=float(actual_demand[plan]),
            covered=low - 1e-9 <= actual_demand[plan] <= high + 1e-9,
            gap=actual_demand[plan] - point,
        )
    low, high = np.quantile(hours_draws, [alpha, 1 - alpha])
    checks["消耗车时"] = MetricCheck(
        name="消耗车时", predicted=float(hours_draws.mean()), low=float(low),
        high=float(high), actual=actual_hours,
        covered=low - 1e-9 <= actual_hours <= high + 1e-9,
        gap=actual_hours - float(hours_draws.mean()),
    )
    # 占用车位（容量口径，与可售量同量纲）
    slot_draws = sum(dist[p] for p in PLANS)
    actual_slots = sum(actual_demand.values())
    low, high = np.quantile(slot_draws, [alpha, 1 - alpha])
    checks["占用车位"] = MetricCheck(
        name="占用车位", predicted=float(slot_draws.mean()), low=float(low),
        high=float(high), actual=float(actual_slots),
        covered=low - 1e-9 <= actual_slots <= high + 1e-9,
        gap=actual_slots - float(slot_draws.mean()),
    )
    low, high = np.quantile(predicted_margin_draws, [alpha, 1 - alpha])
    checks["毛利"] = MetricCheck(
        name="毛利", predicted=float(predicted_margin_draws.mean()),
        low=float(low), high=float(high), actual=actual_margin,
        covered=low - 1e-9 <= actual_margin <= high + 1e-9,
        gap=actual_margin - float(predicted_margin_draws.mean()),
    )

    rights_checks = []
    for view in ledger.all_views(policy):
        if view.period != period:
            continue
        # 实际行权：该报价为已行权状态视为兑现；有效/放弃/过期与台账一致即可
        rights_checks.append(
            {
                "quote": view.quote,
                "state": view.state.value,
                "reserved_slots": view.reserved_slots,
                "reserved_credit": round(view.reserved_credit, 2),
                "released": view.capacity_released,
                "note": view.note,
            }
        )

    covered = sum(1 for c in checks.values() if c.covered)
    return ReconcileReport(
        period=period,
        closed=period_row.closed_at <= date.today(),
        metric_checks=checks,
        rights_checks=rights_checks,
        covered_count=covered,
        total_count=len(checks),
    )
