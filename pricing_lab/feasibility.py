"""时段定案的联合可行性检验。

三条硬约束同时成立才允许定案：
1. 时段可售量（可并行车位）：随机轮换客群模拟售出件数 + 有效旧价权利预留车位数
   不超过 ``fleet_hours``（资料中该字段为同时段可并行服务的车位数，
   真实履约件数与它同量纲；车时成本在毛利模块按套餐车时另行核算）；
2. 旧价适用套餐范围：新价表必须覆盖该时段全部有效权利的等价替代套餐；
3. 补差预算：同一批回来行权旅程的模拟补差 + 已预留价差 ≤ 每时段补差上限。

关键口径：容量与价差落在同一批回来行权的旅程上，因此每次模拟共用同一组
bootstrap 抽样、同一组旅程选择与同一组权利行权抽样，
严禁分别取两组最有利的预测。
试算模式（``TRIAL``）允许输出较高价差风险并照常给区间；定案模式（``FINALIZE``）
在策略支持水平（``support_level``）下同时守三条线。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

import numpy as np

from .estimator import DemandModel
from .evidence import JourneyEvidence
from .io import DataBundle
from .models import PLANS, Policy
from .rights import PeriodObligation, RightsLedger


class TrialMode(str, Enum):
    TRIAL = "试算"
    FINALIZE = "定案"


@dataclass
class JointFeasibility:
    period: str
    policy_version: str
    mode: TrialMode
    n_draws: int
    sellable_slots: float
    credit_ceiling: float
    # 每次模拟（共用抽样）的总占用车位与总补差
    slot_draws: np.ndarray
    credit_draws: np.ndarray
    demand_draws: dict[str, np.ndarray]
    reserved_slots: float
    reserved_credit: float
    scope_violations: tuple[str, ...]
    support_gaps: tuple[str, ...]
    feasible_rate: float
    capacity_violation_rate: float
    credit_violation_rate: float
    scope_ok: bool
    support_ok: bool
    slot_quantile: tuple[float, float]
    credit_quantile: tuple[float, float]
    decision: str
    reasons: tuple[str, ...] = field(default_factory=tuple)

    def as_dict(self) -> dict:
        return {
            "period": self.period,
            "policy_version": self.policy_version,
            "mode": self.mode.value,
            "decision": self.decision,
            "feasible_rate": round(self.feasible_rate, 4),
            "sellable_slots": self.sellable_slots,
            "slot_quantile": [round(x, 2) for x in self.slot_quantile],
            "reserved_slots": self.reserved_slots,
            "credit_ceiling": self.credit_ceiling,
            "credit_quantile": [round(x, 2) for x in self.credit_quantile],
            "reserved_credit": round(self.reserved_credit, 2),
            "scope_ok": self.scope_ok,
            "support_ok": self.support_ok,
            "scope_violations": list(self.scope_violations),
            "support_gaps": list(self.support_gaps),
            "reasons": list(self.reasons),
        }


def evaluate_feasibility(
    bundle: DataBundle,
    model: DemandModel,
    ledger: RightsLedger,
    policy: Policy,
    period: str,
    cohort: list[JourneyEvidence],
    support_gaps: tuple[str, ...] = (),
    mode: TrialMode = TrialMode.FINALIZE,
    n_draws: int = 1000,
    exercise_prob: float = 1.0,
    rng: np.random.Generator | None = None,
) -> JointFeasibility:
    """对单个时段做联合可行性检验。"""
    rng = rng or np.random.default_rng()
    period_row = bundle.periods[period]
    obligation: PeriodObligation = ledger.period_obligation(period, policy)
    prices = {p: policy.plans[p] for p in PLANS}

    # 约束 2：旧价适用套餐范围
    scope_violations = []
    for view in obligation.active_views:
        missing = [p for p in view.alternatives if p not in policy.plans]
        if missing:
            scope_violations.append(
                f"报价 {view.quote} 的等价替代 {missing} 不在价表 "
                f"{policy.version} 适用范围内"
            )
    scope_ok = not scope_violations

    # 联合离散模拟：同一批 bootstrap 参数、旅程选择与权利行权抽样贯穿两项约束，
    # 严禁对容量与价差分别取最有利预测
    available = {p: True for p in PLANS}
    demand_dist = model.choice_simulation(
        cohort, available, prices, n_draws=n_draws, rng=rng
    )
    slot_draws = np.zeros(n_draws)
    credit_draws = np.zeros(n_draws)
    for d in range(n_draws):
        # 一件已售套餐占一个并行车位
        slot_draws[d] = sum(demand_dist[p][d] for p in PLANS)
        # 同一批回来行权的旅程：有效权利按行权概率抽样，
        # 补差取各权利等价替代范围内的最大正差预留口径
        credit = 0.0
        for view in obligation.active_views:
            if rng.random() < exercise_prob:
                credit += view.reserved_credit
        credit_draws[d] = credit

    total_slots = slot_draws + obligation.reserved_slots
    total_credit = credit_draws + obligation.reserved_credit
    level = policy.support_level
    q = lambda arr: (float(np.quantile(arr, (1 - level) / 2)),
                     float(np.quantile(arr, 1 - (1 - level) / 2)))
    slot_q = q(total_slots)
    credit_q = q(total_credit)

    capacity_bad = total_slots > period_row.fleet_hours + 1e-9
    credit_bad = total_credit > policy.credit_ceiling + 1e-9
    capacity_rate = float(capacity_bad.mean())
    credit_rate = float(credit_bad.mean())
    feasible_rate = float((~capacity_bad & ~credit_bad).mean())
    support_ok = not support_gaps

    reasons: list[str] = []
    if capacity_rate > 1 - level:
        reasons.append(
            f"售出车位在 {level:.0%} 支持水平下达 {slot_q[1]:.1f} 个，"
            f"超过时段可售量 {period_row.fleet_hours:.0f} 个"
        )
    if credit_rate > 1 - level:
        reasons.append(
            f"补差需求在 {level:.0%} 支持水平下达 {credit_q[1]:.0f} 元，"
            f"超过每时段上限 {policy.credit_ceiling:.0f} 元"
        )
    if not scope_ok:
        reasons.extend(scope_violations)
    if mode is TrialMode.FINALIZE and not support_ok:
        reasons.append("存在共同支持缺口，超范围价格只能试探不能定案")

    if mode is TrialMode.TRIAL:
        decision = "允许试算（风险已量化，不得据此定案）"
        if reasons:
            decision = "仅可试探：" + "；".join(reasons)
    else:
        decision = (
            "可定案"
            if feasible_rate >= level and scope_ok and support_ok
            else "暂缓定案：" + "；".join(reasons)
        )

    return JointFeasibility(
        period=period,
        policy_version=policy.version,
        mode=mode,
        n_draws=n_draws,
        sellable_slots=period_row.fleet_hours,
        credit_ceiling=policy.credit_ceiling,
        slot_draws=total_slots,
        credit_draws=total_credit,
        demand_draws=demand_dist,
        reserved_slots=obligation.reserved_slots,
        reserved_credit=obligation.reserved_credit,
        scope_violations=tuple(scope_violations),
        support_gaps=tuple(support_gaps),
        feasible_rate=feasible_rate,
        capacity_violation_rate=capacity_rate,
        credit_violation_rate=credit_rate,
        scope_ok=scope_ok,
        support_ok=support_ok,
        slot_quantile=slot_q,
        credit_quantile=credit_q,
        decision=decision,
        reasons=tuple(reasons),
    )
