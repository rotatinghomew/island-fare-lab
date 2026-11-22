"""旧报价权利台账。

口径：
- 同一报价的多版本状态按 ``change_at`` 形成时间线，快照取截至时点的最新版本；
- 有效（未过期、未行权、未放弃）权利在所属时段持续受保护；
  到期或明确放弃即释放，行权后转为已兑现义务；
- 受保护旅程只占一份容量：一份有效权利预留一个并行车位，不按替代数重复占量；
- 价差预算 = 新价表下各可行权套餐「发布价 − 旧报价」的正差，按最大正差预留
  （字典：企业承担的正差）；旧报价高于发布价时不产生补差；
- 容量（车位）与价差落在同一批回来行权的旅程上，台账同时给出两者的联合口径。
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import Enum

from .io import DataBundle
from .models import Policy, QuoteRight


class RightState(str, Enum):
    ACTIVE = "有效"
    EXERCISED = "已行权"
    WAIVED = "放弃"
    EXPIRED = "过期"
    NOT_YET_VISIBLE = "尚未可见"


@dataclass(frozen=True)
class RightView:
    quote: str
    period: str
    origin: str
    old_price: float
    alternatives: tuple[str, ...]
    until: datetime
    state: RightState
    latest_change_at: datetime
    reserved_slots: float
    # 各替代套餐在新价表下的企业补差（正差）
    credit_by_plan: dict[str, float]
    reserved_credit: float
    note: str = ""

    @property
    def protected(self) -> bool:
        return self.state is RightState.ACTIVE

    @property
    def capacity_released(self) -> bool:
        return self.state in (RightState.WAIVED, RightState.EXPIRED)


@dataclass(frozen=True)
class PeriodObligation:
    period: str
    active_views: tuple[RightView, ...]
    exercised_views: tuple[RightView, ...]
    released_views: tuple[RightView, ...]
    reserved_slots: float
    reserved_credit: float

    def counts(self) -> dict[str, int]:
        return {
            "有效": len(self.active_views),
            "已行权": len(self.exercised_views),
            "已释放": len(self.released_views),
        }


class RightsLedger:
    def __init__(self, bundle: DataBundle, as_of: datetime | None = None):
        self.bundle = bundle
        self.as_of = as_of
        self._timelines: dict[str, list[QuoteRight]] = {}
        for right in bundle.quote_rights:
            if as_of is not None and right.change_at > as_of:
                continue
            self._timelines.setdefault(right.quote, []).append(right)
        for rows in self._timelines.values():
            rows.sort(key=lambda r: r.change_at)

    # ---- 单个报价 ----
    def view(self, quote: str, policy: Policy | None = None) -> RightView:
        rows = self._timelines.get(quote)
        if not rows:
            raise KeyError(f"快照下报价 {quote} 尚无可见状态")
        latest = rows[-1]
        state = RightState(latest.state)
        note = ""
        if state is RightState.ACTIVE and self.as_of is not None \
                and latest.until <= self.as_of:
            state = RightState.EXPIRED
            note = f"已于 {latest.until.isoformat()} 到期，车位与价差预算释放"
        credit_by_plan, reserved_credit = self._credits(latest, policy)
        reserved_slots = 1.0 if state is RightState.ACTIVE else 0.0
        if state in (RightState.WAIVED, RightState.EXPIRED):
            note = note or "权利已释放，不再占用车位与价差预算"
        return RightView(
            quote=latest.quote,
            period=latest.period,
            origin=latest.origin,
            old_price=latest.price,
            alternatives=latest.alternatives,
            until=latest.until,
            state=state,
            latest_change_at=latest.change_at,
            reserved_slots=reserved_slots,
            credit_by_plan=credit_by_plan,
            reserved_credit=reserved_credit if state is RightState.ACTIVE else 0.0,
            note=note,
        )

    def all_views(self, policy: Policy | None = None) -> list[RightView]:
        return [self.view(q, policy) for q in sorted(self._timelines)]

    # ---- 时段义务汇总 ----
    def period_obligation(
        self, period: str, policy: Policy | None = None
    ) -> PeriodObligation:
        views = [
            self.view(q, policy)
            for q in sorted(self._timelines)
            if self._timelines[q][-1].period == period
        ]
        active = tuple(v for v in views if v.state is RightState.ACTIVE)
        exercised = tuple(v for v in views if v.state is RightState.EXERCISED)
        released = tuple(v for v in views if v.capacity_released)
        return PeriodObligation(
            period=period,
            active_views=active,
            exercised_views=exercised,
            released_views=released,
            reserved_slots=sum(v.reserved_slots for v in active),
            reserved_credit=sum(v.reserved_credit for v in active),
        )

    def obligations(
        self, policy: Policy | None = None
    ) -> dict[str, PeriodObligation]:
        periods = {self._timelines[q][-1].period for q in self._timelines}
        return {p: self.period_obligation(p, policy) for p in sorted(periods)}

    # ---- 内部口径 ----
    def _credits(
        self, right: QuoteRight, policy: Policy | None
    ) -> tuple[dict[str, float], float]:
        if policy is None:
            return {}, 0.0
        credit = {}
        for plan in right.alternatives:
            if plan not in policy.plans:
                continue
            credit[plan] = max(policy.plans[plan] - right.price, 0.0)
        return credit, max(credit.values(), default=0.0)
