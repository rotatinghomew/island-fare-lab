"""证据分层与旅程分类。

两类证据严格分开（业务口径：旅客没有看到某套餐，与看到后没有选择，属于不同证据）：
- ``seen``：该套餐在随机轮换触达中实际出现在可购买选项里；
- ``unseen``：该套餐未出现在任何随机轮换触达中，只能作为「缺失证据」，
  不能当作旅客拒绝该套餐。

旅程分类决定能否进入随机轮换的因果比较：
- ``RANDOM``：随机时段且全程未出现可自行选择旧价的触达，可进入因果比较；
- ``OLD_CHOICE_DIAGNOSTIC``：出现过 ``old_choice`` 触达，分配被旅客自选行为污染，
  仅作适用性诊断；
- ``NON_RANDOM_DIAGNOSTIC``：渠道临时关闭（``random=false``）时段，仅作诊断；
- ``INELIGIBLE``：快照下尚不可见的旅程。
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from .io import DataBundle
from .models import Exposure, Journey, PLANS


class EvidenceClass(str, Enum):
    RANDOM = "random"
    OLD_CHOICE_DIAGNOSTIC = "old_choice_diagnostic"
    NON_RANDOM_DIAGNOSTIC = "non_random_diagnostic"
    INELIGIBLE = "ineligible"


@dataclass(frozen=True)
class JourneyEvidence:
    """单个旅程在快照下的全部可分析证据。"""

    journey: str
    period: str
    arm: str
    area: str
    segment: str
    random_period: bool
    evidence_class: EvidenceClass
    random_exposures: tuple[Exposure, ...]
    old_choice_exposures: tuple[Exposure, ...]
    chosen_plan: str | None
    paid: float | None
    use: str | None
    # 套餐级证据标签：seen / unseen（仅统计随机轮换触达）
    plan_evidence: dict[str, str]
    # 随机轮换触达中实际出现过的价格（套餐 -> 价格集合）
    seen_prices: dict[str, tuple[float, ...]]
    # 随机轮换触达中出现过相互矛盾的价格，需诊断
    conflicting_prices: bool

    @property
    def causal(self) -> bool:
        return self.evidence_class is EvidenceClass.RANDOM

    @property
    def converted(self) -> bool:
        return self.chosen_plan is not None

    @property
    def realized_use(self) -> bool:
        """最终实际使用（未到场计为已转化但未使用）。"""
        return self.use == "已使用"


def segment_of(journey: Journey) -> str:
    """客群分档：请求时长 × 车位条件。"""
    length = "长时" if journey.length_hours >= 8 else "短时"
    return f"{length}/{journey.needs}"


def classify_journey(bundle: DataBundle, journey_id: str) -> JourneyEvidence:
    journey = bundle.journeys[journey_id]
    period = bundle.periods[journey.period]
    exposures = bundle.exposures_of(journey_id)
    if not exposures:
        evidence_class = EvidenceClass.INELIGIBLE
    elif not period.random:
        evidence_class = EvidenceClass.NON_RANDOM_DIAGNOSTIC
    elif any(e.old_choice for e in exposures):
        evidence_class = EvidenceClass.OLD_CHOICE_DIAGNOSTIC
    else:
        evidence_class = EvidenceClass.RANDOM

    random_exposures = tuple(e for e in exposures if not e.old_choice)
    old_exposures = tuple(e for e in exposures if e.old_choice)

    plan_evidence: dict[str, str] = {}
    seen_prices: dict[str, list[float]] = {p: [] for p in PLANS}
    for exp in random_exposures:
        for plan, price in exp.price_map.items():
            plan_evidence[plan] = "seen"
            seen_prices[plan].append(price)
    for plan in PLANS:
        plan_evidence.setdefault(plan, "unseen")

    conflicting = any(len(set(prices)) > 1 for prices in seen_prices.values())

    fulfilment = bundle.fulfilment_of(journey_id)
    return JourneyEvidence(
        journey=journey_id,
        period=journey.period,
        arm=period.arm,
        area=period.area,
        segment=segment_of(journey),
        random_period=period.random,
        evidence_class=evidence_class,
        random_exposures=random_exposures,
        old_choice_exposures=old_exposures,
        chosen_plan=fulfilment.plan if fulfilment else None,
        paid=fulfilment.paid if fulfilment else None,
        use=fulfilment.use if fulfilment else None,
        plan_evidence=plan_evidence,
        seen_prices={k: tuple(sorted(v)) for k, v in seen_prices.items()},
        conflicting_prices=conflicting,
    )


def build_evidence(bundle: DataBundle) -> dict[str, JourneyEvidence]:
    """对快照内全部旅程建立证据档案。"""
    return {jid: classify_journey(bundle, jid) for jid in bundle.journeys}


def causal_rows(evidence: dict[str, JourneyEvidence]) -> list[JourneyEvidence]:
    """可进入随机轮换因果比较的旅程。"""
    return [e for e in evidence.values() if e.causal]


def diagnostic_rows(evidence: dict[str, JourneyEvidence]) -> list[JourneyEvidence]:
    """仅作适用性诊断、不得并入随机比较的旅程。"""
    return [
        e
        for e in evidence.values()
        if e.evidence_class
        in (EvidenceClass.OLD_CHOICE_DIAGNOSTIC, EvidenceClass.NON_RANDOM_DIAGNOSTIC)
    ]
