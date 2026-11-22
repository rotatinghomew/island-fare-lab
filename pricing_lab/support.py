"""共同支持（common support）诊断。

随机轮换中实际观察到的价格档与客群构成证据支持域：
- 策略价落在观察到的价格网格内，且该客群在对应套餐上有 ``seen`` 证据，才算有支持；
- 超出共同支持的报价方案只能当试探，输出缺证的价格档与客群，不得当作可采用结论；
- 非随机时段与自选旧价旅程不贡献支持证据。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from .areas import area_in_scope
from .evidence import EvidenceClass, JourneyEvidence, build_evidence
from .io import DataBundle
from .models import PLANS, Policy


class SupportVerdict(str, Enum):
    SUPPORTED = "有证据可采用"
    EXTRAPOLATION = "超支持范围仅可试探"


@dataclass(frozen=True)
class PlanSupport:
    plan: str
    policy_price: float
    observed_prices: tuple[float, ...]
    verdict: SupportVerdict
    segments_observed: tuple[str, ...]
    segments_missing: tuple[str, ...]


@dataclass(frozen=True)
class SupportReport:
    policy_version: str
    area: str
    plan_reports: tuple[PlanSupport, ...]
    overall: SupportVerdict
    gaps: tuple[str, ...] = field(default_factory=tuple)

    def as_dict(self) -> dict:
        return {
            "policy_version": self.policy_version,
            "area": self.area,
            "overall": self.overall.value,
            "gaps": list(self.gaps),
            "plans": [
                {
                    "plan": r.plan,
                    "policy_price": r.policy_price,
                    "observed_prices": list(r.observed_prices),
                    "verdict": r.verdict.value,
                    "segments_observed": list(r.segments_observed),
                    "segments_missing": list(r.segments_missing),
                }
                for r in self.plan_reports
            ],
        }


def observed_price_grid(
    evidence: dict[str, JourneyEvidence], area: str | None = None
) -> dict[str, set[float]]:
    """因果可比旅程在随机轮换触达中实际看到的价格网格。"""
    grid: dict[str, set[float]] = {p: set() for p in PLANS}
    for ev in evidence.values():
        if ev.evidence_class is not EvidenceClass.RANDOM:
            continue
        if area is not None and not area_in_scope(ev.area, area):
            continue
        for plan in PLANS:
            if ev.plan_evidence[plan] == "seen":
                grid[plan].update(ev.seen_prices[plan])
    return grid


def observed_segments(
    evidence: dict[str, JourneyEvidence], plan: str, area: str | None = None
) -> set[str]:
    out = set()
    for ev in evidence.values():
        if ev.evidence_class is not EvidenceClass.RANDOM:
            continue
        if area is not None and not area_in_scope(ev.area, area):
            continue
        if ev.plan_evidence[plan] == "seen":
            out.add(ev.segment)
    return out


def support_matrix(bundle: DataBundle, policy: Policy) -> SupportReport:
    evidence = build_evidence(bundle)
    all_segments = sorted({ev.segment for ev in evidence.values()})
    grid = observed_price_grid(evidence, area=policy.area)
    plan_reports: list[PlanSupport] = []
    gaps: list[str] = []
    for plan in PLANS:
        price = float(policy.plans[plan])
        observed = tuple(sorted(grid[plan]))
        seg_seen = tuple(sorted(observed_segments(evidence, plan, policy.area)))
        seg_missing = tuple(s for s in all_segments if s not in seg_seen)
        in_grid = price in grid[plan]
        verdict = (
            SupportVerdict.SUPPORTED if in_grid else SupportVerdict.EXTRAPOLATION
        )
        if not in_grid:
            gaps.append(
                f"{plan} 拟刊价 {price:.0f} 元不在 {policy.area} 观察价格档 "
                f"{[round(x) for x in observed]} 内，属超支持试探"
            )
        if seg_missing:
            gaps.append(
                f"{plan} 在客群 {list(seg_missing)} 没有任何可见证据"
            )
        plan_reports.append(
            PlanSupport(
                plan=plan,
                policy_price=price,
                observed_prices=observed,
                verdict=verdict,
                segments_observed=seg_seen,
                segments_missing=seg_missing,
            )
        )
    overall = (
        SupportVerdict.SUPPORTED
        if all(r.verdict is SupportVerdict.SUPPORTED and not r.segments_missing
               for r in plan_reports)
        else SupportVerdict.EXTRAPOLATION
    )
    return SupportReport(
        policy_version=policy.version,
        area=policy.area,
        plan_reports=tuple(plan_reports),
        overall=overall,
        gaps=tuple(gaps),
    )
