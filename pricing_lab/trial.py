"""定价试算总编排：把证据分层、需求估计、支持域、毛利、旧价权利、
联合可行性与对账串成一次可复现的试算。
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

import numpy as np

from .areas import area_in_scope
from .commitments import ManualQueue
from .estimator import (
    DemandEstimate,
    DemandModel,
    HoldoutReport,
    evaluate_holdout,
    fit_demand,
)
from .evidence import (
    EvidenceClass,
    JourneyEvidence,
    build_evidence,
    causal_rows,
    diagnostic_rows,
)
from .feasibility import JointFeasibility, TrialMode, evaluate_feasibility
from .io import DataBundle
from .margin import gross_margin_components
from .models import PLANS, Policy
from .reconcile import ReconcileReport, reconcile_period
from .rights import RightsLedger
from .support import SupportReport, SupportVerdict, support_matrix


@dataclass
class PolicyTrialResult:
    policy_version: str
    support: SupportReport
    demand: DemandEstimate
    margin: dict
    feasibility: dict[str, JointFeasibility]
    diagnostics: dict
    finalizable_periods: list[str]
    blocked_periods: list[str]


@dataclass
class TrialReport:
    as_of: datetime
    n_journeys: int
    class_counts: dict[str, int]
    holdout: HoldoutReport
    policies: dict[str, PolicyTrialResult]
    manual_queue: ManualQueue
    diagnostics_summary: dict

    def as_dict(self) -> dict:
        return {
            "as_of": self.as_of.isoformat(),
            "n_journeys": self.n_journeys,
            "evidence_classes": self.class_counts,
            "holdout": {
                "mean_log_loss": round(self.holdout.mean_log_loss, 4),
                "mean_abs_error": round(self.holdout.mean_abs_error, 4),
                "choice_accuracy": round(self.holdout.mean_choice_accuracy, 4),
                "folds": [
                    {
                        "periods": list(f.periods),
                        "n_rows": f.n_rows,
                        "log_loss": round(f.log_loss, 4),
                        "mean_abs_error": round(f.mean_abs_error, 4),
                        "choice_accuracy": round(f.choice_accuracy, 4),
                    }
                    for f in self.holdout.folds
                ],
            },
            "diagnostics": self.diagnostics_summary,
            "policies": {
                version: {
                    "support": result.support.as_dict(),
                    "demand": _demand_to_dict(result.demand),
                    "margin": result.margin,
                    "finalizable_periods": result.finalizable_periods,
                    "blocked_periods": result.blocked_periods,
                    "feasibility": {
                        p: f.as_dict() for p, f in result.feasibility.items()
                    },
                    "diagnostics": result.diagnostics,
                }
                for version, result in self.policies.items()
            },
            "manual_queue": self.manual_queue.items,
        }


def _demand_to_dict(estimate: DemandEstimate) -> dict:
    return {
        "cohort_size": estimate.cohort_size,
        "baseline_demand": {k: round(v, 3) for k, v in estimate.baseline_demand.items()},
        "policy_demand": {k: round(v, 3) for k, v in estimate.policy_demand.items()},
        "demand_intervals": {
            k: [round(lo, 3), round(hi, 3)]
            for k, (lo, hi) in estimate.demand_intervals.items()
        },
        "substitution": estimate.substitution,
        "net_increment": round(estimate.net_increment, 3),
        "net_increment_interval": [
            round(estimate.net_increment_interval[0], 3),
            round(estimate.net_increment_interval[1], 3),
        ],
        "support_gaps": estimate.support_gaps,
    }


class PricingTrial:
    def __init__(self, bundle: DataBundle, seed: int = 20260601):
        self.bundle = bundle
        self.rng = np.random.default_rng(seed)
        self.evidence = build_evidence(bundle)
        self.ledger = RightsLedger(bundle)
        self.manual_queue = ManualQueue()
        for issue in bundle.issues:
            self.manual_queue.add_issue(issue)

    # ---- 需求模型与留出表现 ----
    def fit(self, n_bootstrap: int = 1000, cl_level: float = 0.9) -> DemandModel:
        self.cl_level = cl_level
        self.model = fit_demand(
            self.bundle, n_bootstrap=n_bootstrap,
            cl_level=cl_level, rng=self.rng,
        )
        self.holdout = evaluate_holdout(self.bundle, rng=self.rng)
        return self.model

    # ---- 单策略试算 ----
    def evaluate_policy(
        self,
        policy: Policy,
        mode: TrialMode = TrialMode.TRIAL,
        n_draws: int = 1000,
    ) -> PolicyTrialResult:
        if not hasattr(self, "model"):
            self.fit()
        support = support_matrix(self.bundle, policy)
        gaps = support.gaps

        cohort = [
            ev for ev in causal_rows(self.evidence)
            if area_in_scope(ev.area, policy.area)
        ]
        baseline = self.model.expected_demand(
            cohort, {p: True for p in PLANS},
            {p: self.bundle.policies["base"].plans[p] for p in PLANS},
        )
        policy_demand = self.model.expected_demand(
            cohort, {p: True for p in PLANS},
            {p: policy.plans[p] for p in PLANS},
        )
        dist = self.model.demand_distribution(
            cohort, {p: True for p in PLANS},
            {p: policy.plans[p] for p in PLANS},
            n_draws=n_draws, rng=self.rng,
        )
        intervals = {
            key: self.model.interval(samples, policy.support_level)
            for key, samples in dist.items()
        }
        substitution = self._substitution(cohort, policy)
        base_dist = self.model.demand_distribution(
            cohort, {p: True for p in PLANS},
            {p: self.bundle.policies["base"].plans[p] for p in PLANS},
            n_draws=n_draws, rng=self.rng,
        )
        net_draws = (
            sum(dist[p] for p in PLANS) - sum(base_dist[p] for p in PLANS)
        )
        demand_estimate = DemandEstimate(
            policy_version=policy.version,
            cohort_size=len(cohort),
            baseline_demand=baseline,
            policy_demand=policy_demand,
            demand_intervals=intervals,
            substitution=substitution,
            net_increment=float(net_draws.mean()),
            net_increment_interval=self.model.interval(
                net_draws, policy.support_level
            ),
            support_gaps=list(gaps),
        )

        margin = self._margin_compare(policy, dist, n_draws)
        diagnostics = self._diagnostics(policy)

        # 仅随机时段参与定案；有旧价权利的时段同样做联合检验
        feasibility = {}
        periods = sorted({
            ev.period for ev in cohort
            if self.bundle.periods[ev.period].random
        })
        ledger = RightsLedger(self.bundle)
        for period in periods:
            period_cohort = [ev for ev in cohort if ev.period == period]
            period_gaps = gaps if support.overall is SupportVerdict.EXTRAPOLATION else ()
            feasibility[period] = evaluate_feasibility(
                self.bundle, self.model, ledger, policy, period, period_cohort,
                support_gaps=tuple(period_gaps), mode=mode,
                n_draws=n_draws, rng=self.rng,
            )
        finalizable = [
            p for p, f in feasibility.items() if f.decision == "可定案"
        ]
        blocked = sorted(set(feasibility) - set(finalizable))
        return PolicyTrialResult(
            policy_version=policy.version,
            support=support,
            demand=demand_estimate,
            margin=margin,
            feasibility=feasibility,
            diagnostics=diagnostics,
            finalizable_periods=finalizable,
            blocked_periods=blocked,
        )

    # ---- 完整报告 ----
    def run(
        self, mode: TrialMode = TrialMode.TRIAL, n_draws: int = 1000
    ) -> TrialReport:
        if not hasattr(self, "model"):
            self.fit()
        results = {}
        for policy in self.bundle.policies.values():
            results[policy.version] = self.evaluate_policy(
                policy, mode=mode, n_draws=n_draws
            )
        class_counts = {
            cls.value: sum(
                1 for ev in self.evidence.values() if ev.evidence_class is cls
            )
            for cls in EvidenceClass
        }
        diagnostics_summary = {
            "non_random_periods": sorted(
                {ev.period for ev in self.evidence.values()
                 if ev.evidence_class is EvidenceClass.NON_RANDOM_DIAGNOSTIC}
            ),
            "old_choice_journeys": sorted(
                ev.journey for ev in diagnostic_rows(self.evidence)
                if ev.evidence_class is EvidenceClass.OLD_CHOICE_DIAGNOSTIC
            ),
            "conflicting_price_journeys": sorted(
                ev.journey for ev in self.evidence.values()
                if ev.conflicting_prices
            ),
        }
        return TrialReport(
            as_of=datetime.now().astimezone(),
            n_journeys=len(self.evidence),
            class_counts=class_counts,
            holdout=self.holdout,
            policies=results,
            manual_queue=self.manual_queue,
            diagnostics_summary=diagnostics_summary,
        )

    # ---- 对账 ----
    def reconcile(
        self, policy: Policy, period: str, n_draws: int = 1000
    ) -> ReconcileReport:
        if not hasattr(self, "model"):
            self.fit()
        cohort = [
            ev for ev in causal_rows(self.evidence)
            if ev.period == period and area_in_scope(ev.area, policy.area)
        ]
        ledger = RightsLedger(self.bundle)
        return reconcile_period(
            self.bundle, self.model, ledger, policy, period, cohort,
            n_draws=n_draws, rng=self.rng,
        )

    # ---- 内部方法 ----
    def _substitution(
        self, cohort: list[JourneyEvidence], policy: Policy
    ) -> dict:
        """替代与净增的可解释分解（只报告聚合数据上可识别的量）。

        - ``demand_change_by_plan``：各套餐相对基准价表的期望需求变化；
        - ``net_increment_from_no_buy``：总销量变化，即从未转化旅客转化的净增；
        - 逐对套餐流向在聚合选择数据上不可识别，不虚构。
        """
        base_prices = {p: self.bundle.policies["base"].plans[p] for p in PLANS}
        new_prices = {p: policy.plans[p] for p in PLANS}
        base = self.model.expected_demand(cohort, {p: True for p in PLANS}, base_prices)
        new = self.model.expected_demand(cohort, {p: True for p in PLANS}, new_prices)
        total_base = sum(base[p] for p in PLANS)
        total_new = sum(new[p] for p in PLANS)
        net_increment = total_new - total_base
        demand_change = {p: round(new[p] - base[p], 4) for p in PLANS}
        return {
            "demand_change_by_plan": demand_change,
            "net_increment_from_no_buy": round(net_increment, 4),
            "note": "逐对套餐替代流向在聚合选择数据上不可识别，仅报告可识别的总量分解",
        }

    def _margin_compare(
        self, policy: Policy, dist: dict, n_draws: int
    ) -> dict:
        base_policy = self.bundle.policies["base"]
        margin_draws = {"base": np.zeros(n_draws), policy.version: np.zeros(n_draws)}
        unit = {
            version: {
                p: gross_margin_components(
                    self.bundle, p, pol.plans[p]
                ).unit_margin
                for p in PLANS
            }
            for version, pol in (
                ("base", base_policy), (policy.version, policy)
            )
        }
        for d in range(n_draws):
            for p in PLANS:
                margin_draws[policy.version][d] += dist[p][d] * unit[policy.version][p]
        # 基准需求分布重算一遍保证两套口径同源
        cohort = [
            ev for ev in causal_rows(self.evidence)
            if area_in_scope(ev.area, policy.area)
        ]
        base_dist = self.model.demand_distribution(
            cohort, {p: True for p in PLANS},
            {p: base_policy.plans[p] for p in PLANS},
            n_draws=n_draws, rng=self.rng,
        )
        for d in range(n_draws):
            for p in PLANS:
                margin_draws["base"][d] += base_dist[p][d] * unit["base"][p]
        delta = margin_draws[policy.version] - margin_draws["base"]
        return {
            "unit_margin": {
                version: {p: round(v, 2) for p, v in rows.items()}
                for version, rows in unit.items()
            },
            "total_margin_interval": {
                version: [
                    round(float(np.quantile(draws, (1 - policy.support_level) / 2)), 2),
                    round(float(np.quantile(draws, 1 - (1 - policy.support_level) / 2)), 2),
                ]
                for version, draws in margin_draws.items()
            },
            "delta_margin_mean": round(float(delta.mean()), 2),
            "delta_margin_interval": [
                round(float(np.quantile(delta, (1 - policy.support_level) / 2)), 2),
                round(float(np.quantile(delta, 1 - (1 - policy.support_level) / 2)), 2),
            ],
            "cost_components": {
                p: {
                    "vehicle_cost": self.bundle.catalog[p].vehicle_cost,
                    "transfer": self.bundle.catalog[p].transfer,
                    "operator_coupon": self.bundle.catalog[p].operator_coupon,
                }
                for p in PLANS
            },
        }

    def _diagnostics(self, policy: Policy) -> dict:
        diag = diagnostic_rows(self.evidence)
        return {
            "old_choice_journeys": [
                ev.journey for ev in diag
                if ev.evidence_class is EvidenceClass.OLD_CHOICE_DIAGNOSTIC
                and area_in_scope(ev.area, policy.area)
            ],
            "non_random_periods": sorted(
                {
                    ev.period for ev in diag
                    if ev.evidence_class is EvidenceClass.NON_RANDOM_DIAGNOSTIC
                    and area_in_scope(ev.area, policy.area)
                }
            ),
            "note": "诊断样本不并入随机轮换因果比较，仅用于适用性判断",
        }
