"""联合可行性：同一批抽样守容量与补差；支持缺口阻断定案但允许试算。"""
import unittest

import numpy as np

from pricing_lab.evidence import build_evidence, causal_rows
from pricing_lab.feasibility import TrialMode, evaluate_feasibility
from pricing_lab.io import load_bundle
from pricing_lab.rights import RightsLedger
from pricing_lab.support import SupportVerdict, support_matrix

DATA = __import__("pathlib").Path(__file__).resolve().parents[1] / "data"


class FeasibilityTest(unittest.TestCase):
    def setUp(self):
        self.bundle = load_bundle(DATA)
        self.rng = np.random.default_rng(7)
        from pricing_lab.estimator import fit_demand
        self.model = fit_demand(
            self.bundle, n_bootstrap=200, rng=self.rng
        )
        self.evidence = build_evidence(self.bundle)

    def _cohort(self, period, policy):
        from pricing_lab.areas import area_in_scope
        return [
            ev for ev in causal_rows(self.evidence)
            if ev.period == period and area_in_scope(ev.area, policy.area)
        ]

    def test_capacity_and_credit_share_same_draws(self):
        policy = self.bundle.policies["ladder-a"]
        ledger = RightsLedger(self.bundle)
        result = evaluate_feasibility(
            self.bundle, self.model, ledger, policy, "P23",
            self._cohort("P23", policy), mode=TrialMode.FINALIZE,
            n_draws=300, rng=self.rng,
        )
        # P23 有 R1 有效权利：车位与补差预留同时出现
        self.assertEqual(result.reserved_slots, 1.0)
        self.assertEqual(result.reserved_credit, 70.0)
        self.assertEqual(len(result.slot_draws), len(result.credit_draws))

    def test_support_gap_blocks_finalize_but_allows_trial(self):
        policy = self.bundle.policies["ladder-b"]
        support = support_matrix(self.bundle, policy)
        self.assertIs(support.overall, SupportVerdict.EXTRAPOLATION)
        ledger = RightsLedger(self.bundle)
        period = "P02"  # 港城随机时段
        trial = evaluate_feasibility(
            self.bundle, self.model, ledger, policy, period,
            self._cohort(period, policy), support_gaps=support.gaps,
            mode=TrialMode.TRIAL, n_draws=200, rng=self.rng,
        )
        final = evaluate_feasibility(
            self.bundle, self.model, ledger, policy, period,
            self._cohort(period, policy), support_gaps=support.gaps,
            mode=TrialMode.FINALIZE, n_draws=200, rng=self.rng,
        )
        self.assertIn("试算", trial.decision)
        self.assertTrue(final.decision.startswith("暂缓定案"))
        self.assertFalse(final.support_ok)

    def test_supported_policy_can_finalize_some_period(self):
        policy = self.bundle.policies["ladder-a"]
        ledger = RightsLedger(self.bundle)
        result = evaluate_feasibility(
            self.bundle, self.model, ledger, policy, "P00",
            self._cohort("P00", policy), mode=TrialMode.FINALIZE,
            n_draws=300, rng=self.rng,
        )
        self.assertEqual(result.decision, "可定案")
        self.assertTrue(result.scope_ok and result.support_ok)
        # 离散模拟：车位为整数
        self.assertTrue(np.all(result.slot_draws == result.slot_draws.astype(int)))


if __name__ == "__main__":
    unittest.main()
