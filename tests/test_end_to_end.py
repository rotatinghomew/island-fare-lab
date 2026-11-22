"""端到端试算与历史留出期对账。"""
import unittest

import numpy as np

from pricing_lab import PricingTrial
from pricing_lab.feasibility import TrialMode
from pricing_lab.io import load_bundle

DATA = __import__("pathlib").Path(__file__).resolve().parents[1] / "data"


class EndToEndTest(unittest.TestCase):
    def setUp(self):
        self.bundle = load_bundle(DATA)
        self.trial = PricingTrial(self.bundle, seed=123)
        self.trial.fit(n_bootstrap=200)

    def test_report_partitions_all_journeys_once(self):
        report = self.trial.run(mode=TrialMode.TRIAL, n_draws=200)
        counts = report.class_counts
        self.assertEqual(sum(counts.values()), len(self.bundle.journeys))
        # 诊断旅程数与原始数据一致（65 自选 + 22 非随机）
        self.assertEqual(counts["old_choice_diagnostic"], 65)
        self.assertEqual(counts["non_random_diagnostic"], 22)

    def test_ladder_b_is_trial_only_never_finalizable(self):
        report = self.trial.run(mode=TrialMode.FINALIZE, n_draws=200)
        result = report.policies["ladder-b"]
        self.assertEqual(result.finalizable_periods, [])
        self.assertTrue(result.blocked_periods)
        self.assertTrue(result.demand.support_gaps)

    def test_substitution_direction_matches_price_change(self):
        report = self.trial.run(mode=TrialMode.TRIAL, n_draws=200)
        sub = report.policies["ladder-a"].demand.substitution
        change = sub["demand_change_by_plan"]
        # semi 480→540 涨价，day 780→720 降价：semi 降、day 升
        self.assertLess(change["semi"], 0)
        self.assertGreater(change["day"], 0)

    def test_reconcile_closed_period_covers_actuals(self):
        rec = self.trial.reconcile(
            self.bundle.policies["base"], "P00", n_draws=300
        )
        self.assertTrue(rec.closed)
        self.assertEqual(rec.total_count, 6)  # 3 需求 + 车时 + 车位 + 毛利
        # 历史留出期的实际值应大多落在区间内
        self.assertGreaterEqual(rec.covered_count, 4)

    def test_manual_queue_carries_data_issues(self):
        report = self.trial.run(mode=TrialMode.TRIAL, n_draws=100)
        types = {item["type"] for item in report.manual_queue.items}
        self.assertIn("paid_price_mismatch", types)


if __name__ == "__main__":
    unittest.main()
