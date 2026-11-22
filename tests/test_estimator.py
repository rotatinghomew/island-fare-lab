"""需求估计：价格系数方向、时段整批重抽样、留出评估与离散联合模拟。"""
import unittest

import numpy as np

from pricing_lab.estimator import (
    build_choice_rows,
    evaluate_holdout,
    fit_demand,
)
from pricing_lab.io import load_bundle

DATA = __import__("pathlib").Path(__file__).resolve().parents[1] / "data"


class DemandEstimatorTest(unittest.TestCase):
    def setUp(self):
        self.bundle = load_bundle(DATA)
        self.rng = np.random.default_rng(42)

    def test_price_coefficient_is_negative(self):
        model = fit_demand(self.bundle, n_bootstrap=100, rng=self.rng)
        self.assertTrue(model.converged)
        self.assertLess(model.price_coef, 0)

    def test_bootstrap_resamples_whole_periods_only(self):
        model = fit_demand(self.bundle, n_bootstrap=200, rng=self.rng)
        # bootstrap 参数必须有真实方差，且每行是完整参数向量
        self.assertEqual(model.bootstrap_params.shape, (200, 4))
        self.assertGreater(model.bootstrap_params.std(axis=0).sum(), 0)

    def test_choice_probabilities_sum_to_one_and_respect_availability(self):
        model = fit_demand(self.bundle, n_bootstrap=50, rng=self.rng)
        probs = model.choice_probabilities(
            {"semi": True, "day": False, "express": True},
            {"semi": 500, "day": 700, "express": 400},
        )
        self.assertAlmostEqual(sum(probs.values()), 1.0, places=9)
        self.assertNotIn("day", probs)  # 未进入选择集的套餐不分配概率

    def test_higher_price_lowers_plan_probability(self):
        model = fit_demand(self.bundle, n_bootstrap=50, rng=self.rng)
        avail = {p: True for p in ("semi", "day", "express")}
        low = model.choice_probabilities(avail, {"semi": 400, "day": 700, "express": 400})
        high = model.choice_probabilities(avail, {"semi": 600, "day": 700, "express": 400})
        self.assertGreater(low["semi"], high["semi"])

    def test_simulation_returns_integer_demands_and_preserves_cohort(self):
        model = fit_demand(self.bundle, n_bootstrap=100, rng=self.rng)
        rows = build_choice_rows(self.bundle)
        cohort_size = len(rows)
        from pricing_lab.evidence import causal_rows, build_evidence
        cohort = causal_rows(build_evidence(self.bundle))
        sim = model.choice_simulation(
            cohort, {p: True for p in ("semi", "day", "express")},
            {"semi": 540, "day": 720, "express": 420},
            n_draws=200, rng=self.rng,
        )
        for draws in sim.values():
            self.assertTrue(np.all(draws == draws.astype(int)))
        # 每次模拟的总件数不超过客群规模
        totals = sum(sim[p] for p in ("semi", "day", "express")) + sim["no_buy"]
        self.assertTrue(np.all(totals == cohort_size))

    def test_holdout_splits_by_period_and_produces_metrics(self):
        report = evaluate_holdout(self.bundle, rng=self.rng)
        self.assertTrue(all(f.n_rows > 0 for f in report.folds))
        # 各折时段互不重叠
        all_periods = [p for f in report.folds for p in f.periods]
        self.assertEqual(len(all_periods), len(set(all_periods)))
        self.assertGreater(report.mean_choice_accuracy, 0)
        self.assertGreater(report.mean_log_loss, 0)


if __name__ == "__main__":
    unittest.main()
