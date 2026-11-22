"""毛利口径与共同支持诊断。"""
import unittest

from pricing_lab.io import load_bundle
from pricing_lab.margin import gross_margin_components
from pricing_lab.support import SupportVerdict, support_matrix

DATA = __import__("pathlib").Path(__file__).resolve().parents[1] / "data"


class MarginTest(unittest.TestCase):
    def setUp(self):
        self.bundle = load_bundle(DATA)

    def test_margin_deducts_all_variable_costs(self):
        m = gross_margin_components(self.bundle, "semi", 540)
        # 4 车时 × 45 + 30 接驳 + 18 优惠 = 228
        self.assertEqual(m.vehicle_cost, 180)
        self.assertEqual(m.transfer, 30)
        self.assertEqual(m.operator_coupon, 18)
        self.assertEqual(m.unit_margin, 540 - 228)

    def test_substitution_changes_cost_with_plan(self):
        semi = gross_margin_components(self.bundle, "semi", 500).unit_margin
        day = gross_margin_components(self.bundle, "day", 720).unit_margin
        # 替换套餐后车时成本不同，单位毛利随之变化
        self.assertNotEqual(semi, day)


class SupportTest(unittest.TestCase):
    def setUp(self):
        self.bundle = load_bundle(DATA)

    def test_observed_grid_marks_supported_policy(self):
        report = support_matrix(self.bundle, self.bundle.policies["ladder-a"])
        self.assertIs(report.overall, SupportVerdict.SUPPORTED)
        for plan in ("semi", "day", "express"):
            pr = next(r for r in report.plan_reports if r.plan == plan)
            self.assertIs(pr.verdict, SupportVerdict.SUPPORTED)

    def test_out_of_grid_policy_is_extrapolation_with_gaps(self):
        report = support_matrix(self.bundle, self.bundle.policies["ladder-b"])
        self.assertIs(report.overall, SupportVerdict.EXTRAPOLATION)
        self.assertTrue(report.gaps)
        # 港城只观察到 480/540 的 semi，580 缺证
        semi = next(r for r in report.plan_reports if r.plan == "semi")
        self.assertNotIn(580.0, semi.observed_prices)

    def test_combined_area_pooles_both_sub_areas(self):
        # 组合范围「海湾和港城」的观察网格不窄于任一单一区域
        combined = support_matrix(self.bundle, self.bundle.policies["base"])
        harbor_only = support_matrix(
            self.bundle, self.bundle.policies["ladder-b"]
        )
        combined_semi = next(
            r for r in combined.plan_reports if r.plan == "semi"
        ).observed_prices
        harbor_semi = next(
            r for r in harbor_only.plan_reports if r.plan == "semi"
        ).observed_prices
        self.assertGreaterEqual(len(combined_semi), len(harbor_semi))


if __name__ == "__main__":
    unittest.main()
