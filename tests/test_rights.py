"""旧价权利台账：多版本状态、过期/放弃释放、一份权利一个车位、补差正差。"""
import unittest
from datetime import datetime

from pricing_lab.io import load_bundle
from pricing_lab.rights import RightsLedger, RightState

DATA = __import__("pathlib").Path(__file__).resolve().parents[1] / "data"


class RightsLedgerTest(unittest.TestCase):
    def setUp(self):
        self.bundle = load_bundle(DATA)
        self.policy = self.bundle.policies["ladder-a"]

    def test_latest_state_wins_in_timeline(self):
        ledger = RightsLedger(self.bundle)
        r2 = ledger.view("R2")  # 6/21 有效，6/24 放弃
        self.assertIs(r2.state, RightState.WAIVED)
        self.assertEqual(r2.reserved_slots, 0)
        self.assertTrue(r2.capacity_released)

    def test_snapshot_hides_future_versions(self):
        as_of = datetime.fromisoformat("2026-06-22T12:00:00+08:00")
        ledger = RightsLedger(self.bundle, as_of=as_of)
        self.assertIs(ledger.view("R2").state, RightState.ACTIVE)
        with self.assertRaises(KeyError):
            ledger.view("R4")  # 首个状态版本 6/24 才可见

    def test_expiry_releases_capacity_and_credit(self):
        as_of = datetime.fromisoformat("2026-06-25T12:00:00+08:00")
        ledger = RightsLedger(self.bundle, as_of=as_of)
        r1 = ledger.view("R1", self.policy)
        self.assertIs(r1.state, RightState.EXPIRED)
        self.assertEqual(r1.reserved_slots, 0)
        self.assertEqual(r1.reserved_credit, 0)

    def test_one_right_occupies_one_slot_only(self):
        ledger = RightsLedger(self.bundle)
        r1 = ledger.view("R1", self.policy)  # 替代含 day、semi 两个套餐
        self.assertEqual(r1.reserved_slots, 1.0)

    def test_credit_is_positive_gap_against_published_price(self):
        ledger = RightsLedger(self.bundle)
        r1 = ledger.view("R1", self.policy)
        # 旧价 650；ladder-a: day 720 → 补差 70，semi 540 → 不补差
        self.assertEqual(r1.credit_by_plan["day"], 70)
        self.assertEqual(r1.credit_by_plan["semi"], 0)
        self.assertEqual(r1.reserved_credit, 70)  # 按最大正差预留

    def test_period_obligation_sums_protected_only(self):
        ledger = RightsLedger(self.bundle)
        ob = ledger.period_obligation("P23", self.policy)
        # R1 有效、R2 放弃：仅 1 个车位、70 元补差
        self.assertEqual(ob.reserved_slots, 1.0)
        self.assertEqual(ob.reserved_credit, 70.0)
        self.assertEqual(ob.counts()["已释放"], 1)


if __name__ == "__main__":
    unittest.main()
