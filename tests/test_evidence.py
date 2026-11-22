"""证据分层：同一旅程多渠道询价仍归一；没看到 ≠ 看到没选；
自选旧价与非随机时段只作诊断。
"""
import unittest

from pricing_lab.evidence import (
    EvidenceClass,
    build_evidence,
    causal_rows,
    diagnostic_rows,
)
from pricing_lab.io import load_bundle

DATA = __import__("pathlib").Path(__file__).resolve().parents[1] / "data"


class EvidenceTest(unittest.TestCase):
    def setUp(self):
        self.bundle = load_bundle(DATA)
        self.evidence = build_evidence(self.bundle)

    def test_multi_channel_same_journey_is_one_unit(self):
        # J0001 在两个渠道询价，仍是一条旅程证据而非两个新客
        ev = self.evidence["J0001"]
        self.assertEqual(len(ev.random_exposures) + len(ev.old_choice_exposures), 2)
        self.assertEqual(ev.chosen_plan, "day")

    def test_seen_vs_unseen_are_distinct_evidence(self):
        # 存在随机旅程从未看到 day（semi+express 选项），day 必须标 unseen
        unseen_day = [
            ev for ev in causal_rows(self.evidence)
            if ev.plan_evidence["day"] == "unseen"
        ]
        self.assertTrue(unseen_day)
        for ev in unseen_day:
            # unseen 套餐不得出现在 seen_prices 中
            self.assertEqual(ev.seen_prices["day"], ())
            # 看到过的套餐必须标 seen
            self.assertEqual(ev.plan_evidence["semi"], "seen")

    def test_old_choice_journeys_are_diagnostic_only(self):
        old = [
            ev for ev in self.evidence.values()
            if ev.evidence_class is EvidenceClass.OLD_CHOICE_DIAGNOSTIC
        ]
        self.assertTrue(old)
        causal_ids = {ev.journey for ev in causal_rows(self.evidence)}
        self.assertFalse(any(ev.journey in causal_ids for ev in old))

    def test_non_random_periods_are_diagnostic_only(self):
        diag = diagnostic_rows(self.evidence)
        non_random = [
            ev for ev in diag
            if ev.evidence_class is EvidenceClass.NON_RANDOM_DIAGNOSTIC
        ]
        self.assertEqual({ev.period for ev in non_random}, {"P09", "P22"})

    def test_no_journey_double_counted_in_causal_set(self):
        rows = causal_rows(self.evidence)
        self.assertEqual(len(rows), len({ev.journey for ev in rows}))


if __name__ == "__main__":
    unittest.main()
