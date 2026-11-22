"""资料读取、时间快照、异常体检与承诺防覆盖。"""
import tempfile
import unittest
from datetime import datetime
from pathlib import Path

from pricing_lab.commitments import CommitmentRegistry, CommittedQuote, ManualQueue
from pricing_lab.io import load_bundle, snapshot
from pricing_lab.models import DataIssue

DATA = Path(__file__).resolve().parents[1] / "data"


class IOTest(unittest.TestCase):
    def setUp(self):
        self.bundle = load_bundle(DATA)

    def test_late_fulfilment_hidden_before_received_day(self):
        # J0002 对账记录 6/7 才可见
        early = snapshot(
            self.bundle, datetime.fromisoformat("2026-06-03T23:59:59+08:00")
        )
        self.assertIsNone(early.fulfilment_of("J0002"))
        full = snapshot(
            self.bundle, datetime.fromisoformat("2026-06-08T00:00:00+08:00")
        )
        self.assertEqual(full.fulfilment_of("J0002").plan, "day")

    def test_exposure_visibility_uses_seen_at(self):
        early = snapshot(
            self.bundle, datetime.fromisoformat("2026-06-01T09:30:00+08:00")
        )
        # J0001 的酒店触达 10:00 才出现
        self.assertEqual(len(early.exposures_of("J0001")), 1)

    def test_paid_price_mismatch_goes_to_issues_not_silently_rewritten(self):
        codes = {i.code for i in self.bundle.issues}
        self.assertIn("paid_price_mismatch", codes)
        mismatch = next(i for i in self.bundle.issues
                        if i.code == "paid_price_mismatch")
        # 实收事实未被改写
        f = self.bundle.fulfilment_of(mismatch.refs["journey"])
        self.assertEqual(f.paid, mismatch.refs["paid"])

    def test_closed_period_missing_fulfilment_flagged(self):
        codes = {i.code for i in self.bundle.issues}
        self.assertIn("closed_period_missing_fulfilment", codes)


class CommitmentTest(unittest.TestCase):
    def test_confirmed_quote_cannot_be_overridden_by_new_data(self):
        path = Path(tempfile.mktemp(suffix=".json"))
        try:
            reg = CommitmentRegistry(path)
            reg.confirm(CommittedQuote(
                "J1", "直订", "semi", 480,
                datetime.fromisoformat("2026-06-01T09:00:00+08:00"), "E1", "首次",
            ))
            with self.assertRaises(ValueError):
                reg.confirm(CommittedQuote(
                    "J1", "直订", "semi", 470,
                    datetime.fromisoformat("2026-06-02T09:00:00+08:00"),
                    "E2", "渠道补回",
                ))
            reg.supersede("J1", "定价会批准")
            reg.confirm(CommittedQuote(
                "J1", "直订", "semi", 470,
                datetime.fromisoformat("2026-06-02T10:00:00+08:00"),
                "E2", "人工变更",
            ))
            self.assertEqual(reg.active_for("J1").price, 470)
            reloaded = CommitmentRegistry(path)
            self.assertEqual(reloaded.active_for("J1").price, 470)
        finally:
            path.unlink(missing_ok=True)

    def test_manual_queue_groups_by_type(self):
        queue = ManualQueue()
        queue.add_issue(DataIssue("paid_price_mismatch", "m", {}))
        queue.add("right_manual", "r")
        self.assertEqual(len(queue.by_type("paid_price_mismatch")), 1)
        self.assertEqual(len(queue.by_type("right_manual")), 1)


if __name__ == "__main__":
    unittest.main()
