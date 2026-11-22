#!/usr/bin/env python3
"""定价试算端到端演示：从资料加载到试算报告、权利台账与留出对账。

运行（在仓库根目录）：
    PYTHONPATH=. python3 scripts/run_trial_demo.py
数据路径以参数传入，脚本不依赖运行目录，也不从数据目录加载任何代码。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

# 允许在 python -I 隔离模式下从任意目录运行：显式加入仓库根目录
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from pricing_lab import PricingTrial, load_bundle
from pricing_lab.feasibility import TrialMode
from pricing_lab.rights import RightsLedger


def main(data_dir: str = "data") -> int:
    bundle = load_bundle(data_dir)
    trial = PricingTrial(bundle, seed=20260601)
    trial.fit(n_bootstrap=500)

    print("=" * 68)
    print("一、证据分层（同一旅程多渠道询价只算一个单位）")
    print("=" * 68)
    report = trial.run(mode=TrialMode.TRIAL, n_draws=500)
    for cls, n in report.class_counts.items():
        print(f"  {cls:28s} {n}")
    print(f"  留出表现：log_loss={report.holdout.mean_log_loss:.3f}，"
          f"选择准确率={report.holdout.mean_choice_accuracy:.3f}")

    final_report = trial.run(mode=TrialMode.FINALIZE, n_draws=500)

    print("\n" + "=" * 68)
    print("二、各价表：支持域、替代/净增、毛利与可定案时段")
    print("=" * 68)
    for version, result in report.policies.items():
        d = result.demand
        print(f"\n【{version}】支持结论：{result.support.overall.value}")
        for gap in result.support.gaps:
            print(f"  缺证：{gap}")
        print(f"  替代/净增：{json.dumps(d.substitution, ensure_ascii=False)}")
        print(f"  净增需求区间：{d.net_increment_interval}")
        print(f"  毛利差（相对基准，元）：{result.margin['delta_margin_interval']}")
        final_result = final_report.policies[version]
        print(f"  试算模式：可定案 {len(result.finalizable_periods)} 个，"
              f"受阻 {len(result.blocked_periods)} 个")
        print(f"  定案模式：可定案 {len(final_result.finalizable_periods)} 个，"
              f"受阻 {len(final_result.blocked_periods)} 个")

    print("\n" + "=" * 68)
    print("三、旧价权利：预留/兑现/释放义务（ladder-a 补差口径）")
    print("=" * 68)
    ledger = RightsLedger(bundle)
    for period, ob in ledger.obligations(bundle.policies["ladder-a"]).items():
        print(f"  {period}: {ob.counts()}，预留车位 {ob.reserved_slots:.0f}，"
              f"预留补差 {ob.reserved_credit:.0f} 元")

    print("\n" + "=" * 68)
    print("四、历史留出期对账（P00，基准价表）")
    print("=" * 68)
    rec = trial.reconcile(bundle.policies["base"], "P00", n_draws=500)
    for name, check in rec.metric_checks.items():
        flag = "覆盖" if check.covered else "偏差"
        print(f"  {name:8s} 预测 {check.predicted:7.2f} "
              f"区间[{check.low:6.2f}, {check.high:6.2f}] "
              f"实际 {check.actual:7.2f} → {flag}")

    print("\n" + "=" * 68)
    print("五、需要人工接住的事项")
    print("=" * 68)
    for item in report.manual_queue.items[:8]:
        print(f"  [{item['type']}] {item['message']}")
    print(f"  …共 {len(report.manual_queue.items)} 项")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else "data"))
