"""命令行入口：试算、支持域、权利台账、对账与人工队列。

示例：
  python -m pricing_lab.cli trial --mode 试算
  python -m pricing_lab.cli support --policy ladder-a
  python -m pricing_lab.cli rights --policy ladder-a
  python -m pricing_lab.cli reconcile --policy base --period P00
  python -m pricing_lab.cli queue
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

from .feasibility import TrialMode
from .io import load_bundle, snapshot
from .rights import RightsLedger
from .trial import PricingTrial

DEFAULT_DATA = Path(__file__).resolve().parents[1] / "data"
DEFAULT_OUT = Path(__file__).resolve().parents[1] / "runs"


def _print(obj) -> None:
    print(json.dumps(obj, ensure_ascii=False, indent=2, default=str))


def _load(args):
    bundle = load_bundle(args.data)
    if args.as_of:
        bundle = snapshot(bundle, datetime.fromisoformat(args.as_of))
    return bundle


def cmd_trial(args) -> None:
    bundle = _load(args)
    trial = PricingTrial(bundle, seed=args.seed)
    trial.fit(n_bootstrap=args.bootstrap, cl_level=args.level)
    report = trial.run(mode=TrialMode(args.mode), n_draws=args.draws)
    payload = report.as_dict()
    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = out_dir / f"trial_{stamp}.json"
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    report.manual_queue.save(out_dir / f"manual_queue_{stamp}.json")
    print(f"试算报告已写入 {path}")
    print(f"证据分层：{payload['evidence_classes']}")
    print(f"留出表现：log_loss={payload['holdout']['mean_log_loss']}, "
          f"准确率={payload['holdout']['choice_accuracy']}")
    for version, result in payload["policies"].items():
        print(
            f"- {version}: 支持={result['support']['overall']}，"
            f"可定案时段={len(result['finalizable_periods'])}，"
            f"受阻时段={len(result['blocked_periods'])}，"
            f"毛利差区间={result['margin']['delta_margin_interval']}"
        )


def cmd_support(args) -> None:
    bundle = _load(args)
    policy = bundle.policies[args.policy]
    from .support import support_matrix
    _print(support_matrix(bundle, policy).as_dict())


def cmd_rights(args) -> None:
    bundle = _load(args)
    as_of = datetime.fromisoformat(args.as_of) if args.as_of else None
    ledger = RightsLedger(bundle, as_of=as_of)
    policy = bundle.policies.get(args.policy) if args.policy else None
    payload = {}
    for period, ob in ledger.obligations(policy).items():
        payload[period] = {
            "counts": ob.counts(),
            "reserved_slots": ob.reserved_slots,
            "reserved_credit": round(ob.reserved_credit, 2),
            "active": [
                {
                    "quote": v.quote,
                    "state": v.state.value,
                    "alternatives": list(v.alternatives),
                    "reserved_slots": v.reserved_slots,
                    "credit_by_plan": {k: round(x, 2) for k, x in v.credit_by_plan.items()},
                    "reserved_credit": round(v.reserved_credit, 2),
                    "note": v.note,
                }
                for v in ob.active_views
            ],
            "exercised": [v.quote for v in ob.exercised_views],
            "released": [
                {"quote": v.quote, "state": v.state.value, "note": v.note}
                for v in ob.released_views
            ],
        }
    _print(payload)


def cmd_reconcile(args) -> None:
    bundle = _load(args)
    trial = PricingTrial(bundle, seed=args.seed)
    trial.fit(n_bootstrap=args.bootstrap, cl_level=args.level)
    report = trial.reconcile(
        bundle.policies[args.policy], args.period, n_draws=args.draws
    )
    _print(report.as_dict())


def cmd_queue(args) -> None:
    bundle = _load(args)
    _print([
        {"code": i.code, "message": i.message, "refs": i.refs}
        for i in bundle.issues
    ])


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="pricing_lab", description="沿海包车定价试算工具")
    parser.add_argument("--data", default=str(DEFAULT_DATA), help="资料目录")
    parser.add_argument("--as-of", help="快照时点（ISO 时间），默认使用全部已到资料")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("trial", help="运行完整试算并输出报告")
    p.add_argument("--mode", choices=[m.value for m in TrialMode], default="试算")
    p.add_argument("--bootstrap", type=int, default=1000)
    p.add_argument("--draws", type=int, default=1000)
    p.add_argument("--level", type=float, default=0.9)
    p.add_argument("--seed", type=int, default=20260601)
    p.add_argument("--out", default=str(DEFAULT_OUT))
    p.set_defaults(func=cmd_trial)

    p = sub.add_parser("support", help="查看某价表的共同支持诊断")
    p.add_argument("--policy", required=True)
    p.set_defaults(func=cmd_support)

    p = sub.add_parser("rights", help="查看旧价权利台账与预留义务")
    p.add_argument("--policy", help="可选：按该价表计算补差预算")
    p.set_defaults(func=cmd_rights)

    p = sub.add_parser("reconcile", help="时段试算与实际对账")
    p.add_argument("--policy", required=True)
    p.add_argument("--period", required=True)
    p.add_argument("--bootstrap", type=int, default=1000)
    p.add_argument("--draws", type=int, default=1000)
    p.add_argument("--level", type=float, default=0.9)
    p.add_argument("--seed", type=int, default=20260601)
    p.set_defaults(func=cmd_reconcile)

    p = sub.add_parser("queue", help="列出需要人工接住的异常与权益")
    p.set_defaults(func=cmd_queue)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    args.func(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
