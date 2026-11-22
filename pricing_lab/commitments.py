"""渠道实收报价承诺登记与人工队列。

口径：
- 渠道实收报价一旦确认已发，就按当时条件核兑并冻结为承诺；之后渠道补回的曝光、
  履约数据可以重新比较新方案（可采用范围可升可降），但不得覆盖已发承诺；
- 承诺只增不改；同一旅程确认新承诺需带显式变更依据，否则进人工队列；
- 资料异常（实收与所见价不符、购买未见套餐、收齐后缺记录等）全部进人工队列。
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path

from .models import DataIssue


@dataclass(frozen=True)
class CommittedQuote:
    journey: str
    channel: str
    plan: str
    price: float
    confirmed_at: datetime
    source_exposure: str
    basis: str
    superseded: bool = False
    change_note: str = ""


class CommitmentRegistry:
    def __init__(self, path: str | Path | None = None):
        self._path = Path(path) if path else None
        self._items: list[CommittedQuote] = []
        if self._path and self._path.exists():
            for row in json.loads(self._path.read_text(encoding="utf-8")):
                self._items.append(
                    CommittedQuote(
                        journey=row["journey"], channel=row["channel"],
                        plan=row["plan"], price=row["price"],
                        confirmed_at=datetime.fromisoformat(row["confirmed_at"]),
                        source_exposure=row["source_exposure"],
                        basis=row["basis"], superseded=row.get("superseded", False),
                        change_note=row.get("change_note", ""),
                    )
                )

    def confirm(self, quote: CommittedQuote) -> CommittedQuote:
        """登记一条已确认承诺；与既有承诺冲突时拒绝自动覆盖。"""
        active = [
            q for q in self._items
            if q.journey == quote.journey and not q.superseded
        ]
        for prior in active:
            if (prior.plan, prior.price) != (quote.plan, quote.price):
                raise ValueError(
                    f"旅程 {quote.journey} 已有生效承诺 {prior.plan}@"
                    f"{prior.price:.0f}（{prior.confirmed_at.isoformat()}），"
                    "新数据不得自动覆盖，请走人工变更"
                )
        if not active:
            self._items.append(quote)
        self._flush()
        return quote

    def supersede(self, journey: str, note: str) -> None:
        """显式人工变更：旧承诺标记失效并留痕，不做物理删除。"""
        for q in self._items:
            if q.journey == journey and not q.superseded:
                object.__setattr__(q, "superseded", True)
                object.__setattr__(q, "change_note", note)
        self._flush()

    def active_for(self, journey: str) -> CommittedQuote | None:
        for q in reversed(self._items):
            if q.journey == journey and not q.superseded:
                return q
        return None

    def all_items(self) -> tuple[CommittedQuote, ...]:
        return tuple(self._items)

    def _flush(self) -> None:
        if self._path is None:
            return
        payload = []
        for q in self._items:
            row = asdict(q)
            row["confirmed_at"] = q.confirmed_at.isoformat()
            payload.append(row)
        self._path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )


@dataclass
class ManualQueue:
    """需要人工接住的权益与异常，按类型留痕、可回查。"""

    items: list[dict] = field(default_factory=list)

    def add_issue(self, issue: DataIssue) -> None:
        self.items.append(
            {
                "type": issue.code,
                "message": issue.message,
                "refs": issue.refs,
            }
        )

    def add(self, type_: str, message: str, refs: dict | None = None) -> None:
        self.items.append({"type": type_, "message": message, "refs": refs or {}})

    def by_type(self, type_: str) -> list[dict]:
        return [x for x in self.items if x["type"] == type_]

    def save(self, path: str | Path) -> None:
        Path(path).write_text(
            json.dumps(self.items, ensure_ascii=False, indent=2), encoding="utf-8"
        )
