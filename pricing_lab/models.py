"""领域对象与轻量校验。

记录保留原始字段的冲突与未决关系（字典说明允许），
校验只做结构与外键，不强行清洗业务事实。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, date

PLANS = ("semi", "day", "express")
PLAN_LABELS = {"semi": "半日", "day": "整日", "express": "直通景点"}


def parse_dt(value: str) -> datetime:
    return datetime.fromisoformat(value)


def parse_day(value: str) -> date:
    return date.fromisoformat(value)


@dataclass(frozen=True)
class Period:
    period: str
    day: date
    area: str
    window: str
    arm: str
    random: bool
    fleet_hours: float
    closed_at: date


@dataclass(frozen=True)
class Journey:
    journey: str
    period: str
    requested_at: datetime
    length_hours: float
    party: int
    needs: str


@dataclass(frozen=True)
class Exposure:
    exposure: str
    journey: str
    channel: str
    seen_at: datetime
    options: tuple[str, ...]
    prices: tuple[float, ...]
    old_choice: bool

    @property
    def price_map(self) -> dict[str, float]:
        return dict(zip(self.options, self.prices))


@dataclass(frozen=True)
class Fulfilment:
    journey: str
    plan: str
    paid: float
    use: str
    paid_at: datetime
    received_day: date


@dataclass(frozen=True)
class CatalogItem:
    plan: str
    hours: float
    transfer: float
    operator_coupon: float
    hour_cost: float

    @property
    def vehicle_cost(self) -> float:
        """车时成本 = 车时 × 车时单价。"""
        return self.hours * self.hour_cost


@dataclass(frozen=True)
class QuoteRight:
    quote: str
    period: str
    origin: str
    price: float
    alternatives: tuple[str, ...]
    until: datetime
    state: str
    change_at: datetime


@dataclass(frozen=True)
class Policy:
    version: str
    area: str
    plans: dict[str, float]
    credit_ceiling: float
    support_level: float


@dataclass(frozen=True)
class DataIssue:
    """资料层记录的可疑事实，进入人工队列，不自动改写数据。"""

    code: str
    message: str
    refs: dict = field(default_factory=dict)
