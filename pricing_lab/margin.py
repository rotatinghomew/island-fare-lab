"""毛利与成本口径。

单位毛利（元）= 实收报价 − 车时成本 − 接驳采购 − 运营方承担优惠。
车时成本随套餐替换而变（车时 × 车时单价），接驳采购与优惠承担额按套餐目录取值。
"""
from __future__ import annotations

from dataclasses import dataclass

from .io import DataBundle
from .models import CatalogItem, Policy


@dataclass(frozen=True)
class MarginComponents:
    plan: str
    revenue: float
    vehicle_cost: float
    transfer: float
    operator_coupon: float
    unit_margin: float


def plan_unit_cost(bundle: DataBundle, plan: str) -> tuple[float, float, float]:
    """返回（车时成本, 接驳采购, 运营方承担优惠）。"""
    item: CatalogItem = bundle.catalog[plan]
    return item.vehicle_cost, item.transfer, item.operator_coupon


def policy_price(policy: Policy, plan: str) -> float:
    return float(policy.plans[plan])


def gross_margin_components(
    bundle: DataBundle, plan: str, price: float
) -> MarginComponents:
    vehicle, transfer, coupon = plan_unit_cost(bundle, plan)
    return MarginComponents(
        plan=plan,
        revenue=price,
        vehicle_cost=vehicle,
        transfer=transfer,
        operator_coupon=coupon,
        unit_margin=price - vehicle - transfer - coupon,
    )


def realized_unit_margin(bundle: DataBundle, plan: str, paid: float) -> float:
    """历史实际单位毛利，按实收金额计。"""
    return gross_margin_components(bundle, plan, paid).unit_margin
