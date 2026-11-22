"""沿海包车定价试算工具。

模块划分与业务口径对应关系见 README「方法与口径」。
所有金额单位为人民币元，车时单位为小时，时间均为北京时间。
"""
from .io import DataBundle, load_bundle, snapshot
from .models import (
    CatalogItem,
    Exposure,
    Fulfilment,
    Journey,
    Period,
    Policy,
    QuoteRight,
)
from .evidence import classify_journey, EvidenceClass
from .estimator import DemandModel, DemandEstimate, fit_demand
from .support import support_matrix, SupportVerdict
from .margin import plan_unit_cost, policy_price, gross_margin_components
from .rights import RightsLedger, RightState
from .feasibility import JointFeasibility, evaluate_feasibility, TrialMode
from .commitments import CommitmentRegistry, ManualQueue
from .reconcile import reconcile_period, ReconcileReport
from .trial import PricingTrial

__all__ = [
    "DataBundle",
    "load_bundle",
    "snapshot",
    "CatalogItem",
    "Exposure",
    "Fulfilment",
    "Journey",
    "Period",
    "Policy",
    "QuoteRight",
    "classify_journey",
    "EvidenceClass",
    "DemandModel",
    "DemandEstimate",
    "fit_demand",
    "support_matrix",
    "SupportVerdict",
    "plan_unit_cost",
    "policy_price",
    "gross_margin_components",
    "RightsLedger",
    "RightState",
    "JointFeasibility",
    "evaluate_feasibility",
    "TrialMode",
    "CommitmentRegistry",
    "ManualQueue",
    "reconcile_period",
    "ReconcileReport",
    "PricingTrial",
]
