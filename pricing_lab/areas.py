"""区域口径：价表提案范围与时段分析区域的匹配。

价表范围可以是单个区域（如「港城」）或组合范围（如「海湾和港城」），
时段区域按是否被范围字符串包含判断。
"""
from __future__ import annotations


def area_in_scope(period_area: str, policy_area: str) -> bool:
    return period_area in policy_area
