"""按交易日读取合约乘数。

标量规格全样本有效。分段规格按 start 生效，日期早于第一段或数值非正时停止。
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from data import config, symbol_spec


def multiplier_on(symbol: str, day: str | pd.Timestamp) -> float:
    key = str(symbol).upper().removesuffix("_MAIN")
    products = (config().get("products") or {})
    product = products.get(key)
    if product and "multiplier" in product:
        return _schedule(product["multiplier"], pd.Timestamp(day), key)
    value = symbol_spec(symbol).get("multiplier")
    if isinstance(value, list):
        return _schedule(value, pd.Timestamp(day), key)
    multiplier = float(value)
    if multiplier <= 0:
        raise ValueError(f"{key} multiplier 必须为正")
    return multiplier


def _schedule(value: Any, day: pd.Timestamp, symbol: str) -> float:
    if not isinstance(value, list):
        multiplier = float(value)
    else:
        chosen = None
        for item in value:
            start = pd.Timestamp(item["start"])
            if day >= start:
                chosen = float(item["value"])
        if chosen is None:
            raise ValueError(f"{symbol} 在 {day.date()} 没有生效的 multiplier")
        multiplier = chosen
    if multiplier <= 0:
        raise ValueError(f"{symbol} multiplier 必须为正")
    return multiplier
