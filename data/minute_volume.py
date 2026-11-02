"""把 5 分钟行情里 2020 年前的双边成交量折成单边。

日频 Parquet 入库时已经是单边，不能再除。5 分钟文件保留交易所公布数，
只在读取后调整，不回写文件。
"""
from __future__ import annotations

from typing import Any

import pandas as pd


def minute_volume_convention() -> dict[str, Any]:
    """读取 ``configs/data.yaml`` 的单双边口径。缺配置时停止。"""
    from data import data_config

    raw = data_config().get("volume_convention")
    if not isinstance(raw, dict):
        raise RuntimeError("configs/data.yaml 缺少 volume_convention")
    since = raw.get("single_sided_since")
    fields = raw.get("fields")
    double_sided = raw.get("minute_double_sided_exchanges")
    always_single = raw.get("minute_always_single_exchanges")
    if not since or not fields or not double_sided or not always_single:
        raise RuntimeError(
            "volume_convention 需要 single_sided_since、fields、"
            "minute_double_sided_exchanges 和 minute_always_single_exchanges"
        )
    return {
        "since": pd.Timestamp(since).normalize(),
        "fields": tuple(str(name) for name in fields),
        "double_sided": {str(name).upper() for name in double_sided},
        "always_single": {str(name).upper() for name in always_single},
    }


def minute_volume_convention_record() -> dict[str, Any]:
    """写入 run.json 的口径说明。"""
    rule = minute_volume_convention()
    return {
        "storage": "raw_exchange_print",
        "read_adjustment": "divide_by_2",
        "single_sided_since": rule["since"].date().isoformat(),
        "fields": list(rule["fields"]),
        "exchanges": sorted(rule["double_sided"]),
        "daily_panel": "already_single_sided",
    }


def apply_minute_single_sided(frame: pd.DataFrame) -> pd.DataFrame:
    """2020-01-01 之前，指定交易所的成交量、成交额、持仓除以 2。

    价格列不动。中金所等始终单边的交易所不除。未登记交易所在切换日之前出现时报错。
    """
    rule = minute_volume_convention()
    fields = [name for name in rule["fields"] if name in frame.columns]
    if frame.empty or not fields:
        return frame
    if "trade_date" not in frame.columns:
        raise ValueError("5 分钟行情缺少 trade_date，不能折算单边成交量")
    out = frame.copy()
    trade_date = pd.to_datetime(out["trade_date"], errors="coerce").dt.normalize()
    if trade_date.isna().any():
        raise ValueError("5 分钟 trade_date 无法解析，不能折算单边成交量")
    before = trade_date.lt(rule["since"])
    if not bool(before.any()):
        return out
    if "exchange" not in out.columns:
        raise ValueError("5 分钟行情缺少 exchange，不能判断 2020 年前是否为双边口径")
    exchange = out["exchange"].astype("string").str.strip().str.upper()
    missing_exchange = before & exchange.isna()
    if bool(missing_exchange.any()):
        raise ValueError("5 分钟在单边切换日之前有空的 exchange，不能折算成交量")
    known = rule["double_sided"] | rule["always_single"]
    unknown = before & ~exchange.isin(known)
    if bool(unknown.any()):
        names = sorted(set(exchange.loc[unknown].dropna()))
        raise ValueError(f"5 分钟存在未登记单双边口径的交易所: {names}")
    adjust = before & exchange.isin(rule["double_sided"])
    if bool(adjust.any()):
        for name in fields:
            values = pd.to_numeric(out[name], errors="coerce").astype("float64")
            out[name] = values
            out.loc[adjust, name] = values.loc[adjust] / 2.0
    return out
