"""统一因子研究与账户回测的时点、收益和年化协议。"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from data import data_config


def protocol_config() -> dict[str, Any]:
    return data_config().get("protocol", {})


def signal_lag() -> int:
    return int(protocol_config().get("signal_lag", 1))


def oos_start() -> str:
    """样本外起点。该日及之后为样本外，更早为样本内。"""
    start = str(protocol_config().get("oos_start") or "").strip()
    if not start:
        raise ValueError("configs/data.yaml 的 protocol.oos_start 缺失，不能切分样本内外")
    return start


def annualization(frequency: str = "daily") -> int:
    key = "weekly_annualization" if frequency == "weekly" else "daily_annualization"
    return int(protocol_config().get(key, 52 if frequency == "weekly" else 243))


def lag_factor(
    frame: pd.DataFrame,
    factor_column: str,
    group_column: str = "symbol",
    lag: int | None = None,
) -> pd.Series:
    """把 T 收盘信号转成执行日可用信号，永不跨品种移位。"""
    lag = signal_lag() if lag is None else int(lag)
    if lag < 1:
        raise ValueError("signal lag 必须至少为 1")
    if group_column not in frame:
        return frame[factor_column].shift(lag)
    return frame.groupby(group_column, sort=False)[factor_column].shift(lag)


def book_mode() -> str:
    """研究连续价或账户原始结算价。缺省保持研究口径。"""
    book = str(protocol_config().get("book", "research")).strip().lower()
    if book not in {"research", "account"}:
        raise ValueError(f"protocol.book 只能是 research 或 account，收到 {book}")
    return book


def execution_price_column() -> str:
    """研究标签仍用后复权成交价，不随账户账本切换。"""
    cfg = protocol_config()
    return str(cfg.get("research_execution_price", cfg.get("execution_price", "adj_settle")))


def ledger_execution_column() -> str:
    """回测定手和成交使用的价格列。"""
    cfg = protocol_config()
    if book_mode() == "account":
        column = str(cfg.get("account_execution_price") or "").strip()
        if not column:
            raise ValueError("账户回测需要 protocol.account_execution_price")
        return column
    return execution_price_column()


def ledger_mark_column() -> str:
    """回测盯市价格列。账户账本与成交列同为原始结算价。"""
    cfg = protocol_config()
    if book_mode() == "account":
        column = str(cfg.get("account_mark_price") or "").strip()
        if not column:
            raise ValueError("账户回测需要 protocol.account_mark_price")
        return column
    return execution_price_column()


def forward_return(
    frame: pd.DataFrame,
    group_column: str = "symbol",
    holding_days: int = 1,
) -> pd.Series:
    """以 T 日 adj_close 信号为基准，计算 T+1 成交价至后续成交价收益。

    当前研究库每天每个品种只有一条合约，成交价使用后复权 ``adj_settle``。
    持有 H 天的收益是 adj_settle[T+1+H] / adj_settle[T+1] - 1。
    """
    if holding_days < 1:
        raise ValueError("holding_days 必须至少为 1")
    price_column = execution_price_column()
    if price_column not in frame.columns:
        raise ValueError(f"收益计算缺少成交价字段: {price_column}")
    grouped = frame.groupby(group_column, sort=False)[price_column]
    entry = grouped.shift(-signal_lag())
    exit_price = grouped.shift(-(signal_lag() + holding_days))
    return exit_price / entry - 1.0


def forward_open_return(
    frame: pd.DataFrame,
    group_column: str = "symbol",
    holding_days: int = 1,
) -> pd.Series:
    """兼容旧脚本的别名；实际按配置使用 T+1 结算价到后续结算价。"""
    return forward_return(
        frame, group_column=group_column, holding_days=holding_days
    )


def canonical_sample_split(split: dict[str, Any] | None = None) -> dict[str, Any]:
    """样本切分。起点只读配置：该日及之后为样本外，更早为样本内。"""
    payload = dict(split or {})
    canonical = oos_start()
    given = str(payload.get("oos_start") or "").strip()
    if given and pd.Timestamp(given) != pd.Timestamp(canonical):
        raise ValueError(f"样本外起点固定为 {canonical}，不能改成 {given}")
    payload["oos_start"] = canonical
    return payload


def sample_split(
    frame: pd.DataFrame,
    date_column: str = "date",
    boundary: str | None = None,
) -> pd.Series:
    """给每一行打 is / oos。省略边界时用协议起点。"""
    stamp = pd.Timestamp(boundary or oos_start())
    return pd.Series(
        np.where(pd.to_datetime(frame[date_column]) >= stamp, "oos", "is"),
        index=frame.index,
    )


def protocol_metadata(frequency: str = "daily") -> dict[str, Any]:
    cfg = protocol_config()
    book = book_mode()
    exec_col = ledger_execution_column()
    mark_col = ledger_mark_column()
    if book == "account":
        definition = (
            f"{exec_col} 成交并按名义金额定手，{mark_col} 盯市；"
            "换月日平旧合约盯市后按旧名义在新 settle 上重算手数，价差不进盈亏"
        )
    else:
        definition = (
            f"{execution_price_column()}[T+1+H] / "
            f"{execution_price_column()}[T+1] - 1"
        )
    return {
        "book": book,
        "signal_price": cfg.get("signal_price", "adj_close"),
        "execution_price": exec_col,
        "mark_price": mark_col,
        "signal_lag": signal_lag(),
        "oos_start": oos_start(),
        "annualization": annualization(frequency),
        "return_definition": definition,
    }
