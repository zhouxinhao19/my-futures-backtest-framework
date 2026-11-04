from __future__ import annotations

import pandas as pd


def gate_signal_days(
    long: pd.DataFrame,
    short: pd.DataFrame,
    rule: str = "month_end",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """只保留调仓日上的多空掩码。月末用日历月末。"""
    due = due_on(pd.DatetimeIndex(long.index), hold_days=1, rule=rule)
    keep = due.reindex(long.index).fillna(False).to_numpy()
    return long.mul(keep, axis=0), short.mul(keep, axis=0)


def due_on(dates: pd.DatetimeIndex, hold_days: int, rule: str = "hold") -> pd.Series:
    """持有 H 天：开仓后第 H 个交易日到期。周末和月末只标记日历规则。"""
    index = pd.DatetimeIndex(dates)
    if rule == "hold":
        if hold_days < 1:
            raise ValueError("hold_days 必须至少为 1")
        age = pd.Series(range(len(index)), index=index)
        return age.mod(hold_days).eq(0)
    if rule == "month_end":
        return pd.Series(index.is_month_end, index=index)
    if rule == "week_end":
        return pd.Series(index.weekday == 4, index=index)
    raise ValueError(f"未知调仓规则: {rule}")
