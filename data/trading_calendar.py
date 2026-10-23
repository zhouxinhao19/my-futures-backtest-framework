from __future__ import annotations

import pandas as pd


def sessions(frame: pd.DataFrame) -> pd.DatetimeIndex:
    """从已验证行情提取该品种真实交易日，而不是用普通工作日推断。"""
    if "date" not in frame:
        raise ValueError("行情缺少 date")
    return pd.DatetimeIndex(sorted(pd.to_datetime(frame["date"]).dropna().unique()))


def last_session_of_week(frame: pd.DataFrame) -> pd.DatetimeIndex:
    dates = sessions(frame)
    if dates.empty:
        return dates
    return pd.DatetimeIndex(pd.Series(dates, index=dates).groupby(
        dates.to_period("W-FRI")
    ).max().tolist())


def next_session(sessions_: pd.DatetimeIndex, date: pd.Timestamp) -> pd.Timestamp | None:
    future = sessions_[sessions_ > pd.Timestamp(date)]
    return future[0] if len(future) else None
