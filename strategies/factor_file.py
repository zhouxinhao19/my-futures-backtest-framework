"""外部因子宽表的时序趋势掩码。

文件第一列是 ``TRADE_DATE``，其余每列一个品种。大于 0 做多，小于 0 做空，
空白和 0 当天该品种不交易。周频只保留每个自然周最后一个交易日的掩码，
持有到下一次调仓。这里只产掩码，成交滞后和盈亏由回测层处理。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from data import pool_symbols
from factors.signals.external_factor import load_factor_wide, monthly_schedule, weekly_schedule
from backtest.masks import long_short_masks

DATE_COLUMN = "TRADE_DATE"
MODE = "time_series_trend"


def factor_file_symbols(path: str | Path) -> list[str]:
    """读取宽表表头里的品种，保持文件中的列顺序。"""
    columns = [str(column).strip() for column in pd.read_csv(path, nrows=0).columns]
    if not columns or columns[0] != DATE_COLUMN:
        raise ValueError(f"因子文件第一列必须是 {DATE_COLUMN}: {path}")
    symbols = [column.upper() for column in columns[1:]]
    if not symbols:
        raise ValueError(f"因子文件没有品种列: {path}")
    duplicated = sorted({symbol for symbol in symbols if symbols.count(symbol) > 1})
    if duplicated:
        raise ValueError(f"因子文件品种列重复: {duplicated}")
    return symbols


def require_pool_symbols(symbols: list[str], universe: str) -> list[str]:
    """品种必须与配置品种池完全一致，并按品种池顺序返回。"""
    pool = pool_symbols(universe)
    extra = sorted(set(symbols) - set(pool))
    missing = sorted(set(pool) - set(symbols))
    if extra or missing:
        raise ValueError(
            f"因子文件品种与品种池 {universe} 不一致: 多出 {extra}，缺少 {missing}"
        )
    return list(pool)


def time_series_trend_masks(path: str | Path, symbols: list[str]) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """空白保持缺失。0 和缺失都不进多空掩码。"""
    values, quality = load_factor_wide(path, symbols)
    wide = (
        values.pivot(index="date", columns="symbol", values="expression")
        .reindex(columns=symbols)
        .sort_index()
    )
    long_mask, short_mask = long_short_masks(wide, MODE)
    active = long_mask.to_numpy() | short_mask.to_numpy()
    quality = dict(quality)
    quality["long_cells"] = int(long_mask.to_numpy().sum())
    quality["short_cells"] = int(short_mask.to_numpy().sum())
    quality["flat_or_blank_cells"] = int((~active).sum())
    quality["long_only_days"] = int((long_mask.any(axis=1) & ~short_mask.any(axis=1)).sum())
    quality["short_only_days"] = int((short_mask.any(axis=1) & ~long_mask.any(axis=1)).sum())
    return long_mask, short_mask, quality


def apply_weekly_rebalance(
    long_mask: pd.DataFrame,
    short_mask: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    """每个自然周最后一个交易日出信号，持有到下一次信号。

    周五休市时用该周最后一个交易日，不用“必须是周五”。没有下一次成交日的
    最后一周不交易。返回值第三项是每个信号日要持有的交易日数，其余日期为 0。
    """
    schedule = weekly_schedule(long_mask.index)
    keep = pd.Series(False, index=long_mask.index)
    holding = pd.Series(0, index=long_mask.index, dtype=int)
    if not schedule.empty:
        signal_dates = pd.DatetimeIndex(schedule["signal_date"])
        keep.loc[signal_dates] = True
        holding.loc[signal_dates] = schedule["holding_days"].to_numpy()
    kept_long = long_mask.copy()
    kept_short = short_mask.copy()
    kept_long.loc[~keep, :] = False
    kept_short.loc[~keep, :] = False
    return kept_long, kept_short, holding


def apply_every_n_rebalance(
    long_mask: pd.DataFrame,
    short_mask: pd.DataFrame,
    hold_days: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    """每 hold_days 个交易日出一次信号，持有到下一次信号。

    信号日是交易日历上的第 0、N、2N… 日。没有下一次信号的最后一段不新开仓。
    返回的持有天数是相邻信号日之间的交易日数，供差额调仓使用。
    """
    if int(hold_days) < 1:
        raise ValueError("hold_days 必须至少为 1")
    step = int(hold_days)
    dates = pd.DatetimeIndex(long_mask.index)
    positions = list(range(0, len(dates), step))
    keep = pd.Series(False, index=dates)
    holding = pd.Series(0, index=dates, dtype=int)
    for index, pos in enumerate(positions):
        if index + 1 >= len(positions) or pos + 1 >= len(dates):
            break
        keep.iloc[pos] = True
        holding.iloc[pos] = int(positions[index + 1] - pos)
    kept_long = long_mask.copy()
    kept_short = short_mask.copy()
    kept_long.loc[~keep, :] = False
    kept_short.loc[~keep, :] = False
    return kept_long, kept_short, holding


def apply_monthly_rebalance(
    long_mask: pd.DataFrame,
    short_mask: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    """每个自然月最后一个交易日出信号，持有到下一次信号。"""
    schedule = monthly_schedule(long_mask.index)
    keep = pd.Series(False, index=long_mask.index)
    holding = pd.Series(0, index=long_mask.index, dtype=int)
    if not schedule.empty:
        signal_dates = pd.DatetimeIndex(schedule["signal_date"])
        keep.loc[signal_dates] = True
        holding.loc[signal_dates] = schedule["holding_days"].to_numpy()
    kept_long = long_mask.copy()
    kept_short = short_mask.copy()
    kept_long.loc[~keep, :] = False
    kept_short.loc[~keep, :] = False
    return kept_long, kept_short, holding
