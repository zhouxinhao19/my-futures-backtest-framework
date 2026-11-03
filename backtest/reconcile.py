"""冻结手数复算。只用于和旧研究日表对照。"""

from __future__ import annotations

import numpy as np
import pandas as pd


def infer_old_settle(
    previous_old_settle: float,
    current_adj_settle: float,
    previous_adj_settle: float,
) -> float:
    values = (previous_old_settle, current_adj_settle, previous_adj_settle)
    if any(pd.isna(value) for value in values) or previous_adj_settle == 0:
        raise ValueError("换月反推缺少旧结算价或复权结算价")
    return float(previous_old_settle) * float(current_adj_settle) / float(previous_adj_settle)


def replay_adj_pnl(
    lots: pd.DataFrame,
    price: pd.DataFrame,
    specs: dict[str, dict],
    holding: int,
    *,
    lag: int = 1,
    cost_bps: float = 0.0,
    capital: float = 100_000_000.0,
    price_change: pd.DataFrame | None = None,
) -> pd.DataFrame:
    products = list(price.columns)
    dates = price.index
    n = len(dates)
    held = lots.reindex(index=dates, columns=products).fillna(0.0).to_numpy(float)
    mult = np.array([float(specs[p]["multiplier"]) for p in products])
    values = price.to_numpy(float)
    if price_change is None:
        change = np.full_like(values, np.nan)
        change[1:] = values[1:] - values[:-1]
    else:
        change = price_change.reindex(index=dates, columns=products).to_numpy(float)
    entry = np.full_like(values, np.nan)
    entry[: n - lag] = values[lag:]
    unit = np.where(np.isfinite(entry), entry * mult, np.nan)
    notional_t = np.nansum(np.abs(held) * np.nan_to_num(unit), axis=1)
    exit_price = np.full_like(values, np.nan)
    exit_price[: n - lag - holding] = values[lag + holding :]
    exit_notional = np.nansum(np.abs(held) * np.nan_to_num(exit_price) * mult, axis=1)
    pnl = np.zeros(n)
    cost = np.zeros(n)
    rate = cost_bps / 10_000.0
    for offset in range(lag + 1, lag + holding + 1):
        if offset >= n:
            continue
        shifted = np.full_like(change, np.nan)
        shifted[: n - offset] = change[offset:]
        contrib = np.where(np.isfinite(held * shifted), held * mult * shifted, 0.0)
        target = np.arange(n - offset) + offset
        pnl[target] += contrib[: n - offset].sum(axis=1)
    entry_h = lag + 1
    cost[entry_h:] += notional_t[: n - entry_h] * rate
    exit_h = lag + holding + 1
    if exit_h < n:
        cost[exit_h:] += exit_notional[: n - exit_h] * rate
    return pd.DataFrame({"pnl": pnl, "cost": cost, "net": pnl - cost}, index=dates)


def raw_change_without_roll_gap(
    settle: pd.DataFrame,
    adj_settle: pd.DataFrame,
    code: pd.DataFrame,
) -> pd.DataFrame:
    change = settle.diff()
    for symbol in settle.columns:
        codes = code[symbol].astype(str)
        rolled = codes.ne(codes.shift(1)) & codes.shift(1).notna()
        for day in settle.index[rolled.fillna(False)]:
            loc = settle.index.get_loc(day)
            if isinstance(loc, slice) or loc == 0:
                continue
            previous = settle.index[loc - 1]
            try:
                old = infer_old_settle(
                    float(settle.at[previous, symbol]),
                    float(adj_settle.at[day, symbol]),
                    float(adj_settle.at[previous, symbol]),
                )
            except ValueError:
                # 换月前后 settle/adj_settle 不完整时无法剔除价差，当日该品种盯市变化置 0。
                change.at[day, symbol] = 0.0
                continue
            change.at[day, symbol] = old - float(settle.at[previous, symbol])
    return change
