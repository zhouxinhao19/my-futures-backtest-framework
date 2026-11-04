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


def _code_blank(value: object) -> bool:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return True
    text = str(value).strip().lower()
    return text in {"", "nan", "none"}


def apply_roll_lot_resize_for_day(
    lots: np.ndarray,
    day: int,
    codes: np.ndarray,
    values: np.ndarray,
    mult: np.ndarray,
    *,
    lag: int,
    holding: int,
    cohort_holding: np.ndarray | None,
) -> None:
    """在换月日 ``day`` 结束后，把仍持有中的各批手数按旧名义在新 settle 上重算。"""
    if day <= 0 or day >= codes.shape[0]:
        return
    span_cap = holding if cohort_holding is None else int(np.max(cohort_holding))
    start = max(0, day - lag - span_cap)
    for col in range(codes.shape[1]):
        previous_code = codes[day - 1, col]
        current_code = codes[day, col]
        if _code_blank(previous_code) or _code_blank(current_code):
            continue
        if str(previous_code) == str(current_code):
            continue
        prev_px = values[day - 1, col]
        new_px = values[day, col]
        if not (np.isfinite(prev_px) and np.isfinite(new_px) and new_px != 0):
            continue
        unit_old = float(prev_px) * float(mult[col])
        unit_new = float(new_px) * float(mult[col])
        for signal in range(start, day + 1):
            span = holding if cohort_holding is None else int(cohort_holding[signal])
            if span < 1:
                continue
            age = day - signal
            if not (lag + 1 <= age <= lag + span):
                continue
            old = lots[signal, col]
            if old == 0.0:
                continue
            notional = abs(old) * unit_old
            lots[signal, col] = np.sign(old) * np.floor(notional / unit_new)


def apply_roll_lot_resize(
    lots: np.ndarray,
    codes: np.ndarray,
    values: np.ndarray,
    mult: np.ndarray,
    *,
    lag: int,
    holding: int,
    cohort_holding: np.ndarray | None,
) -> None:
    """账户换月：逐换月日调用 ``apply_roll_lot_resize_for_day``（原地修改 lots）。"""
    if codes is None or len(lots) < 2:
        return
    for day in range(1, codes.shape[0]):
        apply_roll_lot_resize_for_day(
            lots, day, codes, values, mult,
            lag=lag, holding=holding, cohort_holding=cohort_holding,
        )


def apply_roll_position_resize_for_day(
    position: np.ndarray,
    day: int,
    codes: np.ndarray,
    values: np.ndarray,
    mult: np.ndarray,
) -> None:
    """差额调仓路径：换月日按旧名义重算净手数向量。"""
    if day <= 0:
        return
    for col in range(codes.shape[1]):
        previous_code = codes[day - 1, col]
        current_code = codes[day, col]
        if _code_blank(previous_code) or _code_blank(current_code):
            continue
        if str(previous_code) == str(current_code):
            continue
        lots = float(position[col])
        if lots == 0.0:
            continue
        prev_px = values[day - 1, col]
        new_px = values[day, col]
        if not (np.isfinite(prev_px) and np.isfinite(new_px) and new_px != 0):
            continue
        unit_old = float(prev_px) * float(mult[col])
        unit_new = float(new_px) * float(mult[col])
        notional = abs(lots) * unit_old
        position[col] = np.sign(lots) * np.floor(notional / unit_new)
