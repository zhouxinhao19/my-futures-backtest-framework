"""研究回测记账。

冻结权益时与原来的 ``continuous_backtest`` 相同。正式回测从初始权益开始，
当日净盈亏计入权益，下一批开仓按当时权益定手。
"""

from __future__ import annotations

from dataclasses import dataclass
import time

import numpy as np
import pandas as pd
from numba import njit

from data import symbol_spec

CAPITAL = 100_000_000.0
LEG_NOTIONAL = CAPITAL / 2
COST_BPS = 5.0


@dataclass
class ResearchResult:
    daily: pd.DataFrame
    trades: pd.DataFrame
    positions: pd.DataFrame


def tradable_specs(products: list[str]) -> tuple[dict[str, dict], list[str]]:
    """返回有乘数的品种。保证金只记录，缺失时不补默认值，也不因此剔除品种。"""
    ok: dict[str, dict] = {}
    missing: list[str] = []
    for product in products:
        try:
            spec = symbol_spec(product)
        except KeyError:
            missing.append(product)
            continue
        if spec.get("multiplier") is None:
            missing.append(product)
            continue
        margin = spec.get("margin_rate")
        ok[product] = {
            "multiplier": float(spec["multiplier"]),
            "margin_rate": float(margin) if margin is not None else float("nan"),
            "quality": spec.get("quality"),
        }
    return ok, missing


def signal_lots(
    long_mask: pd.DataFrame,
    short_mask: pd.DataFrame,
    price: pd.DataFrame,
    specs: dict[str, dict],
    holding: int,
    *,
    lag: int = 1,
    leg_notional: float = LEG_NOTIONAL,
) -> pd.DataFrame:
    """返回信号日的带符号手数。多空腿不完整的信号日整行记 0。"""
    frame, _zero = _size_lots(
        long_mask, short_mask, price, specs, holding, lag=lag,
        leg_notional=np.full(len(price.index), float(leg_notional)),
    )
    return frame


def continuous_backtest(
    long_mask: pd.DataFrame,
    short_mask: pd.DataFrame,
    price: pd.DataFrame,
    specs: dict[str, dict],
    holding: int,
    *,
    lag: int = 1,
    cost_bps: float = COST_BPS,
    leg_notional: float = LEG_NOTIONAL,
    capital: float = CAPITAL,
) -> pd.DataFrame:
    """冻结权益的研究日表。收益率用净盈亏除以初始权益。"""
    result = research_backtest(
        long_mask, short_mask, price, specs, holding,
        lag=lag, cost_bps=cost_bps, capital=capital, compound=False,
        leg_notional=leg_notional,
    )
    return result.daily.drop(columns=["equity_before", "equity_after"])


def research_backtest(
    long_mask: pd.DataFrame,
    short_mask: pd.DataFrame,
    price: pd.DataFrame,
    specs: dict[str, dict],
    holding: int,
    *,
    lag: int = 1,
    cost_bps: float = COST_BPS,
    capital: float = CAPITAL,
    compound: bool = True,
    leg_notional: float | None = None,
    require_both_legs: bool = True,
    cohort_holding: np.ndarray | None = None,
    allocation: str = "leg",
    execution: str = "cohort",
    contracts: pd.DataFrame | None = None,
    long_weight: pd.DataFrame | None = None,
    short_weight: pd.DataFrame | None = None,
    timings: dict[str, float] | None = None,
    price_change: pd.DataFrame | None = None,
) -> ResearchResult:
    """正式研究回测。``compound=False`` 时每批仍用初始权益的一半。

    ``cohort_holding`` 给出每个信号日各自的持有交易日数，0 表示当天不开仓。
    周频调仓用它，因为节假日周长短不同，而且同一时间只有一批仓位，
    定手不再按持有天数把资金拆开。
    ``execution="cohort"`` 时每一批单独开仓、到期整批平仓。
    ``execution="adjust"`` 时把同时持有的批次轧成净手数，只对净手数变化收取成本。
    ``contracts`` 给出每天的主力合约代码。账户换月日盯市仍只反映旧合约涨跌，
    换月后按旧名义在新 settle 上重算手数，并按平旧、开新各计一次成本。
    """
    if allocation not in {"leg", "equal_active"}:
        raise ValueError("allocation 只能是 leg 或 equal_active")
    if execution not in {"cohort", "adjust"}:
        raise ValueError("execution 只能是 cohort 或 adjust")
    if lag < 1 or (cohort_holding is None and holding < 1):
        raise ValueError("holding、lag 必须为正整数")
    if cost_bps < 0:
        raise ValueError("cost_bps 不能为负")
    products = list(price.columns)
    if set(products) - set(specs):
        raise ValueError("price 中存在没有交易规格的品种")
    dates = price.index
    n = len(dates)
    values = price.to_numpy(float)
    mult = np.array([float(specs[p]["multiplier"]) for p in products])
    margin_rate = np.array([float(specs[p]["margin_rate"]) for p in products])
    change = _price_change(values, price_change, dates, products)
    spans = _cohort_spans(cohort_holding, n)
    size_holding = 1 if spans is not None else holding
    codes = None if contracts is None else contracts.reindex(index=dates, columns=products).to_numpy(dtype=object)
    sizing_started = time.perf_counter()
    if compound:
        lots = _compound_lots(
            long_mask, short_mask, values, mult, margin_rate, size_holding, lag, cost_bps, capital,
            require_both_legs=require_both_legs, cohort_holding=spans, allocation=allocation,
            execution=execution, codes=codes, long_weight=long_weight, short_weight=short_weight,
            change=change,
        )
        if timings is not None:
            timings["sizing"] = time.perf_counter() - sizing_started
        zero_lot = np.zeros(n, dtype=int)
    else:
        if leg_notional is None:
            budget = float(capital if allocation == "equal_active" else capital / 2)
        else:
            budget = float(leg_notional)
        sized, zero_lot = _size_lots(
            long_mask, short_mask, price, specs, size_holding, lag=lag,
            leg_notional=np.full(n, budget),
            require_both_legs=require_both_legs, allocation=allocation,
            long_weight=long_weight, short_weight=short_weight,
        )
        lots = sized.to_numpy(float)
        if timings is not None:
            timings["sizing"] = time.perf_counter() - sizing_started
    roll_resize = price_change is not None and codes is not None
    daily, trades, positions = _account(
        lots, zero_lot, values, mult, margin_rate, dates, products, holding, lag, cost_bps, capital, compound,
        require_both_legs=require_both_legs, cohort_holding=spans, execution=execution, codes=codes,
        timings=timings, change=change, roll_resize=roll_resize,
    )
    return ResearchResult(daily=daily, trades=trades, positions=positions)


def _price_change(
    values: np.ndarray,
    price_change: pd.DataFrame | None,
    dates: pd.Index,
    products: list,
) -> np.ndarray:
    """盯市价差。缺省为价格差分；账户回测传入已剔除换月价差的差分。"""
    if price_change is None:
        change = np.full_like(values, np.nan)
        change[1:] = values[1:] - values[:-1]
        return change
    aligned = price_change.reindex(index=dates, columns=products).to_numpy(float)
    if aligned.shape != values.shape:
        raise ValueError("price_change 与价格矩阵形状不一致")
    return aligned


def _weight_values(weight: pd.DataFrame, dates: pd.Index, products: list) -> np.ndarray:
    return (
        weight.reindex(index=dates, columns=products)
        .apply(pd.to_numeric, errors="coerce")
        .fillna(0.0)
        .to_numpy(float)
    )


def _apply_weight_budget(
    values: np.ndarray,
    valid: np.ndarray,
    leg_notional: np.ndarray,
    holding: int,
) -> np.ndarray:
    """腿内按信号绝对值分配名义金额。无效品种权重记为 0。"""
    clipped = np.where(valid, np.clip(values, 0.0, None), 0.0)
    total = clipped.sum(axis=1)
    share = np.zeros_like(clipped)
    np.divide(clipped, total[:, None], out=share, where=total[:, None] > 0)
    return share * (leg_notional / holding)[:, None]


def _weight_budget(
    weight: pd.DataFrame,
    dates: pd.Index,
    products: list,
    valid: np.ndarray,
    leg_notional: np.ndarray,
    holding: int,
) -> np.ndarray:
    """腿内按信号绝对值分配名义金额。无效品种权重记为 0。"""
    return _apply_weight_budget(_weight_values(weight, dates, products), valid, leg_notional, holding)


def _aligned_masks(
    long_mask: pd.DataFrame,
    short_mask: pd.DataFrame,
    dates: pd.Index,
    products: list,
) -> tuple[np.ndarray, np.ndarray]:
    def bits(mask: pd.DataFrame) -> np.ndarray:
        return mask.reindex(index=dates, columns=products).astype("boolean").fillna(False).astype(bool).to_numpy()

    return bits(long_mask), bits(short_mask)


def _entry_prices(prices: np.ndarray, lag: int) -> np.ndarray:
    entry = np.full_like(prices, np.nan)
    entry[: len(prices) - lag] = prices[lag:]
    return entry


def _lots_on_entry(
    lm: np.ndarray,
    sm: np.ndarray,
    entry: np.ndarray,
    mult: np.ndarray,
    leg_notional: np.ndarray,
    holding: int,
    *,
    require_both_legs: bool = True,
    allocation: str = "leg",
    long_weight: np.ndarray | None = None,
    short_weight: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """按已经对齐的入场价定手。入场价是信号日之后第 lag 天的价格，调用方必须用完整序列算好。"""
    valid_l = lm & np.isfinite(entry)
    valid_s = sm & np.isfinite(entry)
    nl = valid_l.sum(axis=1)
    ns = valid_s.sum(axis=1)
    if allocation == "equal_active":
        each = np.where(nl + ns > 0, leg_notional / holding / np.maximum(nl + ns, 1), 0.0)
        budget_l = np.broadcast_to(each[:, None], valid_l.shape).copy()
        budget_s = np.broadcast_to(each[:, None], valid_s.shape).copy()
    elif long_weight is not None or short_weight is not None:
        if long_weight is None or short_weight is None:
            raise ValueError("多空权重必须同时给出")
        budget_l = _apply_weight_budget(long_weight, valid_l, leg_notional, holding)
        budget_s = _apply_weight_budget(short_weight, valid_s, leg_notional, holding)
    else:
        each_l = np.where(nl > 0, leg_notional / holding / np.maximum(nl, 1), 0.0)
        each_s = np.where(ns > 0, leg_notional / holding / np.maximum(ns, 1), 0.0)
        budget_l = np.broadcast_to(each_l[:, None], valid_l.shape).copy()
        budget_s = np.broadcast_to(each_s[:, None], valid_s.shape).copy()
    unit = np.where(np.isfinite(entry), entry * mult, np.nan)
    with np.errstate(invalid="ignore", divide="ignore"):
        lots_l = np.where(valid_l, np.floor(budget_l / unit), 0.0)
        lots_s = np.where(valid_s, np.floor(budget_s / unit), 0.0)
    lots_l = np.nan_to_num(lots_l)
    lots_s = np.nan_to_num(lots_s)
    zero_lot = (valid_l & (lots_l == 0)).sum(axis=1) + (valid_s & (lots_s == 0)).sum(axis=1)
    lots = lots_l - lots_s
    long_book = lots_l.sum(axis=1) > 0
    short_book = lots_s.sum(axis=1) > 0
    cohort_ok = (long_book & short_book) if require_both_legs else (long_book | short_book)
    lots = np.where(cohort_ok[:, None], lots, 0.0)
    return lots, zero_lot


def _size_lots(
    long_mask: pd.DataFrame,
    short_mask: pd.DataFrame,
    price: pd.DataFrame,
    specs: dict[str, dict],
    holding: int,
    *,
    lag: int,
    leg_notional: np.ndarray,
    require_both_legs: bool = True,
    allocation: str = "leg",
    long_weight: pd.DataFrame | None = None,
    short_weight: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, np.ndarray]:
    products = list(price.columns)
    dates = price.index
    lm, sm = _aligned_masks(long_mask, short_mask, dates, products)
    mult = np.array([specs[p]["multiplier"] for p in products])
    entry = _entry_prices(price.to_numpy(float), lag)
    long_values = None if long_weight is None else _weight_values(long_weight, dates, products)
    short_values = None if short_weight is None else _weight_values(short_weight, dates, products)
    lots, zero_lot = _lots_on_entry(
        lm, sm, entry, mult, leg_notional, holding,
        require_both_legs=require_both_legs, allocation=allocation,
        long_weight=long_values, short_weight=short_values,
    )
    frame = pd.DataFrame(lots, index=dates, columns=products)
    frame.attrs["zero_lot_names"] = zero_lot
    return frame, zero_lot


def _cohort_spans(cohort_holding: np.ndarray | None, n: int) -> np.ndarray | None:
    if cohort_holding is None:
        return None
    spans = np.asarray(cohort_holding, dtype=int)
    if spans.shape != (n,):
        raise ValueError("cohort_holding 必须与行情日期等长")
    if np.any(spans < 0):
        raise ValueError("cohort_holding 不能为负")
    return spans


def _compound_weight(
    weight: pd.DataFrame | None,
    dates: pd.Index,
    products: list,
) -> np.ndarray | None:
    if weight is None:
        return None
    frame = weight.copy()
    frame.columns = products
    return _weight_values(frame, dates, products)


@njit(cache=True)
def _compound_lots_numba(
    lm: np.ndarray,
    sm: np.ndarray,
    values: np.ndarray,
    mult: np.ndarray,
    holding: int,
    lag: int,
    rate: float,
    capital: float,
    require_both_legs: bool,
    roll: np.ndarray,
    moves: np.ndarray,
) -> np.ndarray:
    n, width = values.shape
    lots = np.zeros((n, width), dtype=np.float64)
    equity = capital
    for day in range(n):
        pnl = 0.0
        cost = 0.0
        start = day - (lag + holding + 1)
        if start < 0:
            start = 0
        for signal in range(start, day):
            age = day - signal
            if lag + 1 <= age <= lag + holding:
                for col in range(width):
                    moved = moves[day, col]
                    if lots[signal, col] != 0.0 and np.isfinite(moved):
                        pnl += lots[signal, col] * mult[col] * moved
            if age == lag + 1:
                entry_day = signal + lag
                for col in range(width):
                    if entry_day < n and np.isfinite(values[entry_day, col]):
                        cost += abs(lots[signal, col]) * values[entry_day, col] * mult[col] * rate
            if age == lag + holding + 1:
                exit_day = signal + lag + holding
                for col in range(width):
                    if exit_day < n and np.isfinite(values[exit_day, col]):
                        cost += abs(lots[signal, col]) * values[exit_day, col] * mult[col] * rate
        if day > 0:
            for col in range(width):
                if roll[day, col]:
                    previous = 0.0
                    current = 0.0
                    for signal in range(start, day):
                        previous_age = day - 1 - signal
                        current_age = day - signal
                        if lag + 1 <= previous_age <= lag + holding:
                            previous += lots[signal, col]
                        if lag + 1 <= current_age <= lag + holding:
                            current += lots[signal, col]
                    base = (abs(previous) + abs(current)) * values[day - 1, col] * mult[col] * rate
                    # Replace the per-symbol open/close costs on a roll day.
                    for signal in range(start, day):
                        age = day - signal
                        if age == lag + 1:
                            entry_day = signal + lag
                            if entry_day < n and np.isfinite(values[entry_day, col]):
                                cost -= abs(lots[signal, col]) * values[entry_day, col] * mult[col] * rate
                        if age == lag + holding + 1:
                            exit_day = signal + lag + holding
                            if exit_day < n and np.isfinite(values[exit_day, col]):
                                cost -= abs(lots[signal, col]) * values[exit_day, col] * mult[col] * rate
                    cost += base
        equity += pnl - cost
        budget = max(equity, 0.0) * 0.5
        long_count = 0
        short_count = 0
        entry_day = day + lag
        for col in range(width):
            valid = entry_day < n and np.isfinite(values[entry_day, col])
            if valid and lm[day, col]:
                long_count += 1
            if valid and sm[day, col]:
                short_count += 1
        long_each = budget / holding / long_count if long_count > 0 else 0.0
        short_each = budget / holding / short_count if short_count > 0 else 0.0
        row_long = 0
        row_short = 0
        for col in range(width):
            valid = entry_day < n and np.isfinite(values[entry_day, col])
            if valid and lm[day, col]:
                row_long += int(np.floor(long_each / (values[entry_day, col] * mult[col])))
            if valid and sm[day, col]:
                row_short += int(np.floor(short_each / (values[entry_day, col] * mult[col])))
        if (require_both_legs and row_long > 0 and row_short > 0) or (
            not require_both_legs and (row_long > 0 or row_short > 0)
        ):
            for col in range(width):
                valid = entry_day < n and np.isfinite(values[entry_day, col])
                if valid and lm[day, col]:
                    lots[day, col] = np.floor(long_each / (values[entry_day, col] * mult[col]))
                if valid and sm[day, col]:
                    lots[day, col] -= np.floor(short_each / (values[entry_day, col] * mult[col]))
    return lots


def _compound_lots(
    long_mask: pd.DataFrame,
    short_mask: pd.DataFrame,
    values: np.ndarray,
    mult: np.ndarray,
    margin_rate: np.ndarray,
    holding: int,
    lag: int,
    cost_bps: float,
    capital: float,
    require_both_legs: bool = True,
    cohort_holding: np.ndarray | None = None,
    allocation: str = "leg",
    execution: str = "cohort",
    codes: np.ndarray | None = None,
    long_weight: pd.DataFrame | None = None,
    short_weight: pd.DataFrame | None = None,
    change: np.ndarray | None = None,
) -> np.ndarray:
    """按信号日收盘后的权益逐批定手。当天盈亏只来自更早的批次。

    掩码和入场价只对齐一次。每一天只用当天权益计算这一行手数，公式与 ``_size_lots`` 相同。
    """
    if change is None:
        change = np.full_like(values, np.nan)
        change[1:] = values[1:] - values[:-1]
    if (
        cohort_holding is None
        and allocation == "leg"
        and execution == "cohort"
        and long_weight is None
        and short_weight is None
        and _day_flow is _ORIGINAL_DAY_FLOW
        and np.isfinite(values).all()
        and not long_mask.isna().to_numpy().any()
        and not short_mask.isna().to_numpy().any()
    ):
        lm, sm = _aligned_masks(long_mask, short_mask, long_mask.index, list(long_mask.columns))
        roll = np.zeros(values.shape, dtype=np.bool_) if codes is None else _roll_mask(codes)
        return _compound_lots_numba(
            np.ascontiguousarray(lm),
            np.ascontiguousarray(sm),
            np.ascontiguousarray(values, dtype=np.float64),
            np.ascontiguousarray(mult, dtype=np.float64),
            int(holding),
            int(lag),
            float(cost_bps / 10_000.0),
            float(capital),
            bool(require_both_legs),
            np.ascontiguousarray(roll),
            np.ascontiguousarray(change, dtype=np.float64),
        )
    dates_n, n_products = values.shape
    dates = long_mask.index
    products = list(range(n_products))
    price = pd.DataFrame(values, index=dates, columns=products)
    long_frame = long_mask.copy()
    short_frame = short_mask.copy()
    long_frame.columns = products
    short_frame.columns = products
    lm, sm = _aligned_masks(long_frame, short_frame, dates, products)
    entry = _entry_prices(price.to_numpy(float), lag)
    multiplier = np.array([float(mult[i]) for i in products])
    if long_weight is not None:
        long_frame_w = long_weight.copy()
        short_frame_w = short_weight.copy()
        long_frame_w.columns = products
        short_frame_w.columns = products
        long_values = _weight_values(long_frame_w, dates, products)
        short_values = _weight_values(short_frame_w, dates, products)
    else:
        long_values = None
        short_values = _compound_weight(short_weight, dates, products)
    lots = np.zeros_like(values)
    equity = float(capital)
    rate = cost_bps / 10_000.0
    share = 1.0 if allocation == "equal_active" else 0.5
    roll = None if codes is None else _roll_mask(codes)
    for day in range(dates_n):
        day_pnl, day_cost = _day_flow(
            lots, values, change, mult, day, holding, lag, rate, cohort_holding=cohort_holding,
            execution=execution, codes=codes, rolled=None if roll is None else roll[day],
        )
        equity = equity + day_pnl - day_cost
        budget = np.zeros(1)
        budget[0] = max(equity, 0.0) * share
        row, _zero = _lots_on_entry(
            lm[day:day + 1], sm[day:day + 1], entry[day:day + 1], multiplier, budget, holding,
            require_both_legs=require_both_legs, allocation=allocation,
            long_weight=None if long_values is None else long_values[day:day + 1],
            short_weight=None if short_values is None else short_values[day:day + 1],
        )
        lots[day] = row[0]
    return lots


def _signal_start(day: int, holding: int, lag: int, cohort_holding: np.ndarray | None) -> int:
    """当天盈亏和平仓成本只会用到这个信号日及之后的批次。"""
    if day <= 0:
        return 0
    span_cap = holding if cohort_holding is None else int(np.max(cohort_holding))
    return max(0, day - (lag + span_cap + 1))


def _day_flow(
    lots: np.ndarray,
    values: np.ndarray,
    change: np.ndarray,
    mult: np.ndarray,
    day: int,
    holding: int,
    lag: int,
    rate: float,
    cohort_holding: np.ndarray | None = None,
    execution: str = "cohort",
    codes: np.ndarray | None = None,
    rolled: np.ndarray | None = None,
) -> tuple[float, float]:
    pnl = 0.0
    cost = 0.0
    symbol_cost = np.zeros(lots.shape[1]) if codes is not None else None
    if codes is not None and rolled is None and day > 0:
        rolled = _code_changed(codes[day - 1], codes[day])
    n = len(lots)
    for signal in range(_signal_start(day, holding, lag, cohort_holding), day):
        span = holding if cohort_holding is None else int(cohort_holding[signal])
        if span < 1:
            continue
        age = day - signal
        if lag + 1 <= age <= lag + span and np.isfinite(change[day]).any():
            held = lots[signal]
            moved = change[day]
            contrib = held * mult * moved
            pnl += float(np.nansum(np.where(np.isfinite(contrib), contrib, 0.0)))
        if execution == "cohort" and age == lag + 1:
            entry = values[signal + lag] if signal + lag < n else np.full(values.shape[1], np.nan)
            opened = np.abs(lots[signal]) * np.nan_to_num(entry) * mult * rate
            cost += float(np.nansum(opened))
            if symbol_cost is not None:
                symbol_cost += opened
        if execution == "cohort" and age == lag + span + 1:
            exit_at = signal + lag + span
            exit_price = values[exit_at] if exit_at < n else np.full(values.shape[1], np.nan)
            closed = np.abs(lots[signal]) * np.nan_to_num(exit_price) * mult * rate
            cost += float(np.nansum(closed))
            if symbol_cost is not None:
                symbol_cost += closed
    if execution == "adjust" and day > 0:
        current = _position_on_day(lots, day, holding, lag, cohort_holding)
        previous = _position_on_day(lots, day - 1, holding, lag, cohort_holding)
        price = values[day - 1]
        traded = np.abs(current - previous) * np.nan_to_num(price) * mult * rate
        if codes is None:
            return pnl, float(np.nansum(traded))
        symbol_cost = _roll_symbol_cost(
            traded, previous, current, price, mult, rate, rolled=rolled,
            open_price=values[day],
        )
        return pnl, float(np.nansum(symbol_cost))
    if symbol_cost is not None and day > 0:
        current = _position_on_day(lots, day, holding, lag, cohort_holding)
        previous = _position_on_day(lots, day - 1, holding, lag, cohort_holding)
        price = values[day - 1]
        symbol_cost = _roll_symbol_cost(
            symbol_cost, previous, current, price, mult, rate, rolled=rolled,
            open_price=values[day],
        )
        cost = float(np.nansum(symbol_cost))
    return pnl, cost


_ORIGINAL_DAY_FLOW = _day_flow


def _position_on_day(
    lots: np.ndarray,
    day: int,
    holding: int,
    lag: int,
    cohort_holding: np.ndarray | None,
) -> np.ndarray:
    """当天仍在持有期内的各批手数之和。"""
    position = np.zeros(lots.shape[1])
    if day < 0:
        return position
    span_cap = holding if cohort_holding is None else int(np.max(cohort_holding))
    start = max(0, day - lag - span_cap)
    for signal in range(start, day + 1):
        span = holding if cohort_holding is None else int(cohort_holding[signal])
        if span < 1:
            continue
        age = day - signal
        if lag + 1 <= age <= lag + span:
            position += lots[signal]
    return position


def _account(
    lots: np.ndarray,
    zero_lot: np.ndarray,
    values: np.ndarray,
    mult: np.ndarray,
    margin_rate: np.ndarray,
    dates: pd.Index,
    products: list[str],
    holding: int,
    lag: int,
    cost_bps: float,
    capital: float,
    compound: bool,
    require_both_legs: bool = True,
    cohort_holding: np.ndarray | None = None,
    execution: str = "cohort",
    codes: np.ndarray | None = None,
    timings: dict[str, float] | None = None,
    change: np.ndarray | None = None,
    roll_resize: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    account_started = time.perf_counter()
    n, n_products = lots.shape
    if change is None:
        change = np.full_like(values, np.nan)
        change[1:] = values[1:] - values[:-1]
    entry = np.full_like(values, np.nan)
    entry[: n - lag] = values[lag:]
    unit = np.where(np.isfinite(entry), entry * mult, np.nan)
    notional_t = np.nansum(np.abs(lots) * np.nan_to_num(unit), axis=1)
    margin_t = np.nansum(np.abs(lots) * np.nan_to_num(unit) * margin_rate, axis=1)
    long_book = np.where(lots > 0, lots, 0).sum(axis=1) > 0
    short_book = np.where(lots < 0, -lots, 0).sum(axis=1) > 0
    cohort_ok = (long_book & short_book) if require_both_legs else (long_book | short_book)
    pnl = np.zeros(n)
    cost = np.zeros(n)
    notional = np.zeros(n)
    margin = np.zeros(n)
    missing = np.zeros(n, dtype=int)
    cohorts = np.zeros(n, dtype=int)
    symbol_pnl = np.zeros_like(lots)
    symbol_cost = np.zeros_like(lots)
    held = np.zeros_like(lots)
    rate = cost_bps / 10_000.0
    resize_holding = 1 if cohort_holding is not None else holding
    if roll_resize:
        if codes is None:
            raise ValueError("roll_resize 需要 contracts 代码矩阵")
        from backtest.reconcile import apply_roll_lot_resize_for_day

        lots_work = np.array(lots, dtype=float, copy=True)
        roll = _roll_mask(codes)
        exit_price = np.full_like(values, np.nan)
        exit_price[: n - lag - resize_holding] = values[lag + resize_holding :]
        exit_notional_w = np.nansum(np.abs(lots_work) * np.nan_to_num(exit_price) * mult, axis=1)
        for day in range(n):
            if cohort_holding is None:
                for offset in range(lag + 1, lag + holding + 1):
                    signal = day - offset
                    if signal < 0:
                        continue
                    moved = change[day]
                    raw = lots_work[signal] * mult * moved
                    finite = np.isfinite(raw)
                    pnl[day] += float(np.nansum(np.where(finite, raw, 0.0)))
                    symbol_pnl[day] += np.where(finite, raw, 0.0)
                    missing[day] += int(((lots_work[signal] != 0) & ~np.isfinite(moved)).sum())
                    notional[day] += notional_t[signal]
                    margin[day] += margin_t[signal]
                    cohorts[day] += cohort_ok[signal]
            else:
                for signal in range(n):
                    span = int(cohort_holding[signal])
                    if span < 1 or not cohort_ok[signal]:
                        continue
                    age = day - signal
                    if not (lag + 1 <= age <= lag + span):
                        continue
                    moved = change[day]
                    raw = lots_work[signal] * mult * moved
                    finite = np.isfinite(raw)
                    pnl[day] += float(np.nansum(np.where(finite, raw, 0.0)))
                    symbol_pnl[day] += np.where(finite, raw, 0.0)
                    missing[day] += int(((lots_work[signal] != 0) & ~np.isfinite(moved)).sum())
                    notional[day] += notional_t[signal]
                    margin[day] += margin_t[signal]
                    cohorts[day] += 1
            if day > 0 and roll[day].any():
                apply_roll_lot_resize_for_day(
                    lots_work, day, codes, values, mult,
                    lag=lag, holding=resize_holding, cohort_holding=cohort_holding,
                )
            held[day] = _position_on_day(
                lots_work, day, resize_holding, lag, cohort_holding,
            )
        if execution == "cohort":
            if cohort_holding is None:
                entry_h = lag + 1
                if entry_h < n:
                    opened = np.zeros_like(lots_work)
                    opened[entry_h:] = np.abs(lots_work[: n - entry_h]) * np.nan_to_num(entry[: n - entry_h]) * mult * rate
                    symbol_cost += opened
                    cost[entry_h:] += notional_t[: n - entry_h] * rate
                exit_h = lag + holding + 1
                if exit_h < n:
                    closed = np.zeros_like(lots_work)
                    closed[exit_h:] = np.abs(lots_work[: n - exit_h]) * np.nan_to_num(exit_price[: n - exit_h]) * mult * rate
                    symbol_cost += closed
                    cost[exit_h:] += exit_notional_w[: n - exit_h] * rate
            else:
                for signal in range(n):
                    span = int(cohort_holding[signal])
                    if span < 1 or not cohort_ok[signal]:
                        continue
                    entry_day = signal + lag + 1
                    if entry_day < n and signal + lag < n:
                        opened = np.abs(lots_work[signal]) * np.nan_to_num(entry[signal]) * mult * rate
                        symbol_cost[entry_day] += opened
                        cost[entry_day] += float(opened.sum())
                    exit_at = signal + lag + span
                    exit_day = exit_at + 1
                    if exit_day < n and exit_at < n:
                        closed = np.abs(lots_work[signal]) * np.nan_to_num(values[exit_at]) * mult * rate
                        symbol_cost[exit_day] += closed
                        cost[exit_day] += float(closed.sum())
    elif cohort_holding is None:
        exit_price = np.full_like(values, np.nan)
        exit_price[: n - lag - holding] = values[lag + holding :]
        exit_notional = np.nansum(np.abs(lots) * np.nan_to_num(exit_price) * mult, axis=1)
        for offset in range(lag + 1, lag + holding + 1):
            if offset >= n:
                continue
            shifted = np.full_like(change, np.nan)
            shifted[: n - offset] = change[offset:]
            raw = lots * mult * shifted
            miss = (lots != 0) & ~np.isfinite(shifted)
            contrib = np.where(np.isfinite(raw), raw, 0.0)
            target = np.arange(n - offset) + offset
            pnl[target] += contrib[: n - offset].sum(axis=1)
            symbol_pnl[target] += contrib[: n - offset]
            missing[target] += miss[: n - offset].sum(axis=1)
            notional[target] += notional_t[: n - offset]
            margin[target] += margin_t[: n - offset]
            cohorts[target] += cohort_ok[: n - offset]
            held[target] += lots[: n - offset]
        if execution == "cohort":
            entry_h = lag + 1
            if entry_h < n:
                opened = np.zeros_like(lots)
                opened[entry_h:] = np.abs(lots[: n - entry_h]) * np.nan_to_num(entry[: n - entry_h]) * mult * rate
                symbol_cost += opened
                cost[entry_h:] += notional_t[: n - entry_h] * rate
            exit_h = lag + holding + 1
            if exit_h < n:
                closed = np.zeros_like(lots)
                closed[exit_h:] = np.abs(lots[: n - exit_h]) * np.nan_to_num(exit_price[: n - exit_h]) * mult * rate
                symbol_cost += closed
                cost[exit_h:] += exit_notional[: n - exit_h] * rate
    else:
        for signal in range(n):
            span = int(cohort_holding[signal])
            if span < 1 or not cohort_ok[signal]:
                continue
            for age in range(lag + 1, lag + span + 1):
                day = signal + age
                if day >= n:
                    break
                moved = change[day]
                raw = lots[signal] * mult * moved
                finite = np.isfinite(raw)
                pnl[day] += float(np.nansum(np.where(finite, raw, 0.0)))
                symbol_pnl[day] += np.where(finite, raw, 0.0)
                missing[day] += int(((lots[signal] != 0) & ~np.isfinite(moved)).sum())
                notional[day] += notional_t[signal]
                margin[day] += margin_t[signal]
                cohorts[day] += 1
                held[day] += lots[signal]
            if execution == "cohort":
                entry_day = signal + lag + 1
                if entry_day < n and signal + lag < n:
                    opened = np.abs(lots[signal]) * np.nan_to_num(entry[signal]) * mult * rate
                    symbol_cost[entry_day] += opened
                    cost[entry_day] += float(opened.sum())
                exit_at = signal + lag + span
                exit_day = exit_at + 1
                if exit_day < n and exit_at < n:
                    closed = np.abs(lots[signal]) * np.nan_to_num(values[exit_at]) * mult * rate
                    symbol_cost[exit_day] += closed
                    cost[exit_day] += float(closed.sum())
    if execution == "adjust":
        cost, symbol_cost = _adjust_cost(held, values, mult, rate)
    roll = None if codes is None else _roll_mask(codes)
    if roll is not None:
        symbol_cost = _roll_book_cost(held, values, mult, rate, symbol_cost, roll)
        cost = symbol_cost.sum(axis=1)
    net = pnl - cost
    equity_before = np.full(n, float(capital))
    equity_after = np.full(n, float(capital))
    if compound:
        equity = float(capital)
        for day in range(n):
            equity_before[day] = equity
            equity = equity + net[day]
            equity_after[day] = equity
        ret = np.divide(net, equity_before, out=np.full(n, np.nan), where=equity_before != 0)
    else:
        equity_after = capital + np.cumsum(net)
        ret = net / capital
    gross = pnl / (equity_before if compound else capital)
    daily = pd.DataFrame({
        "pnl": pnl,
        "cost": cost,
        "notional": notional,
        "margin": margin,
        "ret": ret,
        "gross_ret": gross,
        "missing_legs": missing,
        "cohorts": cohorts,
        "n_long": (held > 0).sum(axis=1) if cohort_holding is not None else (lots > 0).sum(axis=1),
        "n_short": (held < 0).sum(axis=1) if cohort_holding is not None else (lots < 0).sum(axis=1),
        "zero_lot_names": zero_lot,
        "equity_before": equity_before,
        "equity_after": equity_after,
    }, index=dates)
    daily.loc[daily["cohorts"] == 0, ["ret", "gross_ret"]] = np.nan
    if timings is not None:
        timings["account_core"] = time.perf_counter() - account_started
    output_started = time.perf_counter()
    if execution == "adjust":
        trades = _adjust_trades(held, values, mult, dates, products, rate)
    else:
        trades = _trades(
            lots, values, mult, dates, products, holding, lag, rate, cohort_ok, cohort_holding=cohort_holding,
        )
    if roll is not None:
        trades = _rewrite_roll_trades(trades, held, codes, values, mult, dates, products, rate, roll)
    positions = _positions(held, symbol_pnl, symbol_cost, values, mult, dates, products)
    if timings is not None:
        timings["trade_positions"] = time.perf_counter() - output_started
    return daily, trades, positions


def _blank_code(value: object) -> str:
    """空值、浮点 NaN 和大小写 nan 都视为没有合约代码。其余保留原样，只去掉两端空白。"""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return ""
    text = str(value).strip()
    return "" if text.lower() == "nan" else text


def _normalize_codes(codes: np.ndarray) -> np.ndarray:
    flat = np.asarray(codes, dtype=object).ravel()
    return np.array([_blank_code(value) for value in flat], dtype=object).reshape(np.shape(codes))


def _roll_mask(codes: np.ndarray) -> np.ndarray:
    """每个交易日、每个品种是否换月。第 0 行没有上一交易日，全部为假。"""
    out = np.zeros(np.shape(codes), dtype=bool)
    if len(codes) < 2:
        return out
    prev = _normalize_codes(codes[:-1])
    cur = _normalize_codes(codes[1:])
    out[1:] = (prev != "") & (cur != "") & (prev != cur)
    return out


def _code_changed(previous: np.ndarray, current: np.ndarray) -> np.ndarray:
    """相邻两日都有合约代码且代码不同。"""
    stacked = np.vstack([
        np.asarray(previous, dtype=object).reshape(1, -1),
        np.asarray(current, dtype=object).reshape(1, -1),
    ])
    return _roll_mask(stacked)[1]


def _roll_symbol_cost(
    base: np.ndarray,
    previous_pos: np.ndarray,
    current_pos: np.ndarray,
    close_price: np.ndarray,
    mult: np.ndarray,
    rate: float,
    rolled: np.ndarray | None = None,
    previous_code: np.ndarray | None = None,
    current_code: np.ndarray | None = None,
    open_price: np.ndarray | None = None,
) -> np.ndarray:
    """换月日用平旧开新替换该品种当天的原交易成本。平仓用 close_price，开仓用 open_price。"""
    if rolled is None:
        if previous_code is None or current_code is None:
            raise ValueError("换月成本需要换月掩码或相邻两日的合约代码")
        rolled = _code_changed(previous_code, current_code)
    if not np.any(rolled):
        return base
    if open_price is None:
        open_price = close_price
    amount = (
        np.abs(previous_pos) * np.nan_to_num(close_price)
        + np.abs(current_pos) * np.nan_to_num(open_price)
    ) * mult * rate
    out = np.array(base, dtype=float, copy=True)
    out[rolled] = amount[rolled]
    return out


def _roll_book_cost(
    held: np.ndarray,
    values: np.ndarray,
    mult: np.ndarray,
    rate: float,
    symbol_cost: np.ndarray,
    roll: np.ndarray,
) -> np.ndarray:
    out = np.array(symbol_cost, dtype=float, copy=True)
    for day in range(1, len(held)):
        out[day] = _roll_symbol_cost(
            out[day], held[day - 1], held[day], values[day - 1], mult, rate, rolled=roll[day],
            open_price=values[day],
        )
    return out


def _rewrite_roll_trades(
    trades: pd.DataFrame,
    held: np.ndarray,
    codes: np.ndarray,
    values: np.ndarray,
    mult: np.ndarray,
    dates: pd.Index,
    products: list[str],
    rate: float,
    roll: np.ndarray | None = None,
) -> pd.DataFrame:
    if roll is None:
        roll = _roll_mask(codes)
    kept = trades.copy() if trades is not None and not trades.empty else pd.DataFrame(columns=[
        "cohort_date", "symbol", "side", "action", "trade_date", "lots", "price", "multiplier", "notional", "cost",
    ])
    if "trade_date" in kept.columns:
        kept["trade_date"] = pd.to_datetime(kept["trade_date"])
    rows: list[dict[str, object]] = []
    remove_keys: set[tuple[pd.Timestamp, str]] = set()
    for day in range(1, len(dates)):
        rolled = roll[day]
        if not np.any(rolled):
            continue
        if not kept.empty:
            trade_date = pd.Timestamp(dates[day])
            remove_keys.update(
                (trade_date, products[col]) for col in np.flatnonzero(rolled)
            )
        price_close = values[day - 1]
        price_open = values[day]
        for col in np.flatnonzero(rolled):
            close_px = price_close[col]
            open_px = price_open[col]
            if not np.isfinite(close_px):
                continue
            previous = float(held[day - 1, col])
            current = float(held[day, col])
            for lots, action, side, px in (
                (previous, "roll_close", "sell" if previous > 0 else "buy", close_px),
                (current, "roll_open", "buy" if current > 0 else "sell", open_px),
            ):
                if lots == 0 or not np.isfinite(px):
                    continue
                shares = abs(lots)
                notional = shares * float(px) * float(mult[col])
                rows.append({
                    "cohort_date": dates[day],
                    "symbol": products[col],
                    "side": side,
                    "action": action,
                    "trade_date": dates[day],
                    "lots": shares,
                    "price": px,
                    "multiplier": float(mult[col]),
                    "notional": notional,
                    "cost": notional * rate,
                })
    if remove_keys and not kept.empty:
        keys = pd.MultiIndex.from_arrays([kept["trade_date"], kept["symbol"]])
        kept = kept.loc[~keys.isin(remove_keys)]
    extra = pd.DataFrame(rows)
    if kept.empty:
        return extra
    if extra.empty:
        return kept.reset_index(drop=True)
    return pd.concat([kept, extra], ignore_index=True)


def _adjust_cost(
    held: np.ndarray,
    values: np.ndarray,
    mult: np.ndarray,
    rate: float,
) -> tuple[np.ndarray, np.ndarray]:
    """净持仓变化按上一交易日结算价计成本。"""
    cost = np.zeros(len(held))
    symbol_cost = np.zeros_like(held)
    for day in range(1, len(held)):
        amount = np.abs(held[day] - held[day - 1]) * np.nan_to_num(values[day - 1]) * mult
        symbol_cost[day] = amount * rate
        cost[day] = float(np.nansum(symbol_cost[day]))
    return cost, symbol_cost


def _adjust_trades(
    held: np.ndarray,
    values: np.ndarray,
    mult: np.ndarray,
    dates: pd.Index,
    products: list[str],
    rate: float,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for day in range(1, len(dates)):
        delta = held[day] - held[day - 1]
        price = values[day - 1]
        for col, symbol in enumerate(products):
            change = float(delta[col])
            if change == 0 or not np.isfinite(price[col]):
                continue
            px = float(price[col])
            shares = abs(change)
            notional = shares * px * float(mult[col])
            rows.append({
                "cohort_date": dates[day],
                "symbol": symbol,
                "side": "buy" if change > 0 else "sell",
                "action": "adjust",
                "trade_date": dates[day],
                "lots": shares,
                "price": px,
                "multiplier": float(mult[col]),
                "notional": notional,
                "cost": notional * rate,
            })
    return pd.DataFrame(rows)


def _trades(
    lots: np.ndarray,
    values: np.ndarray,
    mult: np.ndarray,
    dates: pd.Index,
    products: list[str],
    holding: int,
    lag: int,
    rate: float,
    cohort_ok: np.ndarray,
    cohort_holding: np.ndarray | None = None,
) -> pd.DataFrame:
    columns = {
        "cohort_date": [],
        "symbol": [],
        "side": [],
        "action": [],
        "trade_date": [],
        "lots": [],
        "price": [],
        "multiplier": [],
        "notional": [],
        "cost": [],
    }

    def append_row(**values: object) -> None:
        for key in columns:
            columns[key].append(values[key])

    n = len(dates)
    for signal in range(n):
        if not cohort_ok[signal]:
            continue
        span = holding if cohort_holding is None else int(cohort_holding[signal])
        if span < 1:
            continue
        open_at = signal + lag
        close_at = signal + lag + span
        open_book = signal + lag + 1
        close_book = signal + lag + span + 1
        for col, symbol in enumerate(products):
            lot = float(lots[signal, col])
            if lot == 0:
                continue
            side = "long" if lot > 0 else "short"
            shares = abs(lot)
            if open_at < n and open_book < n and np.isfinite(values[open_at, col]):
                price = float(values[open_at, col])
                notional = shares * price * float(mult[col])
                append_row(
                    cohort_date=dates[signal], symbol=symbol, side=side, action="open",
                    trade_date=dates[open_book], lots=shares, price=price,
                    multiplier=float(mult[col]), notional=notional, cost=notional * rate,
                )
            if close_at < n and close_book < n and np.isfinite(values[close_at, col]):
                price = float(values[close_at, col])
                notional = shares * price * float(mult[col])
                append_row(
                    cohort_date=dates[signal], symbol=symbol, side=side, action="close",
                    trade_date=dates[close_book], lots=shares, price=price,
                    multiplier=float(mult[col]), notional=notional, cost=notional * rate,
                )
    return pd.DataFrame(columns)


def _positions(
    held: np.ndarray,
    symbol_pnl: np.ndarray,
    symbol_cost: np.ndarray,
    values: np.ndarray,
    mult: np.ndarray,
    dates: pd.Index,
    products: list[str],
) -> pd.DataFrame:
    columns = {
        "date": [],
        "symbol": [],
        "lots": [],
        "notional": [],
        "pnl": [],
        "cost": [],
    }
    for day, date in enumerate(dates):
        for col, symbol in enumerate(products):
            lot = float(held[day, col])
            if lot == 0 and symbol_pnl[day, col] == 0 and symbol_cost[day, col] == 0:
                continue
            price = float(values[day, col]) if np.isfinite(values[day, col]) else float("nan")
            columns["date"].append(date)
            columns["symbol"].append(symbol)
            columns["lots"].append(lot)
            columns["notional"].append(
                abs(lot) * price * float(mult[col]) if np.isfinite(price) else float("nan")
            )
            columns["pnl"].append(float(symbol_pnl[day, col]))
            columns["cost"].append(float(symbol_cost[day, col]))
    return pd.DataFrame(columns)


def delta_rebalance_backtest(
    long_mask: pd.DataFrame,
    short_mask: pd.DataFrame,
    price: pd.DataFrame,
    specs: dict[str, dict],
    cohort_holding: np.ndarray,
    *,
    lag: int = 1,
    cost_bps: float = COST_BPS,
    capital: float = CAPITAL,
    compound: bool = True,
    contracts: pd.DataFrame | None = None,
    weights: pd.DataFrame | None = None,
    price_change: pd.DataFrame | None = None,
) -> ResearchResult:
    """周频调仓只交易目标手数和现有手数的差额。

    信号日按当时权益为有信号的品种等权定下目标手数。下一个交易日的结算价成交。
    成交日当天的涨跌仍算在旧仓上，新仓从再下一个交易日开始计盈亏。
    样本最后一批仓位在计划退出日平掉。
    ``contracts`` 给出每天的主力合约代码。账户换月日盯市仍只反映旧合约涨跌，
    换月后按旧名义在新 settle 上重算手数，并按平旧、开新各计一次成本。
    ``weights`` 给出信号日的目标名义占权益比例，含正负号。给出后不再把权益等分给有信号的品种。
    """
    if lag < 1:
        raise ValueError("lag 必须为正整数")
    if cost_bps < 0:
        raise ValueError("cost_bps 不能为负")
    products = list(price.columns)
    if set(products) - set(specs):
        raise ValueError("price 中存在没有交易规格的品种")
    dates = price.index
    n = len(dates)
    spans = _cohort_spans(cohort_holding, n)
    if spans is None:
        raise ValueError("差额调仓需要每个信号日的持有天数")
    values = price.to_numpy(float)
    codes = None if contracts is None else contracts.reindex(index=dates, columns=products).to_numpy(dtype=object)
    long_on = long_mask.reindex(index=dates, columns=products).astype("boolean").fillna(False).astype(bool).to_numpy()
    short_on = short_mask.reindex(index=dates, columns=products).astype("boolean").fillna(False).astype(bool).to_numpy()
    weight_on = None
    if weights is not None:
        weight_on = (
            weights.reindex(index=dates, columns=products)
            .apply(pd.to_numeric, errors="coerce")
            .fillna(0.0)
            .to_numpy(float)
        )
    mult = np.array([float(specs[p]["multiplier"]) for p in products])
    margin_rate = np.array([float(specs[p]["margin_rate"]) for p in products])
    change = _price_change(values, price_change, dates, products)
    signal_idx = np.flatnonzero(spans > 0)
    executions: dict[int, int | None] = {}
    for signal in signal_idx:
        execute_at = int(signal) + lag
        if execute_at < n:
            executions[execute_at] = int(signal)
    if len(signal_idx):
        last = int(signal_idx[-1])
        flatten_at = last + lag + int(spans[last])
        if flatten_at < n and flatten_at not in executions:
            executions[flatten_at] = None
    rate = cost_bps / 10_000.0
    position = np.zeros(len(products))
    targets: dict[int, np.ndarray] = {}
    scheduled_cost = np.zeros((n, len(products)))
    pnl = np.zeros(n)
    cost = np.zeros(n)
    notional = np.zeros(n)
    margin = np.zeros(n)
    missing = np.zeros(n, dtype=int)
    cohorts = np.zeros(n, dtype=int)
    symbol_pnl = np.zeros((n, len(products)))
    held = np.zeros((n, len(products)))
    zero_lot = np.zeros(n, dtype=int)
    equity = float(capital)
    equity_before = np.full(n, equity)
    equity_after = np.full(n, equity)
    trades: list[dict[str, object]] = []
    roll = None if codes is None else _roll_mask(codes)
    for day in range(n):
        equity_before[day] = equity
        moved = change[day]
        raw = position * mult * moved
        finite = np.isfinite(raw)
        day_pnl = float(np.nansum(np.where(finite, raw, 0.0)))
        symbol_pnl[day] = np.where(finite, raw, 0.0)
        missing[day] = int(((position != 0) & ~np.isfinite(moved)).sum())
        if price_change is not None and roll is not None and day > 0 and roll[day].any():
            from backtest.reconcile import apply_roll_position_resize_for_day

            apply_roll_position_resize_for_day(position, day, codes, values, mult)
        held[day] = position
        if roll is not None and day > 0:
            scheduled_cost[day] = _roll_symbol_cost(
                scheduled_cost[day], held[day - 1], held[day],
                values[day - 1], mult, rate, rolled=roll[day],
                open_price=values[day],
            )
        day_cost_vec = scheduled_cost[day]
        day_cost = float(day_cost_vec.sum())
        pnl[day] = day_pnl
        cost[day] = day_cost
        mark = values[day]
        live_notional = np.abs(position) * np.where(np.isfinite(mark), mark, 0.0) * mult
        notional[day] = float(live_notional.sum())
        margin[day] = float((live_notional * np.nan_to_num(margin_rate)).sum())
        cohorts[day] = int(np.any(position != 0))
        equity = equity + day_pnl - day_cost
        equity_after[day] = equity
        if int(spans[day]) > 0 and day + lag < n:
            budget = equity if compound else float(capital)
            unit = values[day + lag] * mult
            if weight_on is None:
                targets[day + lag] = _equal_active_lots(
                    budget, values[day + lag], long_on[day], short_on[day], mult,
                )
                active = (long_on[day] | short_on[day]) & np.isfinite(unit) & (unit != 0)
                if active.any():
                    each = budget / int(active.sum())
                    raw_lots = np.floor(each / unit)
                    zero_lot[day] = int((active & (np.nan_to_num(raw_lots) == 0)).sum())
            else:
                targets[day + lag] = _weighted_lots(budget, values[day + lag], weight_on[day], mult)
                active = (weight_on[day] != 0) & np.isfinite(unit) & (unit != 0)
                raw_lots = np.floor(np.abs(weight_on[day]) * budget / unit)
                zero_lot[day] = int((active & (np.nan_to_num(raw_lots) == 0)).sum())
        if day not in executions:
            continue
        signal = executions[day]
        target = targets.get(day, np.zeros(len(products)))
        delta = target - position
        trade_px = values[day]
        traded = (delta != 0) & np.isfinite(trade_px)
        trade_cost = np.abs(delta) * np.where(np.isfinite(trade_px), trade_px, 0.0) * mult * rate
        book = day + 1 if day + 1 < n else day
        scheduled_cost[book] += np.where(traded, trade_cost, 0.0)
        if book == day:
            extra = float(np.where(traded, trade_cost, 0.0).sum())
            cost[day] += extra
            equity -= extra
            equity_after[day] = equity
        signal_date = dates[signal] if signal is not None else dates[day]
        for col, symbol in enumerate(products):
            if not traded[col]:
                continue
            shares = abs(float(delta[col]))
            px = float(trade_px[col])
            amount = shares * px * float(mult[col])
            trades.append({
                "cohort_date": signal_date,
                "symbol": symbol,
                "side": "buy" if delta[col] > 0 else "sell",
                "action": "adjust",
                "trade_date": dates[book],
                "lots": shares,
                "price": px,
                "multiplier": float(mult[col]),
                "notional": amount,
                "cost": amount * rate,
            })
        position = np.where(np.isfinite(trade_px), target, position)
    net = pnl - cost
    ret = np.divide(net, equity_before, out=np.full(n, np.nan), where=equity_before != 0)
    gross = np.divide(pnl, equity_before, out=np.full(n, np.nan), where=equity_before != 0)
    daily = pd.DataFrame({
        "pnl": pnl,
        "cost": cost,
        "notional": notional,
        "margin": margin,
        "ret": ret,
        "gross_ret": gross,
        "missing_legs": missing,
        "cohorts": cohorts,
        "n_long": (held > 0).sum(axis=1),
        "n_short": (held < 0).sum(axis=1),
        "zero_lot_names": zero_lot,
        "equity_before": equity_before,
        "equity_after": equity_after,
    }, index=dates)
    inactive = daily["cohorts"].eq(0) & daily["cost"].eq(0) & daily["pnl"].eq(0)
    daily.loc[inactive, ["ret", "gross_ret"]] = np.nan
    trade_frame = pd.DataFrame(trades)
    if roll is not None:
        trade_frame = _rewrite_roll_trades(
            trade_frame, held, codes, values, mult, dates, products, rate, roll,
        )
    positions = _positions(held, symbol_pnl, scheduled_cost, values, mult, dates, products)
    return ResearchResult(daily=daily, trades=trade_frame, positions=positions)


def _weighted_lots(
    equity: float,
    exec_price: np.ndarray,
    weight_row: np.ndarray,
    mult: np.ndarray,
) -> np.ndarray:
    """按权益权重定手。权重是目标名义占权益的比例，含正负号，不再归一。"""
    unit = exec_price * mult
    lots = np.zeros(len(exec_price))
    active = np.isfinite(weight_row) & (weight_row != 0) & np.isfinite(unit) & (unit != 0)
    if equity <= 0 or not active.any():
        return lots
    sized = np.floor(np.abs(weight_row) * float(equity) / unit) * np.sign(weight_row)
    lots[active] = np.nan_to_num(sized[active])
    return lots


def _equal_active_lots(
    equity: float,
    exec_price: np.ndarray,
    long_row: np.ndarray,
    short_row: np.ndarray,
    mult: np.ndarray,
) -> np.ndarray:
    """有信号的品种平分权益，方向由多空掩码决定。"""
    unit = exec_price * mult
    active = (long_row | short_row) & np.isfinite(unit) & (unit != 0)
    lots = np.zeros(len(exec_price))
    count = int(active.sum())
    if count == 0 or equity <= 0:
        return lots
    sized = np.floor((float(equity) / count) / unit)
    sign = np.where(long_row, 1.0, -1.0)
    lots[active] = np.nan_to_num(sized[active]) * sign[active]
    return lots
