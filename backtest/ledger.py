"""研究回测记账。

冻结权益时与原来的 ``continuous_backtest`` 相同。正式回测从初始权益开始，
当日净盈亏计入权益，下一批开仓按当时权益定手。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

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
) -> ResearchResult:
    """正式研究回测。``compound=False`` 时每批仍用初始权益的一半。

    ``cohort_holding`` 给出每个信号日各自的持有交易日数，0 表示当天不开仓。
    周频调仓用它，因为节假日周长短不同，而且同一时间只有一批仓位，
    定手不再按持有天数把资金拆开。
    ``execution="cohort"`` 时每一批单独开仓、到期整批平仓。
    ``execution="adjust"`` 时把同时持有的批次轧成净手数，只对净手数变化收取成本。
    ``contracts`` 给出每天的主力合约代码。代码变化且仍有持仓时，按平掉旧合约、开回新合约计成本，换月价差不进盈亏。
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
    change = np.full_like(values, np.nan)
    change[1:] = values[1:] - values[:-1]
    spans = _cohort_spans(cohort_holding, n)
    size_holding = 1 if spans is not None else holding
    codes = None if contracts is None else contracts.reindex(index=dates, columns=products).to_numpy(dtype=object)
    if compound:
        lots = _compound_lots(
            long_mask, short_mask, values, mult, margin_rate, size_holding, lag, cost_bps, capital,
            require_both_legs=require_both_legs, cohort_holding=spans, allocation=allocation,
            execution=execution, codes=codes, long_weight=long_weight, short_weight=short_weight,
        )
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
    if compound:
        zero_lot = np.zeros(n, dtype=int)
    daily, trades, positions = _account(
        lots, zero_lot, values, mult, margin_rate, dates, products, holding, lag, cost_bps, capital, compound,
        require_both_legs=require_both_legs, cohort_holding=spans, execution=execution, codes=codes,
    )
    return ResearchResult(daily=daily, trades=trades, positions=positions)


def _weight_budget(
    weight: pd.DataFrame,
    dates: pd.Index,
    products: list,
    valid: np.ndarray,
    leg_notional: np.ndarray,
    holding: int,
) -> np.ndarray:
    """腿内按信号绝对值分配名义金额。无效品种权重记为 0。"""
    values = weight.reindex(index=dates, columns=products).apply(pd.to_numeric, errors="coerce").fillna(0.0).to_numpy(float)
    values = np.where(valid, np.clip(values, 0.0, None), 0.0)
    total = values.sum(axis=1)
    share = np.zeros_like(values)
    np.divide(values, total[:, None], out=share, where=total[:, None] > 0)
    return share * (leg_notional / holding)[:, None]


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
    n = len(dates)
    lm = long_mask.reindex(index=dates, columns=products).astype("boolean").fillna(False).astype(bool).to_numpy()
    sm = short_mask.reindex(index=dates, columns=products).astype("boolean").fillna(False).astype(bool).to_numpy()
    mult = np.array([specs[p]["multiplier"] for p in products])
    prices = price.to_numpy(float)
    entry = np.full_like(prices, np.nan)
    entry[: n - lag] = prices[lag:]
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
        budget_l = _weight_budget(long_weight, dates, products, valid_l, leg_notional, holding)
        budget_s = _weight_budget(short_weight, dates, products, valid_s, leg_notional, holding)
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
) -> np.ndarray:
    """按信号日收盘后的权益逐批定手。当天盈亏只来自更早的批次。"""
    dates_n, n_products = values.shape
    price = pd.DataFrame(values, index=long_mask.index)
    specs = {
        i: {"multiplier": float(mult[i]), "margin_rate": float(margin_rate[i])}
        for i in range(n_products)
    }
    # _size_lots expects column labels. Use integer columns matching the masks.
    long_mask = long_mask.copy()
    short_mask = short_mask.copy()
    long_mask.columns = range(n_products)
    short_mask.columns = range(n_products)
    price.columns = range(n_products)
    if long_weight is not None:
        long_weight = long_weight.copy()
        short_weight = short_weight.copy()
        long_weight.columns = range(n_products)
        short_weight.columns = range(n_products)
    lots = np.zeros_like(values)
    equity = float(capital)
    rate = cost_bps / 10_000.0
    change = np.full_like(values, np.nan)
    change[1:] = values[1:] - values[:-1]
    for day in range(dates_n):
        day_pnl, day_cost = _day_flow(
            lots, values, change, mult, day, holding, lag, rate, cohort_holding=cohort_holding,
            execution=execution, codes=codes,
        )
        equity = equity + day_pnl - day_cost
        budget = np.zeros(dates_n)
        share = 1.0 if allocation == "equal_active" else 0.5
        budget[day] = max(equity, 0.0) * share
        sized, _zero = _size_lots(
            long_mask, short_mask, price, specs, holding, lag=lag, leg_notional=budget,
            require_both_legs=require_both_legs, allocation=allocation,
            long_weight=long_weight, short_weight=short_weight,
        )
        lots[day] = sized.to_numpy(float)[day]
    return lots


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
) -> tuple[float, float]:
    pnl = 0.0
    cost = 0.0
    symbol_cost = np.zeros(lots.shape[1]) if codes is not None else None
    n = len(lots)
    for signal in range(day):
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
        symbol_cost = _roll_symbol_cost(traded, previous, current, codes[day - 1], codes[day], price, mult, rate)
        return pnl, float(np.nansum(symbol_cost))
    if symbol_cost is not None and day > 0:
        current = _position_on_day(lots, day, holding, lag, cohort_holding)
        previous = _position_on_day(lots, day - 1, holding, lag, cohort_holding)
        price = values[day - 1]
        symbol_cost = _roll_symbol_cost(symbol_cost, previous, current, codes[day - 1], codes[day], price, mult, rate)
        cost = float(np.nansum(symbol_cost))
    return pnl, cost


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
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    n, n_products = lots.shape
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
    if cohort_holding is None:
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
    if codes is not None:
        symbol_cost = _roll_book_cost(held, codes, values, mult, rate, symbol_cost)
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
    if execution == "adjust":
        trades = _adjust_trades(held, values, mult, dates, products, rate)
    else:
        trades = _trades(
            lots, values, mult, dates, products, holding, lag, rate, cohort_ok, cohort_holding=cohort_holding,
        )
    if codes is not None:
        trades = _rewrite_roll_trades(trades, held, codes, values, mult, dates, products, rate)
    positions = _positions(held, symbol_pnl, symbol_cost, values, mult, dates, products)
    return daily, trades, positions


def _code_changed(previous: np.ndarray, current: np.ndarray) -> np.ndarray:
    """相邻两日都有合约代码且代码不同。"""
    def text(values: np.ndarray) -> pd.Series:
        return pd.Series(values, dtype="object").map(
            lambda value: "" if value is None or (isinstance(value, float) and np.isnan(value)) else str(value).strip()
        )
    prev = text(previous)
    cur = text(current)
    return (prev.ne("") & cur.ne("") & prev.str.lower().ne("nan") & cur.str.lower().ne("nan") & prev.ne(cur)).to_numpy()


def _roll_symbol_cost(
    base: np.ndarray,
    previous_pos: np.ndarray,
    current_pos: np.ndarray,
    previous_code: np.ndarray,
    current_code: np.ndarray,
    price: np.ndarray,
    mult: np.ndarray,
    rate: float,
) -> np.ndarray:
    """换月日用平旧开新替换该品种当天的原交易成本。"""
    rolled = _code_changed(previous_code, current_code)
    if not np.any(rolled):
        return base
    amount = (np.abs(previous_pos) + np.abs(current_pos)) * np.nan_to_num(price) * mult * rate
    out = np.array(base, dtype=float, copy=True)
    out[rolled] = amount[rolled]
    return out


def _roll_book_cost(
    held: np.ndarray,
    codes: np.ndarray,
    values: np.ndarray,
    mult: np.ndarray,
    rate: float,
    symbol_cost: np.ndarray,
) -> np.ndarray:
    out = np.array(symbol_cost, dtype=float, copy=True)
    for day in range(1, len(held)):
        out[day] = _roll_symbol_cost(
            out[day], held[day - 1], held[day], codes[day - 1], codes[day], values[day - 1], mult, rate,
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
) -> pd.DataFrame:
    kept = trades.copy() if trades is not None and not trades.empty else pd.DataFrame(columns=[
        "cohort_date", "symbol", "side", "action", "trade_date", "lots", "price", "multiplier", "notional", "cost",
    ])
    if "trade_date" in kept.columns:
        kept["trade_date"] = pd.to_datetime(kept["trade_date"])
    rows: list[dict[str, object]] = []
    for day in range(1, len(dates)):
        rolled = _code_changed(codes[day - 1], codes[day])
        if not np.any(rolled):
            continue
        names = [products[col] for col in np.flatnonzero(rolled)]
        if not kept.empty:
            kept = kept.loc[~((kept["trade_date"] == pd.Timestamp(dates[day])) & kept["symbol"].isin(names))]
        price = values[day - 1]
        for col in np.flatnonzero(rolled):
            if not np.isfinite(price[col]):
                continue
            px = float(price[col])
            previous = float(held[day - 1, col])
            current = float(held[day, col])
            for lots, action, side in (
                (previous, "roll_close", "sell" if previous > 0 else "buy"),
                (current, "roll_open", "buy" if current > 0 else "sell"),
            ):
                if lots == 0:
                    continue
                shares = abs(lots)
                notional = shares * px * float(mult[col])
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
    rows: list[dict[str, object]] = []
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
                rows.append({
                    "cohort_date": dates[signal],
                    "symbol": symbol,
                    "side": side,
                    "action": "open",
                    "trade_date": dates[open_book],
                    "lots": shares,
                    "price": price,
                    "multiplier": float(mult[col]),
                    "notional": notional,
                    "cost": notional * rate,
                })
            if close_at < n and close_book < n and np.isfinite(values[close_at, col]):
                price = float(values[close_at, col])
                notional = shares * price * float(mult[col])
                rows.append({
                    "cohort_date": dates[signal],
                    "symbol": symbol,
                    "side": side,
                    "action": "close",
                    "trade_date": dates[close_book],
                    "lots": shares,
                    "price": price,
                    "multiplier": float(mult[col]),
                    "notional": notional,
                    "cost": notional * rate,
                })
    return pd.DataFrame(rows)


def _positions(
    held: np.ndarray,
    symbol_pnl: np.ndarray,
    symbol_cost: np.ndarray,
    values: np.ndarray,
    mult: np.ndarray,
    dates: pd.Index,
    products: list[str],
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for day, date in enumerate(dates):
        for col, symbol in enumerate(products):
            lot = float(held[day, col])
            if lot == 0 and symbol_pnl[day, col] == 0 and symbol_cost[day, col] == 0:
                continue
            price = float(values[day, col]) if np.isfinite(values[day, col]) else float("nan")
            rows.append({
                "date": date,
                "symbol": symbol,
                "lots": lot,
                "notional": abs(lot) * price * float(mult[col]) if np.isfinite(price) else float("nan"),
                "pnl": float(symbol_pnl[day, col]),
                "cost": float(symbol_cost[day, col]),
            })
    return pd.DataFrame(rows)


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
) -> ResearchResult:
    """周频调仓只交易目标手数和现有手数的差额。

    信号日按当时权益为有信号的品种等权定下目标手数。下一个交易日的结算价成交。
    成交日当天的涨跌仍算在旧仓上，新仓从再下一个交易日开始计盈亏。
    样本最后一批仓位在计划退出日平掉。
    ``contracts`` 给出每天的主力合约代码。代码变化且仍有持仓时，按平掉旧合约、开回新合约计成本，换月价差不进盈亏。
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
    mult = np.array([float(specs[p]["multiplier"]) for p in products])
    margin_rate = np.array([float(specs[p]["margin_rate"]) for p in products])
    change = np.full_like(values, np.nan)
    change[1:] = values[1:] - values[:-1]
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
    for day in range(n):
        equity_before[day] = equity
        moved = change[day]
        raw = position * mult * moved
        finite = np.isfinite(raw)
        day_pnl = float(np.nansum(np.where(finite, raw, 0.0)))
        symbol_pnl[day] = np.where(finite, raw, 0.0)
        missing[day] = int(((position != 0) & ~np.isfinite(moved)).sum())
        held[day] = position
        if codes is not None and day > 0:
            scheduled_cost[day] = _roll_symbol_cost(
                scheduled_cost[day], held[day - 1], held[day],
                codes[day - 1], codes[day], values[day - 1], mult, rate,
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
            targets[day + lag] = _equal_active_lots(
                budget, values[day + lag], long_on[day], short_on[day], mult,
            )
            unit = values[day + lag] * mult
            active = (long_on[day] | short_on[day]) & np.isfinite(unit) & (unit != 0)
            if active.any():
                each = budget / int(active.sum())
                raw_lots = np.floor(each / unit)
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
    if codes is not None:
        trade_frame = _rewrite_roll_trades(trade_frame, held, codes, values, mult, dates, products, rate)
    positions = _positions(held, symbol_pnl, scheduled_cost, values, mult, dates, products)
    return ResearchResult(daily=daily, trades=trade_frame, positions=positions)


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
