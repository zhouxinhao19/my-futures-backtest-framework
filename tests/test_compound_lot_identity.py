"""复利定手改成每天只算一行之后，手数和回测结果必须与整表算法逐位相同。"""

from __future__ import annotations

import numpy as np
import pandas as pd

from backtest import ledger


def _legacy_compound_lots(
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
    """修改前的复利定手：每个交易日用整表 ``_size_lots``，只保留当天那一行。"""
    dates_n, n_products = values.shape
    price = pd.DataFrame(values, index=long_mask.index)
    specs = {
        i: {"multiplier": float(mult[i]), "margin_rate": float(margin_rate[i])}
        for i in range(n_products)
    }
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
        day_pnl, day_cost = ledger._day_flow(
            lots, values, change, mult, day, holding, lag, rate, cohort_holding=cohort_holding,
            execution=execution, codes=codes,
        )
        equity = equity + day_pnl - day_cost
        budget = np.zeros(dates_n)
        share = 1.0 if allocation == "equal_active" else 0.5
        budget[day] = max(equity, 0.0) * share
        sized, _zero = ledger._size_lots(
            long_mask, short_mask, price, specs, holding, lag=lag, leg_notional=budget,
            require_both_legs=require_both_legs, allocation=allocation,
            long_weight=long_weight, short_weight=short_weight,
        )
        lots[day] = sized.to_numpy(float)[day]
    return lots


def _legacy_research(long_mask, short_mask, price, specs, holding, **kwargs):
    products = list(price.columns)
    dates = price.index
    n = len(dates)
    values = price.to_numpy(float)
    mult = np.array([float(specs[name]["multiplier"]) for name in products])
    margin_rate = np.array([float(specs[name]["margin_rate"]) for name in products])
    spans = ledger._cohort_spans(kwargs.get("cohort_holding"), n)
    size_holding = 1 if spans is not None else holding
    contracts = kwargs.get("contracts")
    codes = None if contracts is None else contracts.reindex(index=dates, columns=products).to_numpy(dtype=object)
    lots = _legacy_compound_lots(
        long_mask, short_mask, values, mult, margin_rate, size_holding, kwargs.get("lag", 1),
        kwargs.get("cost_bps", ledger.COST_BPS), kwargs.get("capital", ledger.CAPITAL),
        require_both_legs=kwargs.get("require_both_legs", True),
        cohort_holding=spans,
        allocation=kwargs.get("allocation", "leg"),
        execution=kwargs.get("execution", "cohort"),
        codes=codes,
        long_weight=kwargs.get("long_weight"),
        short_weight=kwargs.get("short_weight"),
    )
    zero_lot = np.zeros(n, dtype=int)
    return ledger._account(
        lots, zero_lot, values, mult, margin_rate, dates, products, holding, kwargs.get("lag", 1),
        kwargs.get("cost_bps", ledger.COST_BPS), kwargs.get("capital", ledger.CAPITAL), True,
        require_both_legs=kwargs.get("require_both_legs", True),
        cohort_holding=spans,
        execution=kwargs.get("execution", "cohort"),
        codes=codes,
    )


def _market(seed: int, nan_price: bool = False, missing_mask: bool = False):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2019-01-02", periods=40)
    symbols = ["A", "B", "C", "D"]
    levels = 50 + np.cumsum(rng.normal(0.0, 1.0, size=(len(dates), len(symbols))), axis=0)
    levels = np.abs(levels) + 20
    if nan_price:
        levels[5, 1] = np.nan
        levels[18, :] = np.nan
    price = pd.DataFrame(levels, index=dates, columns=symbols)
    signal = rng.normal(size=levels.shape)
    long = pd.DataFrame(signal > 0.2, index=dates, columns=symbols)
    short = pd.DataFrame(signal < -0.2, index=dates, columns=symbols)
    if missing_mask:
        long = long.astype("boolean")
        long.iloc[4, 0] = pd.NA
        short = short.drop(index=dates[6])
    specs = {
        name: {"multiplier": float(5 * (i + 1)), "margin_rate": 0.05 * (i + 1)}
        for i, name in enumerate(symbols)
    }
    bucket = np.arange(len(dates)) // 6
    contracts = pd.DataFrame(
        {name: [f"{name.lower()}{b}" for b in bucket] for name in symbols},
        index=dates,
    )
    weights = pd.DataFrame(rng.normal(size=levels.shape), index=dates, columns=symbols)
    weights.iloc[0, 0] = np.nan
    return long, short, price, specs, contracts, weights.clip(lower=-1)


def _assert_identical(long, short, price, specs, holding, **kwargs):
    current = ledger.research_backtest(long, short, price, specs, holding, compound=True, **kwargs)
    daily, trades, positions = _legacy_research(long, short, price, specs, holding, **kwargs)
    pd.testing.assert_frame_equal(current.daily, daily, check_exact=True)
    pd.testing.assert_frame_equal(current.trades, trades, check_exact=True)
    pd.testing.assert_frame_equal(current.positions, positions, check_exact=True)
    assert current.daily["equity_after"].notna().all()


def test_compound_one_row_matches_full_frame_leg_budget():
    long, short, price, specs, contracts, _weights = _market(1)
    _assert_identical(long, short, price, specs, 5, contracts=contracts, cost_bps=5.0, capital=20_000_000)


def test_compound_one_row_matches_equal_active_allocation():
    long, short, price, specs, contracts, _weights = _market(2)
    _assert_identical(
        long, short, price, specs, 3, contracts=contracts, allocation="equal_active",
        cost_bps=8.0, capital=15_000_000,
    )


def test_compound_one_row_matches_signal_weights():
    long, short, price, specs, contracts, weights = _market(3)
    _assert_identical(
        long, short, price, specs, 4, contracts=contracts,
        long_weight=weights.clip(lower=0), short_weight=(-weights).clip(lower=0),
        cost_bps=5.0, capital=30_000_000,
    )


def test_compound_one_row_matches_single_leg_books():
    long, short, price, specs, contracts, _weights = _market(4)
    _assert_identical(
        long, short, price, specs, 2, contracts=contracts, require_both_legs=False,
        cost_bps=5.0, capital=10_000_000,
    )


def test_compound_one_row_matches_variable_holding_and_adjust():
    long, short, price, specs, contracts, _weights = _market(5)
    spans = np.zeros(len(price), dtype=int)
    spans[::4] = 3
    spans[8] = 0
    _assert_identical(
        long, short, price, specs, 5, contracts=contracts, cohort_holding=spans,
        execution="adjust", cost_bps=5.0, capital=25_000_000,
    )


def test_compound_one_row_matches_when_prices_or_masks_are_missing():
    long, short, price, specs, contracts, _weights = _market(6, nan_price=True, missing_mask=True)
    _assert_identical(
        long, short, price, specs, 4, lag=2, contracts=contracts,
        cost_bps=12.0, capital=8_000_000,
    )


def test_compound_lots_array_matches_legacy_on_every_day():
    long, short, price, specs, contracts, weights = _market(7)
    products = list(price.columns)
    values = price.to_numpy(float)
    mult = np.array([float(specs[name]["multiplier"]) for name in products])
    margin = np.array([float(specs[name]["margin_rate"]) for name in products])
    codes = contracts.to_numpy(dtype=object)
    kwargs = dict(
        require_both_legs=True, allocation="leg", execution="cohort", codes=codes,
        long_weight=weights.clip(lower=0), short_weight=(-weights).clip(lower=0),
    )
    current = ledger._compound_lots(
        long, short, values, mult, margin, 5, 1, 5.0, 12_000_000, **kwargs,
    )
    legacy = _legacy_compound_lots(
        long, short, values, mult, margin, 5, 1, 5.0, 12_000_000, **kwargs,
    )
    assert np.array_equal(current, legacy)
    assert np.any(current != 0)


def _legacy_code_changed(previous: np.ndarray, current: np.ndarray) -> np.ndarray:
    """改前的逐日合约比较：每个交易日单独做 Series.map。"""
    def text(values: np.ndarray) -> pd.Series:
        return pd.Series(values, dtype="object").map(
            lambda value: "" if value is None or (isinstance(value, float) and np.isnan(value)) else str(value).strip()
        )

    prev = text(previous)
    cur = text(current)
    return (
        prev.ne("") & cur.ne("") & prev.str.lower().ne("nan") & cur.str.lower().ne("nan") & prev.ne(cur)
    ).to_numpy()


def _full_history_flow(
    lots, values, change, mult, day, holding, lag, rate, *,
    cohort_holding=None, execution="cohort", codes=None,
):
    """改前的复利扫描：每个交易日从第一批信号看到当天。"""
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
            contrib = lots[signal] * mult * change[day]
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
        current = ledger._position_on_day(lots, day, holding, lag, cohort_holding)
        previous = ledger._position_on_day(lots, day - 1, holding, lag, cohort_holding)
        price = values[day - 1]
        traded = np.abs(current - previous) * np.nan_to_num(price) * mult * rate
        if codes is None:
            return pnl, float(np.nansum(traded))
        adjusted = ledger._roll_symbol_cost(
            traded, previous, current, price, mult, rate,
            previous_code=codes[day - 1], current_code=codes[day],
            open_price=values[day],
        )
        return pnl, float(np.nansum(adjusted))
    if symbol_cost is not None and day > 0:
        current = ledger._position_on_day(lots, day, holding, lag, cohort_holding)
        previous = ledger._position_on_day(lots, day - 1, holding, lag, cohort_holding)
        price = values[day - 1]
        symbol_cost = ledger._roll_symbol_cost(
            symbol_cost, previous, current, price, mult, rate,
            previous_code=codes[day - 1], current_code=codes[day],
            open_price=values[day],
        )
        cost = float(np.nansum(symbol_cost))
    return pnl, cost


def test_roll_mask_matches_daily_code_compare():
    codes = np.array([
        [None, "au01", " nan ", "AU", "rb", ""],
        ["", "au01", "AU", "au", float("nan"), "nan"],
        ["au02", "AU01", "nan", "au", "rb", " RB "],
    ], dtype=object)
    mask = ledger._roll_mask(codes)
    assert not mask[0].any()
    for day in range(1, len(codes)):
        np.testing.assert_array_equal(mask[day], _legacy_code_changed(codes[day - 1], codes[day]))


def test_windowed_day_flow_matches_full_history_scan():
    long, short, price, specs, contracts, _weights = _market(11)
    spans = np.zeros(len(price), dtype=int)
    spans[::3] = 4
    spans[2] = 0
    values = price.to_numpy(float)
    change = np.full_like(values, np.nan)
    change[1:] = values[1:] - values[:-1]
    products = list(price.columns)
    mult = np.array([float(specs[name]["multiplier"]) for name in products])
    codes = contracts.to_numpy(dtype=object)
    lots = ledger._compound_lots(
        long, short, values, mult, np.zeros(len(products)), 5, 1, 5.0, 10_000_000,
        cohort_holding=spans, execution="adjust", codes=codes,
    )
    roll = ledger._roll_mask(codes)
    for day in range(len(price)):
        current = ledger._day_flow(
            lots, values, change, mult, day, 5, 1, 0.0005,
            cohort_holding=spans, execution="adjust", codes=codes, rolled=roll[day],
        )
        legacy = _full_history_flow(
            lots, values, change, mult, day, 5, 1, 0.0005,
            cohort_holding=spans, execution="adjust", codes=codes,
        )
        assert current == legacy
