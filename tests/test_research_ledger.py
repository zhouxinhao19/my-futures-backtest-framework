import numpy as np
import pandas as pd
import pytest

from backtest.ledger import research_backtest
from backtest.ledger import continuous_backtest
from backtest.ledger import _compound_lots_numba
from backtest.masks import long_short_masks

DATES = pd.bdate_range("2020-01-02", periods=12)
SPECS = {"A": {"multiplier": 10.0, "margin_rate": 0.1}, "B": {"multiplier": 5.0, "margin_rate": 0.1}}


def _price(a_step=1.0):
    return pd.DataFrame({
        "A": [100.0 + a_step * i for i in range(len(DATES))],
        "B": [50.0] * len(DATES),
    }, index=DATES)


def _masks():
    long = pd.DataFrame({"A": True, "B": False}, index=DATES)
    short = pd.DataFrame({"A": False, "B": True}, index=DATES)
    return long, short


def _reference_compound_lots(values, long_mask, short_mask, mult, holding, lag, capital):
    lots = np.zeros_like(values, dtype=float)
    equity = float(capital)
    for day in range(len(values)):
        pnl = 0.0
        start = max(0, day - (lag + holding + 1))
        for signal in range(start, day):
            age = day - signal
            if lag + 1 <= age <= lag + holding:
                moved = values[day] - values[day - 1]
                pnl += np.nansum(np.where(np.isfinite(moved), lots[signal] * mult * moved, 0.0))
        equity += pnl
        entry_day = day + lag
        if entry_day >= len(values):
            continue
        valid_l = long_mask[day] & np.isfinite(values[entry_day])
        valid_s = short_mask[day] & np.isfinite(values[entry_day])
        each_l = equity * 0.5 / holding / max(int(valid_l.sum()), 1)
        each_s = equity * 0.5 / holding / max(int(valid_s.sum()), 1)
        long_lots = np.where(valid_l, np.floor(each_l / (values[entry_day] * mult)), 0.0)
        short_lots = np.where(valid_s, np.floor(each_s / (values[entry_day] * mult)), 0.0)
        if long_lots.sum() > 0 and short_lots.sum() > 0:
            lots[day] = long_lots - short_lots
    return lots


def test_trade_costs_match_daily_cost_within_one_yuan():
    long, short = _masks()
    result = research_backtest(
        long, short, _price(0.0), SPECS, holding=2, lag=1,
        cost_bps=5.0, capital=1e8, compound=False, leg_notional=1_000_000.0,
    )
    booked = result.trades.groupby("trade_date")["cost"].sum()
    daily = result.daily["cost"].copy()
    daily.index = pd.to_datetime(daily.index)
    aligned = booked.reindex(daily.index).fillna(0.0)
    assert (aligned - daily).abs().max() <= 1.0


def test_numba_compound_sizing_matches_python_reference():
    values = np.array([
        [100.0, 50.0],
        [101.0, 49.0],
        [102.0, 48.0],
        [99.0, 51.0],
        [98.0, 52.0],
        [100.0, 50.0],
        [103.0, 47.0],
        [104.0, 46.0],
    ])
    long_mask = np.ones(values.shape, dtype=bool)
    short_mask = np.ones(values.shape, dtype=bool)
    mult = np.array([10.0, 5.0])
    expected = _reference_compound_lots(
        values, long_mask, short_mask, mult, holding=2, lag=1, capital=1_000_000.0,
    )
    actual = _compound_lots_numba(
        long_mask, short_mask, values, mult, 2, 1, 0.0, 1_000_000.0, True,
        np.zeros(values.shape, dtype=bool),
        np.vstack([np.full(values.shape[1], np.nan), np.diff(values, axis=0)]),
    )
    np.testing.assert_array_equal(actual, expected)


def test_frozen_daily_matches_legacy_within_one_yuan():
    long, short = _masks()
    price = _price(1.0)
    legacy = continuous_backtest(
        long, short, price, SPECS, holding=2, lag=1,
        cost_bps=5.0, leg_notional=1_000_000.0, capital=1e8,
    )
    result = research_backtest(
        long, short, price, SPECS, holding=2, lag=1,
        cost_bps=5.0, capital=1e8, compound=False, leg_notional=1_000_000.0,
    )
    assert (result.daily["pnl"] - legacy["pnl"]).abs().max() <= 1.0
    assert (result.daily["cost"] - legacy["cost"]).abs().max() <= 1.0


def test_adjust_keeps_pnl_and_drops_roll_cost_when_the_book_is_unchanged():
    long, short = _masks()
    price = _price(0.0)
    common = dict(
        lag=1, cost_bps=5.0, capital=1e8, compound=False, leg_notional=1_000_000.0,
    )
    cohort = research_backtest(long, short, price, SPECS, holding=2, execution="cohort", **common)
    adjusted = research_backtest(long, short, price, SPECS, holding=2, execution="adjust", **common)
    assert (adjusted.daily["pnl"] - cohort.daily["pnl"]).abs().max() == 0
    # 价格不变且每天信号相同，满仓后净手数不再变化，调整不再支付开平成本。
    assert adjusted.daily["cost"].iloc[-1] == 0
    assert cohort.daily["cost"].iloc[-1] > 0
    assert adjusted.daily["cost"].sum() < cohort.daily["cost"].sum()
    booked = adjusted.trades.groupby("trade_date")["cost"].sum()
    daily = adjusted.daily["cost"].copy()
    daily.index = pd.to_datetime(daily.index)
    aligned = booked.reindex(daily.index).fillna(0.0)
    assert (aligned - daily).abs().max() <= 1.0


def test_leg_budget_follows_signal_weights():
    dates = DATES[:4]
    specs = {"A": {"multiplier": 10.0, "margin_rate": 0.1}, "B": {"multiplier": 10.0, "margin_rate": 0.1}, "C": {"multiplier": 10.0, "margin_rate": 0.1}}
    price = pd.DataFrame(100.0, index=dates, columns=["A", "B", "C"])
    long = pd.DataFrame({"A": True, "B": True, "C": False}, index=dates)
    short = pd.DataFrame({"A": False, "B": False, "C": True}, index=dates)
    long_w = pd.DataFrame({"A": 2.0, "B": 1.0, "C": 0.0}, index=dates)
    short_w = pd.DataFrame({"A": 0.0, "B": 0.0, "C": 1.0}, index=dates)
    result = research_backtest(
        long, short, price, specs, holding=1, lag=1, cost_bps=0.0, capital=1e8, compound=False,
        leg_notional=3_000_000.0, long_weight=long_w, short_weight=short_w,
    )
    opened = result.trades[result.trades["action"] == "open"]
    lots = opened.groupby("symbol")["lots"].first()
    assert lots["A"] == pytest.approx(2 * lots["B"])


def test_roll_charges_close_and_reopen_when_lots_do_not_change():
    long, short = _masks()
    price = _price(0.0)
    codes = pd.DataFrame({"A": ["a1"] * len(DATES), "B": ["b1"] * len(DATES)}, index=DATES)
    codes.iloc[6:, 0] = "a2"
    common = dict(lag=1, cost_bps=5.0, capital=1e8, compound=False, leg_notional=1_000_000.0)
    plain = research_backtest(
        long, short, price, SPECS, holding=2, execution="adjust", **common,
    )
    rolled = research_backtest(
        long, short, price, SPECS, holding=2, execution="adjust", contracts=codes, **common,
    )
    roll_day = DATES[6]
    assert plain.daily.loc[roll_day, "cost"] == 0
    assert rolled.daily.loc[roll_day, "cost"] > plain.daily.loc[roll_day, "cost"]
    assert (rolled.daily["pnl"] - plain.daily["pnl"]).abs().max() == 0
    day = rolled.trades[pd.to_datetime(rolled.trades["trade_date"]) == roll_day]
    assert set(day.loc[day["symbol"] == "A", "action"]) == {"roll_close", "roll_open"}
    booked = rolled.trades.groupby("trade_date")["cost"].sum()
    daily = rolled.daily["cost"].copy()
    daily.index = pd.to_datetime(daily.index)
    aligned = booked.reindex(daily.index).fillna(0.0)
    assert (aligned - daily).abs().max() <= 1.0


def test_compound_equity_changes_after_pnl_and_return_uses_prior_equity():
    long, short = _masks()
    result = research_backtest(
        long, short, _price(1.0), SPECS, holding=2, lag=1,
        cost_bps=0.0, capital=1e8, compound=True,
    )
    active = result.daily[result.daily["cohorts"].gt(0) & result.daily["pnl"].ne(0)]
    assert not active.empty
    day = active.index[0]
    net = active.loc[day, "pnl"] - active.loc[day, "cost"]
    assert active.loc[day, "ret"] == pytest.approx(net / active.loc[day, "equity_before"])
    assert result.daily["equity_after"].iloc[-1] != pytest.approx(1e8)


def test_roll_resize_scales_lots_after_main_change():
    from backtest.reconcile import apply_roll_lot_resize_for_day

    lots = np.zeros((4, 1))
    lots[0, 0] = 10.0
    codes = np.array([["a1"], ["a1"], ["a2"], ["a2"]], dtype=object)
    values = np.array([[100.0], [100.0], [200.0], [200.0]])
    mult = np.array([1.0])
    apply_roll_lot_resize_for_day(lots, 2, codes, values, mult, lag=1, holding=2, cohort_holding=None)
    assert lots[0, 0] == pytest.approx(5.0)


def test_account_roll_gap_is_excluded_from_pnl():
    from backtest.reconcile import raw_change_without_roll_gap

    dates = pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05"])
    settle = pd.DataFrame({
        "A": [100.0, 101.0, 200.0, 210.0],
        "B": [50.0, 50.0, 50.0, 50.0],
    }, index=dates)
    adj = pd.DataFrame({
        "A": [100.0, 101.0, 111.0, 116.55],
        "B": [50.0, 50.0, 50.0, 50.0],
    }, index=dates)
    codes = pd.DataFrame({
        "A": ["a1", "a1", "a2", "a2"],
        "B": ["b1", "b1", "b1", "b1"],
    }, index=dates)
    long = pd.DataFrame(True, index=dates, columns=["A", "B"])
    short = pd.DataFrame({"A": False, "B": True}, index=dates)
    change = raw_change_without_roll_gap(settle, adj, codes)
    result = research_backtest(
        long, short, settle, SPECS, holding=2, lag=1, cost_bps=0.0,
        capital=1e8, compound=False, leg_notional=1_000_000.0, contracts=codes,
        price_change=change,
    )
    naive = research_backtest(
        long, short, settle, SPECS, holding=2, lag=1, cost_bps=0.0,
        capital=1e8, compound=False, leg_notional=1_000_000.0, contracts=codes,
    )
    roll_day = dates[2]
    assert change.loc[roll_day, "A"] == pytest.approx(10.0)
    assert abs(result.daily.loc[roll_day, "pnl"]) < abs(naive.daily.loc[roll_day, "pnl"])


def test_four_modes_are_masks_not_account_weights():
    factor = pd.DataFrame({"A": [1.0, -2.0], "B": [-1.0, 3.0]}, index=DATES[:2])
    long, short = long_short_masks(factor, "time_series_trend")
    assert bool(long.iloc[0]["A"]) and bool(short.iloc[0]["B"])
    assert set(long.to_numpy().ravel()).issubset({True, False})
