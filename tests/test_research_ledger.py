import pandas as pd
import pytest

from backtest.ledger import research_backtest
from backtest.ledger import continuous_backtest

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
