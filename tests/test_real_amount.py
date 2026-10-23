import pandas as pd
import pytest

from backtest.ledger import continuous_backtest

DATES = pd.bdate_range("2020-01-02", periods=12)
SPECS = {"A": {"multiplier": 10.0, "margin_rate": 0.1}, "B": {"multiplier": 5.0, "margin_rate": 0.1}}


def _price(a_step=1.0, b_price=50.0):
    return pd.DataFrame({
        "A": [100.0 + a_step * i for i in range(len(DATES))],
        "B": [b_price] * len(DATES),
    }, index=DATES)


def _masks(long_names, short_names):
    lm = pd.DataFrame({p: [p in long_names] * len(DATES) for p in ["A", "B"]}, index=DATES)
    sm = pd.DataFrame({p: [p in short_names] * len(DATES) for p in ["A", "B"]}, index=DATES)
    return lm, sm


def test_pnl_in_yuan_is_lots_times_multiplier_times_adj_change():
    lm, sm = _masks(["A"], ["B"])
    out = continuous_backtest(lm, sm, _price(a_step=1.0), SPECS, holding=2, lag=1,
                              cost_bps=0.0, leg_notional=1_000_000.0, capital=1e8)
    # A 单名名义预算 5e5；手数按开仓价（T+1 后复权结算价 101+t）取整：
    # t=2 开仓价 103 → 485 手；t=3 开仓价 104 → 480 手；每日涨 1 元
    # 第 5 日合计 (485 + 480) * 10 * 1 = 9650；B 价格不变，空腿无盈亏
    assert out["pnl"].iloc[5] == pytest.approx((485 + 480) * 10.0)
    assert out["ret"].iloc[5] == pytest.approx((485 + 480) * 10.0 / 1e8)


def test_empty_leg_means_no_cohort_and_no_return():
    lm, sm = _masks(["A"], [])
    out = continuous_backtest(lm, sm, _price(), SPECS, holding=2, lag=1, cost_bps=0.0)
    assert out["cohorts"].eq(0).all()
    assert out["ret"].isna().all()


def test_budget_too_small_gives_zero_lots_not_filled():
    lm, sm = _masks(["A"], ["B"])
    out = continuous_backtest(lm, sm, _price(), SPECS, holding=2, lag=1,
                              cost_bps=0.0, leg_notional=2.0)
    assert out["pnl"].eq(0.0).all()
    assert out["zero_lot_names"].max() == 2


def test_cost_in_yuan_entry_and_exit_events():
    # 价格不变，只看成本。H=2、lag=1：开仓成本落在 t+2，平仓成本落在 t+4。
    # A：floor(5e5 / (100*10)) = 500 手 × 10 × 100 = 5e5；B：floor(5e5 / (50*5)) = 2000 手 × 5 × 50 = 5e5
    # 合计名义 N = 1e6，每次成本 = N × 5bp = 500 元
    lm, sm = _masks(["A"], ["B"])
    out = continuous_backtest(lm, sm, _price(a_step=0.0), SPECS, holding=2, lag=1,
                              cost_bps=5.0, leg_notional=1_000_000.0)
    assert out["cost"].iloc[2] == pytest.approx(500.0)
    assert out["cost"].iloc[5] == pytest.approx(1000.0)


def test_missing_price_is_counted_not_filled():
    price = _price()
    price.loc[DATES[4], "A"] = float("nan")
    lm, sm = _masks(["A"], ["B"])
    out = continuous_backtest(lm, sm, price, SPECS, holding=2, lag=1, cost_bps=0.0)
    assert out["missing_legs"].iloc[5] > 0


def test_raw_roll_change_uses_inferred_old_settle():
    from backtest.reconcile import raw_change_without_roll_gap

    dates = pd.to_datetime(["2024-01-02", "2024-01-03"])
    settle = pd.DataFrame({"RB": [100.0, 200.0]}, index=dates)
    adj = pd.DataFrame({"RB": [100.0, 110.0]}, index=dates)
    code = pd.DataFrame({"RB": ["a", "b"]}, index=dates)
    change = raw_change_without_roll_gap(settle, adj, code)
    assert change.loc[dates[1], "RB"] == 10.0


def test_frozen_lots_replay_matches_old_pnl_within_one_yuan():
    from backtest.reconcile import replay_adj_pnl
    from backtest.ledger import signal_lots

    lm, sm = _masks(["A"], ["B"])
    price = _price(a_step=1.0)
    original = continuous_backtest(
        lm, sm, price, SPECS, holding=2, lag=1,
        cost_bps=5.0, leg_notional=1_000_000.0, capital=1e8,
    )
    lots = signal_lots(lm, sm, price, SPECS, holding=2, lag=1, leg_notional=1_000_000.0)
    replay = replay_adj_pnl(lots, price, SPECS, holding=2, lag=1, cost_bps=5.0, capital=1e8)
    assert (replay["pnl"] - original["pnl"]).abs().max() <= 1.0
    assert (replay["cost"] - original["cost"]).abs().max() <= 1.0
