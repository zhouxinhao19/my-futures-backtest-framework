import pandas as pd

from research.benchmarks import buy_and_hold, calendar_spread, daily_open_close
from research.runtime import ExecutionConfig


SPEC = {
    "variety": "AU",
    "multiplier": 10,
    "margin_rate": 0.10,
    "tick_size": 1,
    "commission_per_lot": 5,
}


def _frame():
    rows = [
        ("2024-01-02", "au2406", 100, 110, 110, 1000, "au2406"),
        ("2024-01-03", "au2406", 110, 112, 112, 1000, "au2406"),
        ("2024-01-04", "au2406", 113, 114, 114, 1000, "au2409"),
        ("2024-01-04", "au2409", 120, 125, 125, 1000, "au2409"),
    ]
    return pd.DataFrame(rows, columns=[
        "date", "code", "open", "close", "settle", "volume", "main_contract",
    ]).assign(date=lambda x: pd.to_datetime(x["date"]))


def _execution(**kwargs):
    return ExecutionConfig(
        initial_cash=1_000_000,
        slippage_ticks=0,
        slippage_pct_floor=0,
        slippage_pct_ceiling=0,
        participation_rate=1.0,
        **kwargs,
    )


def test_daily_open_close_matches_gross_and_two_commissions():
    result = daily_open_close(_frame().iloc[[0]], SPEC, _execution(), lots=2)
    assert result.nav.iloc[-1]["equity"] == 1_000_180
    assert result.nav.iloc[-1]["gross_pnl"] == 200
    assert result.trades["commission"].sum() == 20
    assert len(result.trades) == 2


def test_buy_hold_rolls_old_contract_before_new_contract():
    main = _frame()[["date", "main_contract"]].drop_duplicates()
    main = main.rename(columns={"main_contract": "contract"}).assign(adj_close=[100, 112, 125])
    result = buy_and_hold(_frame(), main, SPEC, _execution(), lots=1)
    assert list(result.trades["reason"]) == ["initial_open", "roll_close", "roll_open"]
    assert result.trades.iloc[1]["code"] == "au2406"
    assert result.trades.iloc[2]["code"] == "au2409"
    assert result.positions.iloc[-1]["code"] == "au2409"


def test_calendar_spread_has_two_synchronized_legs_and_single_side_margin():
    frame = pd.DataFrame([
        ("2024-01-02", "au2406", 100, 101, 101, 1000),
        ("2024-01-02", "au2409", 110, 109, 109, 900),
        ("2024-01-03", "au2406", 102, 103, 103, 1000),
        ("2024-01-03", "au2409", 111, 110, 110, 900),
    ], columns=["date", "code", "open", "close", "settle", "volume"])
    frame["date"] = pd.to_datetime(frame["date"])
    result = calendar_spread(frame, SPEC, _execution(), lots=1, margin_mode="max")
    assert len(result.trades) == 2
    assert set(result.trades["lots"]) == {1, -1}
    assert result.nav.iloc[-1]["gross_spread_pnl"] == 10
    assert result.nav.iloc[-1]["margin"] == 110
