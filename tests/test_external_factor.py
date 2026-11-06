import numpy as np
import pandas as pd
import pytest

from factors.signals.external_factor import (
    load_factor_wide,
    monthly_schedule,
    weekly_forward_returns,
    weekly_schedule,
)


def _write(tmp_path, text):
    path = tmp_path / "factor.csv"
    path.write_text(text, encoding="utf-8")
    return path


def test_blank_cells_follow_caller_choice(tmp_path):
    path = _write(tmp_path, "TRADE_DATE,A,B\n2024-01-02,,\n2024-01-03,0.5,\n2024-01-04,-1,2\n")
    kept, quality = load_factor_wide(path, ["A", "B"])
    assert kept["expression"].isna().sum() == 3
    assert quality["blank_cells"] == 3
    assert quality["first_non_blank_date"] == "2024-01-03"
    filled, quality = load_factor_wide(path, ["A", "B"], blank_value=0.0)
    assert filled["expression"].isna().sum() == 0
    assert quality["blank_value"] == 0.0
    b = filled.loc[filled["symbol"] == "B", "expression"].tolist()
    assert b == [0.0, 0.0, 2.0]


def test_columns_must_match_pool(tmp_path):
    path = _write(tmp_path, "TRADE_DATE,A,C\n2024-01-02,1,2\n")
    with pytest.raises(ValueError, match="不一致"):
        load_factor_wide(path, ["A", "B"])


def test_non_numeric_cell_stops(tmp_path):
    path = _write(tmp_path, "TRADE_DATE,A\n2024-01-02,abc\n")
    with pytest.raises(ValueError, match="非数值"):
        load_factor_wide(path, ["A"])


def test_weekly_schedule_uses_last_session_and_counts_holiday_weeks():
    days = pd.to_datetime([
        "2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05",
        "2024-01-08", "2024-01-09", "2024-01-10", "2024-01-11",
        "2024-01-15", "2024-01-16",
    ])
    schedule = weekly_schedule(days)
    assert schedule["signal_date"].dt.strftime("%Y-%m-%d").tolist() == ["2024-01-05", "2024-01-11"]
    assert schedule["execution_date"].dt.strftime("%Y-%m-%d").tolist() == ["2024-01-08", "2024-01-15"]
    assert schedule["holding_days"].tolist() == [4, 2]


def test_monthly_schedule_uses_last_session_of_each_month():
    days = pd.to_datetime([
        "2024-01-29", "2024-01-30", "2024-01-31",
        "2024-02-01", "2024-02-02",
        "2024-03-01",
    ])
    schedule = monthly_schedule(days)
    assert schedule["signal_date"].dt.strftime("%Y-%m-%d").tolist() == [
        "2024-01-31", "2024-02-02",
    ]
    assert schedule["execution_date"].dt.strftime("%Y-%m-%d").tolist() == [
        "2024-02-01", "2024-03-01",
    ]
    assert schedule["holding_days"].tolist() == [2, 1]


def test_weekly_return_runs_from_execution_to_next_execution():
    days = pd.bdate_range("2024-01-01", periods=12)
    price = np.arange(100.0, 112.0)
    panel = pd.concat([
        pd.DataFrame({"date": days, "symbol": "A", "adj_settle": price}).drop(index=6),
        pd.DataFrame({"date": days, "symbol": "B", "adj_settle": price * 2}),
    ])
    schedule = weekly_schedule(panel["date"])
    returns = weekly_forward_returns(panel, schedule).set_index(["date", "symbol"])
    assert np.isclose(returns.loc[(days[4], "A"), "future_return"], price[10] / price[5] - 1)
    assert np.isclose(returns.loc[(days[4], "B"), "future_return"], price[10] / price[5] - 1)
    assert returns.loc[(days[4], "A"), "holding_days"] == 5
    assert pd.isna(returns.loc[(days[9], "A"), "future_return"])

from factors.signals.external_factor import weekly_target_weights


def test_weekly_target_weights_match_price_rules():
    frame = pd.DataFrame({
        "execution_date": pd.to_datetime(["2024-01-02"] * 5),
        "symbol": list("ABCDE"),
        "factor_value": [-2.0, -1.0, 0.0, 1.0, 2.0],
    })
    cs = weekly_target_weights(frame, pipeline="cs_ls", direction="trend", leg_fraction=0.2)
    assert np.isclose(cs.iloc[0].sum(), 0.0)
    assert np.isclose(cs.iloc[0]["E"], 0.5) and np.isclose(cs.iloc[0]["A"], -0.5)
    ts = weekly_target_weights(frame, pipeline="ts_cta", direction="trend", leg_fraction=0.2)
    assert ts.iloc[0].get("C", 0.0) == 0.0
    assert np.isclose(ts.iloc[0].abs().sum(), 1.0)
    rev = weekly_target_weights(frame, pipeline="ts_cta", direction="reversal", leg_fraction=0.2)
    assert np.isclose(rev.iloc[0]["A"], 0.25)
