import numpy as np
import pandas as pd
import pytest

from factors.signals.liquidity import (
    apply_mask,
    blank_long_signals,
    liquidity_mask,
    product_turnover,
    research_main_liquidity_mask,
    research_turnover_matrix,
    smooth_by_product,
)


def _prepared():
    dates = pd.bdate_range("2020-01-01", periods=25)
    rows = []
    for d in dates:
        rows.append({"contract_object": "A", "date": d, "contract_key": "A2001@x", "turnover_value": 2e9})
        rows.append({"contract_object": "A", "date": d, "contract_key": "A2005@x", "turnover_value": 1e9})
        rows.append({"contract_object": "B", "date": d, "contract_key": "B2001@x", "turnover_value": 5e8})
    return pd.DataFrame(rows)


def test_product_turnover_sums_all_contracts_per_product_day():
    t = product_turnover(_prepared())
    assert t["A"].iloc[0] == pytest.approx(3e9)
    assert t["B"].iloc[0] == pytest.approx(5e8)


def test_mask_requires_full_window_and_threshold():
    t = product_turnover(_prepared())
    m = liquidity_mask(t, window=20, threshold=1e9)
    # 前 19 日不足窗口，不合格
    assert not m["A"].iloc[:19].any()
    # A 日均 3e9 >= 1e9，合格；B 日均 5e8 < 1e9，不合格
    assert m["A"].iloc[19:].all()
    assert not m["B"].any()


def test_apply_mask_blanks_ineligible_and_keeps_eligible():
    t = product_turnover(_prepared())
    m = liquidity_mask(t, window=20, threshold=1e9)
    f = pd.DataFrame({"A": np.ones(len(t)), "B": np.ones(len(t))}, index=t.index)
    out = apply_mask(f, m)
    assert out["B"].isna().all()
    assert out["A"].iloc[:19].isna().all()
    assert out["A"].iloc[19:].eq(1.0).all()


def _main_panel() -> pd.DataFrame:
    dates = pd.bdate_range("2020-01-01", periods=20)
    return pd.DataFrame({
        "date": list(dates) * 2,
        "symbol": ["RB"] * 20 + ["AU"] * 20,
        "turnover_value": [1_200_000_000.0] * 20 + [400_000_000.0] * 20,
    })


def test_research_main_mask_uses_each_products_own_turnover():
    mask = research_main_liquidity_mask(_main_panel())
    assert bool(mask.loc[mask.index[-1], "RB"])
    assert not bool(mask["AU"].any())
    assert not bool(mask["RB"].iloc[0])


def test_research_turnover_requires_the_main_contract_field():
    panel = _main_panel().drop(columns=["turnover_value"])
    with pytest.raises(ValueError, match="trade_turnover_value"):
        research_turnover_matrix(panel)


def test_smooth_uses_own_calendar_and_requires_full_window():
    idx = pd.bdate_range("2020-01-01", periods=6)
    f = pd.DataFrame({"A": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0], "B": [np.nan, 1.0, 1.0, 1.0, 1.0, 1.0]}, index=idx)
    out = smooth_by_product(f, 3)
    assert out["A"].iloc[2] == pytest.approx(2.0)
    assert out["A"].iloc[:2].isna().all()
    assert out["B"].iloc[3] == pytest.approx(1.0)
    assert out["B"].iloc[:3].isna().all()


def test_window_must_be_positive():
    with pytest.raises(ValueError, match="window"):
        smooth_by_product(pd.DataFrame({"A": [1.0]}), 0)


def test_blank_long_signals_removes_names_before_they_can_rank():
    panel = _main_panel()
    panel["factor"] = 1.0
    out = blank_long_signals(panel, ["factor"])
    last = out["date"] == out["date"].max()
    assert out.loc[last & out["symbol"].eq("RB"), "factor"].notna().all()
    assert out.loc[out["symbol"].eq("AU"), "factor"].isna().all()
