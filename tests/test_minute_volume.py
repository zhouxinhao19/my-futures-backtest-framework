from __future__ import annotations

import pandas as pd
import pytest

from data.minute_volume import apply_minute_single_sided, minute_volume_convention_record


def _bars(day: str, exchange: str, volume: float = 100.0, amount: float = 80.0, oi: float = 40.0) -> pd.DataFrame:
    return pd.DataFrame({
        "symbol": ["RB"],
        "exchange": [exchange],
        "trade_date": [pd.Timestamp(day)],
        "datetime": [pd.Timestamp(f"{day} 09:05:00")],
        "close": [3500.0],
        "volume": [volume],
        "amount": [amount],
        "oi": [oi],
    })


def test_pre_2020_commodity_volume_is_halved():
    adjusted = apply_minute_single_sided(_bars("2019-12-31", "SHF"))
    assert adjusted.loc[0, "volume"] == 50.0
    assert adjusted.loc[0, "amount"] == 40.0
    assert adjusted.loc[0, "oi"] == 20.0
    assert adjusted.loc[0, "close"] == 3500.0


def test_exchange_code_case_does_not_matter():
    adjusted = apply_minute_single_sided(_bars("2019-06-03", "dce", volume=3))
    assert adjusted.loc[0, "volume"] == 1.5


def test_from_switch_date_onward_volume_stays():
    adjusted = apply_minute_single_sided(_bars("2020-01-02", "ZCE", volume=976947))
    assert adjusted.loc[0, "volume"] == 976947


def test_cffex_stays_single_sided_before_the_switch():
    adjusted = apply_minute_single_sided(_bars("2019-12-31", "CFFEX", volume=80))
    assert adjusted.loc[0, "volume"] == 80


def test_unknown_exchange_before_switch_stops():
    with pytest.raises(ValueError, match="未登记"):
        apply_minute_single_sided(_bars("2019-12-31", "GFEX"))


def test_missing_exchange_before_switch_stops():
    frame = _bars("2019-12-31", "SHF").drop(columns="exchange")
    with pytest.raises(ValueError, match="exchange"):
        apply_minute_single_sided(frame)


def test_price_only_bars_are_unchanged():
    frame = _bars("2019-12-31", "SHF")[["symbol", "trade_date", "close"]]
    adjusted = apply_minute_single_sided(frame)
    assert list(adjusted.columns) == ["symbol", "trade_date", "close"]
    assert adjusted.loc[0, "close"] == 3500.0


def test_convention_record_names_the_switch_date():
    record = minute_volume_convention_record()
    assert record["single_sided_since"] == "2020-01-01"
    assert record["read_adjustment"] == "divide_by_2"
    assert record["daily_panel"] == "already_single_sided"
    assert "SHF" in record["exchanges"]
    assert record["fields"] == ["volume", "amount", "oi"]
