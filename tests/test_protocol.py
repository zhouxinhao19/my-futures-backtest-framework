import numpy as np
import pandas as pd

import pytest

from research.protocol import (
    annualization,
    canonical_sample_split,
    forward_open_return,
    lag_factor,
    oos_start,
    sample_split,
)


def test_lag_is_grouped_and_settle_to_settle():
    frame = pd.DataFrame({
        "symbol": ["A", "A", "A", "A", "B", "B", "B", "B"],
        "date": pd.date_range("2024-01-01", periods=4, freq="D").tolist() * 2,
        "factor": [1, 2, 3, 4, 10, 20, 30, 40],
        "open": [9, 10, 11, 12, 90, 100, 110, 120],
        "close": [10, 11, 13, 16, 100, 110, 130, 160],
        "settle": [10.5, 11.5, 13.5, 16.5, 105, 115, 135, 165],
        "adj_settle": [10.5, 11.5, 13.5, 16.5, 105, 115, 135, 165],
    })
    lagged = lag_factor(frame, "factor")
    assert pd.isna(lagged.iloc[0])
    assert pd.isna(lagged.iloc[4])
    assert lagged.iloc[2] == 2
    assert lagged.iloc[6] == 20

    returns = forward_open_return(frame)
    assert np.isclose(returns.iloc[0], 13.5 / 11.5 - 1)
    assert np.isclose(returns.iloc[4], 135 / 115 - 1)
    held = forward_open_return(frame, holding_days=2)
    assert np.isclose(held.iloc[0], 16.5 / 11.5 - 1)
    assert np.isclose(held.iloc[4], 165 / 115 - 1)


def test_execution_return_requires_settle():
    frame = pd.DataFrame({
        "symbol": ["A", "A"],
        "open": [10, 11],
        "adj_close": [10, 11],
        "close": [10, 11],
    })
    try:
        forward_open_return(frame)
    except ValueError as exc:
        assert "adj_settle" in str(exc)
    else:
        raise AssertionError("缺少结算价时必须停止")


def test_annualization_is_explicit():
    assert annualization("daily") == 243
    assert annualization("weekly") == 52


def test_oos_starts_in_2025():
    assert oos_start() == "2025-01-01"


def test_sample_split_uses_protocol_boundary():
    frame = pd.DataFrame({"date": ["2024-12-31", "2025-01-02", "2025-01-03"]})
    labels = sample_split(frame)
    assert labels.tolist() == ["is", "oos", "oos"]
    stamped = canonical_sample_split({"start": "2014-01-01"})
    assert stamped["oos_start"] == "2025-01-01"
    with pytest.raises(ValueError, match="2025-01-01"):
        canonical_sample_split({"oos_start": "2023-01-01"})
