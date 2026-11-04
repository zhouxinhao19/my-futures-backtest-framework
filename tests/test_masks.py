import pandas as pd

from backtest.masks import long_short_masks


def test_quantile_reversal_takes_the_low_tail():
    factor = pd.DataFrame(
        [[1.0, 2.0, 3.0, 4.0, 5.0]],
        index=pd.to_datetime(["2024-01-02"]),
        columns=list("ABCDE"),
    )
    long, short = long_short_masks(
        factor, "cross_sectional_reversal", leg_fraction=0.2,
    )
    assert bool(long.iloc[0]["A"]) and not bool(long.iloc[0]["E"])
    assert bool(short.iloc[0]["E"]) and not bool(short.iloc[0]["A"])


def test_median_reversal_splits_uneven_names():
    factor = pd.DataFrame(
        [[1.0, 2.0, 3.0, 4.0, 5.0]],
        index=pd.to_datetime(["2024-01-02"]),
        columns=list("ABCDE"),
    )
    long, short = long_short_masks(factor, "cross_sectional_reversal")
    assert int(long.iloc[0].sum()) == 2
    assert int(short.iloc[0].sum()) == 3


def test_quantile_requires_enough_names():
    factor = pd.DataFrame(
        [[1.0, 2.0, 3.0, 4.0]],
        index=pd.to_datetime(["2024-01-02"]),
        columns=list("ABCD"),
    )
    long, short = long_short_masks(
        factor, "cross_sectional_trend", leg_fraction=0.2,
    )
    assert not long.to_numpy().any()
    assert not short.to_numpy().any()
