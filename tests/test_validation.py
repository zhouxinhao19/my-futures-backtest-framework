import pandas as pd

from validation.holdout import split_holdout
from validation.statistics import deflated_sharpe, permutation_pvalue


def test_holdout_is_point_in_time():
    frame = pd.DataFrame({"date": pd.date_range("2024-01-01", periods=4), "value": [1, 2, 3, 4]})
    train, test = split_holdout(frame, oos_start="2024-01-03")
    assert train["value"].tolist() == [1, 2]
    assert test["value"].tolist() == [3, 4]


def test_validation_statistics_are_finite():
    values = pd.Series([0.1, -0.02, 0.03, 0.04])
    assert 0 < permutation_pvalue(values, trials=20) <= 1
    assert deflated_sharpe(pd.Series([0.5, 0.8]), trials=10) < 0.8
