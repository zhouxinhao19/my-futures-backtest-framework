from __future__ import annotations

import pandas as pd


def split_holdout(
    frame: pd.DataFrame,
    *,
    date_column: str = "date",
    oos_start: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    dates = pd.to_datetime(frame[date_column])
    boundary = pd.Timestamp(oos_start)
    return frame.loc[dates < boundary].copy(), frame.loc[dates >= boundary].copy()
