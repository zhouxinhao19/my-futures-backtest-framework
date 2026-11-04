"""因子宽表 → 多空布尔掩码（正式回测链路，不含手数与盈亏）。"""

from __future__ import annotations

import numpy as np
import pandas as pd

from research.cross_section import cross_section_quantile_masks

MODES = (
    "time_series_trend",
    "time_series_reversal",
    "cross_sectional_trend",
    "cross_sectional_reversal",
)


def long_short_masks(
    factor: pd.DataFrame,
    mode: str,
    *,
    leg_fraction: float | None = None,
    min_names: int = 5,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """日期 × 品种因子表 → (long, short)。

    时序：因子大于 0 做多、小于 0 做空，0 和空值不交易。
    截面：``leg_fraction`` 为空时按当日中位数分成两腿；给出比例时各取两端
    ``floor(有效品种数 × leg_fraction)`` 个名字。有效品种少于 ``min_names``
    或腿宽不足 1 时，当天两腿都空。趋势做多高值，名字以 reversal 结尾时反过来。
    """
    if mode not in MODES:
        raise ValueError(f"未知策略: {mode}")
    values = factor.apply(pd.to_numeric, errors="coerce")
    if mode.startswith("time_series"):
        long = np.sign(values).gt(0)
        short = np.sign(values).lt(0)
    elif leg_fraction is None:
        rank = values.rank(axis=1, pct=True)
        long = rank.ge(0.5) & values.notna()
        short = rank.lt(0.5) & values.notna()
    else:
        long, short = _tail_legs(values, leg_fraction, min_names)
    if mode.endswith("reversal"):
        long, short = short, long
    return long.fillna(False), short.fillna(False)


def _tail_legs(
    values: pd.DataFrame,
    leg_fraction: float,
    min_names: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    long, short = cross_section_quantile_masks(values, leg_fraction, min_names)
    return (
        pd.DataFrame(long, index=values.index, columns=values.columns),
        pd.DataFrame(short, index=values.index, columns=values.columns),
    )
