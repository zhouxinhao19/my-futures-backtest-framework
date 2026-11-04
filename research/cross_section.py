"""截面分位多空掩码（研究/回测共用，不依赖 backtest 或 factors）。"""
from __future__ import annotations

import numpy as np
import pandas as pd


def cross_section_quantile_masks(
    factor: pd.DataFrame,
    leg_fraction: float,
    min_names: int,
) -> tuple[np.ndarray, np.ndarray]:
    """两端各取 floor(有效品种数 × leg_fraction) 个名字；腿宽不足或品种过少则当天全空。"""
    if not 0 < leg_fraction <= 0.5:
        raise ValueError("leg_fraction 必须在 (0, 0.5] 内")
    values = factor.apply(pd.to_numeric, errors="coerce")
    valid = values.notna()
    count = valid.sum(axis=1).to_numpy()
    width = np.floor(leg_fraction * count).astype(int)
    ranks = values.rank(axis=1, method="first").to_numpy()
    usable = (count >= min_names) & (width >= 1)
    short = valid.to_numpy() & (ranks <= width[:, None]) & usable[:, None]
    long = valid.to_numpy() & (ranks > (count - width)[:, None]) & usable[:, None]
    return long, short
