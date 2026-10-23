from __future__ import annotations

import numpy as np
import pandas as pd


def permutation_pvalue(values: pd.Series, *, trials: int = 1000, seed: int = 7) -> float:
    observed = float(values.mean())
    rng = np.random.default_rng(seed)
    sample = values.dropna().to_numpy(float)
    if sample.size == 0:
        return float("nan")
    exceed = 0
    for _ in range(trials):
        if abs(float(rng.permutation(sample).mean())) >= abs(observed):
            exceed += 1
    return (exceed + 1) / (trials + 1)


def deflated_sharpe(sharpes: pd.Series, *, trials: int, annualization: int = 243) -> float:
    values = pd.to_numeric(sharpes, errors="coerce").dropna()
    if values.empty:
        return float("nan")
    penalty = np.sqrt(2.0 * np.log(max(int(trials), 1)))
    return float(values.max() - penalty / np.sqrt(annualization))
