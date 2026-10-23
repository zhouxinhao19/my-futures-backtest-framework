from __future__ import annotations

import numpy as np
import pandas as pd


def performance_metrics(equity: pd.Series, returns: pd.Series, annualization: int = 243) -> dict[str, float]:
    values = pd.to_numeric(returns, errors="coerce").dropna()
    if values.empty:
        return {"annual_return": np.nan, "volatility": np.nan, "sharpe": np.nan, "max_drawdown": np.nan}
    curve = (1.0 + values).cumprod()
    drawdown = curve / curve.cummax() - 1.0
    volatility = float(values.std(ddof=1) * np.sqrt(annualization))
    annual_return = float((curve.iloc[-1] ** (annualization / len(values))) - 1.0)
    return {
        "annual_return": annual_return,
        "volatility": volatility,
        "sharpe": float(values.mean() / values.std(ddof=1) * np.sqrt(annualization)) if values.std(ddof=1) else np.nan,
        "max_drawdown": float(drawdown.min()),
        "ending_equity": float(equity.iloc[-1]) if len(equity) else np.nan,
        "calmar": float(annual_return / abs(float(drawdown.min()))) if drawdown.min() else np.nan,
        "win_rate": float((values > 0).mean()),
    }


def information_coefficient(frame: pd.DataFrame, factor: str = "factor", forward: str = "forward") -> pd.Series:
    """按日 Spearman IC。"""
    rows = []
    for date, group in frame.dropna(subset=[factor, forward]).groupby("date"):
        if len(group) < 3 or group[factor].nunique() < 2:
            continue
        rows.append((pd.Timestamp(date), float(group[factor].corr(group[forward], method="spearman"))))
    return pd.Series(dict(rows), dtype=float)


def ic_summary(ic: pd.Series) -> dict[str, float]:
    clean = pd.to_numeric(ic, errors="coerce").dropna()
    if clean.empty:
        return {"ic": np.nan, "icir": np.nan}
    deviation = float(clean.std(ddof=1))
    return {
        "ic": float(clean.mean()),
        "icir": float(clean.mean() / deviation) if deviation else np.nan,
    }


def quantile_mean_return(frame: pd.DataFrame, factor: str = "factor", forward: str = "forward", quantiles: int = 5) -> pd.Series:
    rows = []
    for _, group in frame.dropna(subset=[factor, forward]).groupby("date"):
        if group[factor].nunique() < 2:
            continue
        group = group.copy()
        group["bucket"] = pd.qcut(group[factor].rank(method="first"), min(quantiles, group[factor].nunique()), labels=False, duplicates="drop")
        rows.append(group.groupby("bucket")[forward].mean())
    if not rows:
        return pd.Series(dtype=float)
    return pd.concat(rows, axis=1).mean(axis=1)


def symbol_contribution(pnl: pd.DataFrame) -> pd.Series:
    numeric = pnl.apply(pd.to_numeric, errors="coerce")
    return numeric.sum(axis=0).sort_values(ascending=False)


def sample_performance(
    returns: pd.Series,
    dates: pd.Series,
    oos_start: str | None = None,
) -> dict[str, dict[str, float]]:
    from research.protocol import oos_start as protocol_oos_start

    boundary = pd.Timestamp(oos_start or protocol_oos_start())
    stamp = pd.to_datetime(dates)
    full = performance_metrics(pd.Series(dtype=float), returns)
    inside = returns.loc[stamp < boundary]
    outside = returns.loc[stamp >= boundary]
    return {
        "full": full,
        "is": performance_metrics(pd.Series(dtype=float), inside),
        "oos": performance_metrics(pd.Series(dtype=float), outside),
    }
