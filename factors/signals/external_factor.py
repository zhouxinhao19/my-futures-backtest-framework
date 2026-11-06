"""外部因子值文件的周频研究适配。

因子文件是宽表：一列日期，其余每列一个品种。外部因子不进入注册表，
只能由实验脚本调用；账本 quality 随 ``protocol.book``，因子元数据默认 ``account``。
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from research.protocol import annualization, execution_price_column, forward_return


def _nav_execution_stats(nav: pd.DataFrame) -> dict[str, Any]:
    if nav.empty or "gross_return" not in nav.columns:
        return {"avg_turnover": None, "total_cost": None}
    gross = pd.to_numeric(nav["gross_return"], errors="coerce")
    net = pd.to_numeric(nav["net_return"], errors="coerce")
    turnover = pd.to_numeric(nav.get("turnover", pd.Series(dtype=float)), errors="coerce")
    return {
        "avg_turnover": float(turnover.mean()) if len(turnover) else None,
        "total_cost": float((gross - net).sum()),
    }


def load_factor_wide(
    path: str | Path,
    symbols: list[str],
    *,
    date_column: str = "TRADE_DATE",
    blank_value: float | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """读宽表因子值，返回 date / symbol / expression 长表和数据质量统计。

    列必须与品种池完全一致。``blank_value`` 为 None 时空白保留为缺失，
    否则按调用方指定的值填充，并在统计里记下填了多少格。
    """
    raw = pd.read_csv(path, dtype=str, keep_default_na=False)
    if date_column not in raw.columns:
        raise ValueError(f"因子文件缺少日期列 {date_column}: {path}")
    dates = pd.to_datetime(raw[date_column], errors="raise")
    if dates.duplicated().any():
        raise ValueError(f"因子文件日期重复: {sorted(dates[dates.duplicated()].astype(str))[:5]}")
    columns = [column for column in raw.columns if column != date_column]
    extra = sorted(set(columns) - set(symbols))
    missing = sorted(set(symbols) - set(columns))
    if extra or missing:
        raise ValueError(f"因子文件品种与品种池不一致: 多出 {extra}，缺少 {missing}")
    text = raw[columns].apply(lambda column: column.str.strip())
    blank = text.eq("")
    values = text.mask(blank).apply(pd.to_numeric, errors="coerce")
    bad = values.isna() & ~blank
    if bad.any().any():
        where = bad.stack()
        first = where[where].index[0]
        raise ValueError(f"因子文件含非数值: 第 {first[0] + 2} 行 {first[1]}")
    if not np.isfinite(values.where(~blank, 0.0).to_numpy(dtype=float)).all():
        raise ValueError("因子文件含 inf")
    non_blank = ~blank
    first_value = dates[non_blank.any(axis=1)].min()
    if blank_value is not None:
        values = values.fillna(blank_value)
    values.index = pd.DatetimeIndex(dates, name="date")
    values.columns.name = "symbol"
    long = values.stack(future_stack=True).rename("expression").reset_index()
    quality = {
        "rows": int(len(raw)),
        "symbols": int(len(columns)),
        "date_start": str(dates.min().date()),
        "date_end": str(dates.max().date()),
        "first_non_blank_date": None if pd.isna(first_value) else str(first_value.date()),
        "non_blank_cells": int(non_blank.to_numpy().sum()),
        "blank_cells": int(blank.to_numpy().sum()),
        "blank_value": blank_value,
        "first_non_blank_by_symbol": {
            str(symbol): str(dates[non_blank[symbol]].min().date())
            for symbol in columns
            if non_blank[symbol].any()
        },
    }
    return long, quality


def weekly_schedule(calendar: pd.Series | pd.DatetimeIndex) -> pd.DataFrame:
    """每个自然周最后一个交易日出信号，下一个交易日成交，持有到下一次成交。

    ``holding_days`` 是本次成交日到下一次成交日相隔的交易日数，节假日周会少于 5。
    最后一个信号没有下一次成交日，不进入回测。
    """
    days = pd.DatetimeIndex(sorted(pd.to_datetime(pd.Series(calendar)).unique()))
    frame = pd.DataFrame({"date": days, "position": np.arange(len(days))})
    frame["week"] = frame["date"].dt.to_period("W-SUN")
    signal = frame.groupby("week", sort=True).tail(1).reset_index(drop=True)
    holding = signal["position"].shift(-1) - signal["position"]
    execution = signal["position"] + 1
    out = pd.DataFrame({
        "week": signal["week"].astype(str),
        "signal_date": signal["date"],
        "execution_date": [
            days[int(i)] if i < len(days) else pd.NaT for i in execution
        ],
        "holding_days": holding,
    })
    out = out.dropna(subset=["holding_days", "execution_date"]).copy()
    out["holding_days"] = out["holding_days"].astype(int)
    return out.reset_index(drop=True)


def monthly_schedule(calendar: pd.Series | pd.DatetimeIndex) -> pd.DataFrame:
    """每个自然月最后一个交易日出信号，下一个交易日成交，持有到下一次成交。"""
    days = pd.DatetimeIndex(sorted(pd.to_datetime(pd.Series(calendar)).unique()))
    frame = pd.DataFrame({"date": days, "position": np.arange(len(days))})
    frame["month"] = frame["date"].dt.to_period("M")
    signal = frame.groupby("month", sort=True).tail(1).reset_index(drop=True)
    holding = signal["position"].shift(-1) - signal["position"]
    execution = signal["position"] + 1
    out = pd.DataFrame({
        "month": signal["month"].astype(str),
        "signal_date": signal["date"],
        "execution_date": [
            days[int(i)] if i < len(days) else pd.NaT for i in execution
        ],
        "holding_days": holding,
    })
    out = out.dropna(subset=["holding_days", "execution_date"]).copy()
    out["holding_days"] = out["holding_days"].astype(int)
    if (out["holding_days"] < 1).any():
        raise RuntimeError("月频持有天数必须为正")
    return out.reset_index(drop=True)


def weekly_forward_returns(panel: pd.DataFrame, schedule: pd.DataFrame) -> pd.DataFrame:
    """信号日 T 到下一次成交日的收益：``adj_settle[T+1+H] / adj_settle[T+1] - 1``。

    先把每个品种补齐到完整交易日历，缺价日留空，保证移位按交易日而不是按行。
    """
    price = execution_price_column()
    required = {"date", "symbol", price}
    missing = sorted(required - set(panel.columns))
    if missing:
        raise ValueError(f"周频收益缺少字段: {missing}")
    calendar = pd.DatetimeIndex(sorted(pd.to_datetime(panel["date"]).unique()))
    wide = (
        panel.assign(date=pd.to_datetime(panel["date"]), symbol=panel["symbol"].astype(str))
        .pivot_table(index="date", columns="symbol", values=price, aggfunc="last")
        .reindex(calendar)
    )
    wide.index.name = "date"
    wide.columns.name = "symbol"
    grid = (
        wide.stack(future_stack=True).rename(price).reset_index()
        .sort_values(["symbol", "date"]).reset_index(drop=True)
    )
    signal_rows = grid.merge(
        schedule[["signal_date", "holding_days"]].rename(columns={"signal_date": "date"}),
        on="date",
        how="inner",
    )
    parts: list[pd.DataFrame] = []
    for horizon in sorted(schedule["holding_days"].unique()):
        returns = forward_return(grid, holding_days=int(horizon))
        keyed = grid[["date", "symbol"]].assign(future_return=returns)
        rows = signal_rows.loc[signal_rows["holding_days"] == horizon, ["date", "symbol", "holding_days"]]
        parts.append(rows.merge(keyed, on=["date", "symbol"], how="left"))
    return pd.concat(parts, ignore_index=True).sort_values(["date", "symbol"]).reset_index(drop=True)


def weekly_target_weights(
    frame: pd.DataFrame,
    *,
    pipeline: str,
    direction: str,
    leg_fraction: float,
) -> pd.DataFrame:
    """每个信号周的目标资金权重，索引为成交日。

    ``frame`` 需含 ``execution_date``、``symbol``、``factor_value``。权重规则与
    价格口径一致：时序按因子符号等权，截面多空两端各占 50%。
    """
    sign = -1.0 if direction == "reversal" else 1.0
    rows: list[dict[str, float]] = []
    index: list[pd.Timestamp] = []
    for execution, group in frame.groupby("execution_date", sort=True):
        if pipeline == "cs_ls":
            side = max(1, math.ceil(len(group) * leg_fraction))
            if len(group) < 2 * side:
                continue
            ranked = group.assign(score=group["factor_value"] * sign).sort_values(["score", "symbol"])
            weights = {str(s): 0.0 for s in ranked["symbol"]}
            for s in ranked.tail(side)["symbol"]:
                weights[str(s)] = 0.5 / side
            for s in ranked.head(side)["symbol"]:
                weights[str(s)] = -0.5 / side
        else:
            signal = np.sign(group["factor_value"]) * sign
            valid = group.loc[signal != 0]
            if valid.empty:
                continue
            weights = {
                str(symbol): float(signal.loc[i]) / len(valid)
                for i, symbol in zip(valid.index, valid["symbol"])
            }
        rows.append(weights)
        index.append(pd.Timestamp(execution))
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows, index=index).fillna(0.0)


def weekly_nav_stats(nav: pd.DataFrame, frequency: str = "weekly") -> dict[str, Any]:
    """净值统计。价格口径为周频净值，年化次数取 ``weekly_annualization``；
    手数口径输出每日净值，传入 ``frequency="daily"``，年化取 ``daily_annualization``。"""
    r = pd.to_numeric(nav.get("net_return", pd.Series(dtype=float)), errors="coerce").dropna()
    if len(r) < 2:
        return {
            "periods": int(len(r)),
            "annual_return": None,
            "annual_vol": None,
            "sharpe": None,
            "max_drawdown": None,
            "cum_return": None,
            "win_rate": None,
            **_nav_execution_stats(nav),
        }
    ann = annualization(frequency)
    curve = (1.0 + r).cumprod()
    annual_return = float(curve.iloc[-1] ** (ann / len(r)) - 1.0)
    vol = float(r.std(ddof=1) * np.sqrt(ann))
    return {
        "periods": int(len(r)),
        "annual_return": annual_return,
        "annual_vol": vol,
        "sharpe": annual_return / vol if vol else None,
        "max_drawdown": float((curve / curve.cummax() - 1.0).min()),
        "cum_return": float(curve.iloc[-1] - 1.0),
        "win_rate": float((r > 0).mean()),
        **_nav_execution_stats(nav),
    }
