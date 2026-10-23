"""一次研究回测的标准落盘。不覆盖已有目录。

固定文件：
- run.json：含 sample_split，样本外起点只来自协议
- daily.csv：逐日盈亏、成本、权益和收益率
- trades.csv：逐笔成交
- positions.csv：逐品种逐日持仓
- summary.csv：全样本、样本内、样本外三段业绩
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from research.metrics import performance_metrics
from research.protocol import annualization, canonical_sample_split, oos_start

from .ledger import ResearchResult

SAMPLE_NAMES = ("全样本", "样本内", "样本外")


def write_research_run(output_dir: str | Path, metadata: dict[str, Any], result: ResearchResult) -> Path:
    daily = _frame(result.daily)
    payload = dict(metadata)
    payload["sample_split"] = _split_with_dates(payload.get("sample_split"), daily)
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=False)
    payload["initial_equity"] = payload.get("initial_equity", 100_000_000)
    payload["compound"] = bool(payload.get("compound", True))
    payload["price_field"] = "adj_settle"
    payload["framework"] = "backtest"
    summary = sample_summary(daily)
    payload["samples"] = summary.to_dict(orient="records")
    (directory / "run.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    daily.to_csv(directory / "daily.csv", index=False, encoding="utf-8-sig")
    _frame(result.trades).to_csv(directory / "trades.csv", index=False, encoding="utf-8-sig")
    _frame(result.positions).to_csv(directory / "positions.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(directory / "summary.csv", index=False, encoding="utf-8-sig")
    return directory


def sample_summary(daily: pd.DataFrame) -> pd.DataFrame:
    """按协议起点切分逐日权益。该日及之后为样本外。"""
    frame = daily.copy()
    if "date" in frame.columns:
        frame["_date"] = pd.to_datetime(frame["date"])
    elif isinstance(frame.index, pd.DatetimeIndex):
        frame["_date"] = pd.to_datetime(frame.index)
    else:
        raise ValueError("逐日记录缺少日期，不能切分样本内外")
    boundary = pd.Timestamp(oos_start())
    slices = {
        "全样本": frame,
        "样本内": frame.loc[frame["_date"] < boundary],
        "样本外": frame.loc[frame["_date"] >= boundary],
    }
    rows = []
    for name, part in slices.items():
        equity = pd.to_numeric(part.get("equity_after"), errors="coerce")
        returns = pd.to_numeric(part.get("ret"), errors="coerce")
        metrics = performance_metrics(equity, returns, annualization())
        if "ending_equity" not in metrics:
            metrics["ending_equity"] = float(equity.iloc[-1]) if len(equity) else float("nan")
            metrics["calmar"] = float("nan")
            metrics["win_rate"] = float("nan")
        rows.append({
            "sample": name,
            "oos_start": oos_start(),
            "start": None if part.empty else str(part["_date"].min().date()),
            "end": None if part.empty else str(part["_date"].max().date()),
            **metrics,
        })
    return pd.DataFrame(rows)


def _split_with_dates(split: dict[str, Any] | None, daily: pd.DataFrame) -> dict[str, Any]:
    payload = canonical_sample_split(split)
    if daily.empty or "date" not in daily.columns:
        return payload
    dates = pd.to_datetime(daily["date"])
    payload.setdefault("start", str(dates.min().date()))
    payload.setdefault("end", str(dates.max().date()))
    return payload


def _frame(table: pd.DataFrame) -> pd.DataFrame:
    out = table.copy()
    if isinstance(out.index, pd.DatetimeIndex) and "date" not in out.columns:
        out.insert(0, "date", out.index)
    return out.reset_index(drop=True)
