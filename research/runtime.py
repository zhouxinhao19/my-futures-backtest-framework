from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class ExecutionConfig:
    initial_cash: float = 10_000_000.0
    slippage_ticks: int = 1
    slippage_pct_floor: float = 0.0002
    slippage_pct_ceiling: float = 0.01
    allow_limit_fill: bool = False
    participation_rate: float = 0.10
    margin_cap: float = 0.70
    max_symbol_notional: float = 0.35
    delivery_days_before: int = 5
    enforce_delivery_roll: bool = True
    mark_price: str = "settle"
    cash_yield: float = 0.0
    borrow_yield: float = 0.0
    signal_lag: int = 1


def _read_yaml(name: str) -> dict[str, Any]:
    path = ROOT / "configs" / name
    if not path.exists():
        return {}
    with path.open(encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def load_runtime_config() -> tuple[ExecutionConfig, dict[str, Any]]:
    backtest = _read_yaml("backtest.yaml")
    costs = _read_yaml("costs.yaml")
    universe = _read_yaml("universe.yaml")
    merged = {**backtest, **costs}
    participation = float(merged.get("participation_rate", 0.10))
    if not 0 < participation <= 1:
        raise ValueError("participation_rate 必须在 (0, 1] 内")
    execution = ExecutionConfig(
        initial_cash=float(merged.get("initial_cash", 10_000_000)),
        slippage_ticks=int(merged.get("slippage_ticks", 1)),
        slippage_pct_floor=float(merged.get("slippage_pct_floor", 0.0002)),
        slippage_pct_ceiling=float(merged.get("slippage_pct_ceiling", 0.01)),
        allow_limit_fill=bool(merged.get("allow_limit_fill", False)),
        participation_rate=participation,
        margin_cap=float(merged.get("margin_cap", 0.70)),
        max_symbol_notional=float(merged.get("max_symbol_notional", 0.35)),
        delivery_days_before=int(merged.get("delivery_days_before", 5)),
        enforce_delivery_roll=bool(merged.get("enforce_delivery_roll", True)),
        mark_price=str(merged.get("mark_price", "settle")),
        cash_yield=float(merged.get("cash_yield", 0.0)),
        borrow_yield=float(merged.get("borrow_yield", 0.0)),
        signal_lag=int(merged.get("signal_lag", 1)),
    )
    if execution.margin_cap <= 0 or execution.margin_cap > 1:
        raise ValueError("margin_cap 必须在 (0, 1] 内")
    if execution.signal_lag < 1:
        raise ValueError("signal_lag 必须至少为 1")
    return execution, universe
