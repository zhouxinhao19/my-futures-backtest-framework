from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any

import pandas as pd

from research.runtime import ExecutionConfig


@dataclass
class BenchmarkResult:
    nav: pd.DataFrame
    trades: pd.DataFrame
    positions: pd.DataFrame
    checks: pd.DataFrame


def _number(value: Any, default: float = 0.0) -> float:
    result = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    return default if pd.isna(result) else float(result)


def _bar_value(bar: Any, name: str, default: Any = None) -> Any:
    if isinstance(bar, pd.Series):
        return bar.get(name, default)
    return getattr(bar, name, default)


def _slippage(price: float, spec: dict[str, Any], execution: ExecutionConfig) -> float:
    tick = _number(spec.get("tick_size"))
    adverse = max(tick * execution.slippage_ticks, abs(price) * execution.slippage_pct_floor)
    return min(adverse, abs(price) * execution.slippage_pct_ceiling)


def _expiry(code: str) -> int:
    match = re.search(r"(\d{3,6})$", str(code).lower())
    return int(match.group(1)) if match else 10**9


class _Ledger:
    """最小期货账本：现金记录逐日盯市损益，持仓记录合约手数。"""

    def __init__(self, initial_cash: float):
        self.cash = float(initial_cash)
        self.positions: dict[str, int] = {}
        self.marks: dict[str, float] = {}
        self.trades: list[dict[str, Any]] = []

    def trade(
        self,
        date: pd.Timestamp,
        code: str,
        target: int,
        price: float,
        settle: float,
        spec: dict[str, Any],
        execution: ExecutionConfig,
        reason: str,
        volume: float | None = None,
        reference: float | None = None,
    ) -> int:
        old = int(self.positions.get(code, 0))
        delta = int(target) - old
        if not delta:
            return old
        if volume is not None:
            fill_limit = int(max(_number(volume) * execution.participation_rate, 0))
            delta = (1 if delta > 0 else -1) * min(abs(delta), fill_limit)
        if not delta:
            return old

        multiplier = _number(spec["multiplier"])
        commission = abs(delta) * _number(spec.get("commission_per_lot"))
        ref = price if reference is None else reference
        slip = _slippage(price, spec, execution)
        side = 1 if delta > 0 else -1
        execution_price = price + side * slip

        # 先将既有持仓从上次标记价结算到本次成交价。
        if old:
            self.cash += (execution_price - self.marks[code]) * old * multiplier
        new = old + delta
        self.cash -= commission
        self.cash -= abs(execution_price - ref) * abs(delta) * multiplier
        if new:
            self.positions[code] = new
            self.marks[code] = execution_price
        else:
            self.positions.pop(code, None)
            self.marks.pop(code, None)
        self.trades.append({
            "date": date,
            "code": code,
            "lots": delta,
            "target_lots": new,
            "reference_price": price,
            "execution_price": execution_price,
            "commission": commission,
            "slippage_cost": abs(execution_price - ref) * abs(delta) * multiplier,
            "reason": reason,
        })
        return new

    def mark(
        self,
        date: pd.Timestamp,
        bars: dict[str, pd.Series],
        specs: dict[str, dict[str, Any]],
        margin_mode: str = "sum",
    ) -> tuple[float, float]:
        margin = 0.0
        for code, lots in list(self.positions.items()):
            bar = bars.get(code)
            if bar is None:
                continue
            settle = _number(_bar_value(bar, "settle", _bar_value(bar, "close")))
            multiplier = _number(specs[code]["multiplier"])
            self.cash += (settle - self.marks[code]) * lots * multiplier
            self.marks[code] = settle
            margin_value = abs(settle * lots * multiplier * _number(specs[code]["margin_rate"]))
            if margin_mode == "max":
                margin = max(margin, margin_value)
            else:
                margin += margin_value
        return self.cash, margin


def _normalise_frame(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    result["date"] = pd.to_datetime(result["date"])
    result["code"] = result["code"].astype(str).str.lower()
    if "volume" not in result:
        for alias in ("vol", "trade"):
            if alias in result:
                result["volume"] = result[alias]
                break
    if "volume" not in result:
        result["volume"] = 0
    return result.sort_values(["date", "code"]).reset_index(drop=True)


def _result(ledger: _Ledger, nav_rows: list[dict[str, Any]], position_rows: list[dict[str, Any]],
            checks: list[dict[str, Any]]) -> BenchmarkResult:
    nav = pd.DataFrame(nav_rows)
    if not nav.empty:
        nav["return"] = nav["equity"].pct_change().fillna(0.0)
        nav["peak"] = nav["equity"].cummax()
        nav["drawdown"] = nav["equity"] / nav["peak"] - 1.0
    return BenchmarkResult(
        nav=nav,
        trades=pd.DataFrame(ledger.trades),
        positions=pd.DataFrame(position_rows),
        checks=pd.DataFrame(checks),
    )


def buy_and_hold(
    frame: pd.DataFrame,
    main_map: pd.DataFrame,
    spec: dict[str, Any],
    execution: ExecutionConfig,
    lots: int = 1,
) -> BenchmarkResult:
    """主力连续合约买入持有，换月按旧平新开处理。"""
    frame = _normalise_frame(frame)
    mapping = main_map.copy()
    mapping["date"] = pd.to_datetime(mapping["date"] if "date" in mapping else mapping.index)
    mapping["contract"] = mapping["contract"].astype(str).str.lower()
    ledger = _Ledger(execution.initial_cash)
    nav_rows: list[dict[str, Any]] = []
    position_rows: list[dict[str, Any]] = []
    checks: list[dict[str, Any]] = []
    previous: str | None = None

    for date, day in frame.groupby("date", sort=True):
        row = mapping[mapping["date"] == date]
        if row.empty:
            checks.append({"date": date, "check": "main_mapping", "status": "missing"})
            continue
        contract = str(row.iloc[0]["contract"]).lower()
        bars = {str(x.code): x for x in day.itertuples(index=False)}
        if contract not in bars:
            checks.append({"date": date, "check": "main_bar", "status": "missing", "code": contract})
            continue
        bar = bars[contract]
        if previous is not None and previous != contract:
            if previous not in bars:
                checks.append({"date": date, "check": "roll_close", "status": "missing", "code": previous})
                continue
            ledger.trade(date, previous, 0, _number(getattr(bars[previous], "open")),
                         _number(getattr(bars[previous], "settle")), spec, execution, "roll_close",
                         getattr(bars[previous], "volume", None))
            ledger.trade(date, contract, lots, _number(getattr(bar, "open")),
                         _number(getattr(bar, "settle")), spec, execution, "roll_open",
                         getattr(bar, "volume", None))
        elif previous is None:
            ledger.trade(date, contract, lots, _number(getattr(bar, "open")),
                         _number(getattr(bar, "settle")), spec, execution, "initial_open",
                         getattr(bar, "volume", None))
        previous = contract
        equity, margin = ledger.mark(date, bars, {contract: spec})
        nav_rows.append({
            "date": date, "equity": equity, "cash": ledger.cash, "margin": margin,
            "contract": contract, "main_adj_close": _number(row.iloc[0].get("adj_close")),
            "settle": _number(getattr(bar, "settle")),
        })
        position_rows.extend({"date": date, "code": code, "lots": qty}
                              for code, qty in ledger.positions.items())
    return _result(ledger, nav_rows, position_rows, checks)


def daily_open_close(
    frame: pd.DataFrame,
    spec: dict[str, Any],
    execution: ExecutionConfig,
    lots: int = 1,
) -> BenchmarkResult:
    """每日开盘开仓、收盘平仓的固定手数基准。"""
    frame = _normalise_frame(frame)
    ledger = _Ledger(execution.initial_cash)
    nav_rows: list[dict[str, Any]] = []
    checks: list[dict[str, Any]] = []
    for date, day in frame.groupby("date", sort=True):
        if "main_contract" in day and (day["code"] == day["main_contract"]).any():
            day = day[day["code"] == day["main_contract"]]
        row = day.sort_values("volume", ascending=False).iloc[0]
        code = str(row["code"])
        bars = {code: row}
        ledger.trade(date, code, lots, _number(row["open"]), _number(row["settle"]),
                     spec, execution, "daily_open", row.get("volume"))
        ledger.trade(date, code, 0, _number(row["close"]), _number(row["settle"]),
                     spec, execution, "daily_close", row.get("volume"))
        equity, margin = ledger.mark(date, bars, {code: spec})
        gross = (_number(row["close"]) - _number(row["open"])) * lots * _number(spec["multiplier"])
        nav_rows.append({
            "date": date, "equity": equity, "cash": ledger.cash, "margin": margin,
            "code": code, "gross_pnl": gross, "expected_fee": 2 * lots * _number(spec.get("commission_per_lot")),
        })
    return _result(ledger, nav_rows, [], checks=[])


def _spread_legs(day: pd.DataFrame) -> tuple[str, str] | None:
    candidates = day.copy()
    candidates["volume_num"] = pd.to_numeric(candidates.get("volume"), errors="coerce").fillna(0)
    candidates = candidates[candidates["volume_num"] > 0].sort_values(["code"])
    if len(candidates) < 2:
        return None
    # 到期月决定近远，成交量仅用于过滤无流动性合约。
    codes = sorted(candidates["code"].astype(str), key=_expiry)
    return codes[0], codes[1]


def calendar_spread(
    frame: pd.DataFrame,
    spec: dict[str, Any],
    execution: ExecutionConfig,
    lots: int = 1,
    margin_mode: str = "max",
) -> BenchmarkResult:
    """买近卖远跨期价差，默认按单边最大保证金计提。"""
    frame = _normalise_frame(frame)
    ledger = _Ledger(execution.initial_cash)
    nav_rows: list[dict[str, Any]] = []
    checks: list[dict[str, Any]] = []
    previous: tuple[str, str] | None = None
    previous_spread: float | None = None

    for date, day in frame.groupby("date", sort=True):
        legs = _spread_legs(day)
        if legs is None:
            checks.append({"date": date, "check": "spread_legs", "status": "insufficient_liquidity"})
            continue
        bars = {str(x["code"]): x for _, x in day.iterrows()}
        if previous and previous != legs:
            for code in previous:
                if code in bars:
                    ledger.trade(date, code, 0, _number(bars[code]["open"]),
                                  _number(bars[code]["settle"]), spec, execution, "spread_roll_close",
                                  bars[code].get("volume"))
        for code, target in ((legs[0], lots), (legs[1], -lots)):
            if code not in bars:
                continue
            ledger.trade(date, code, target, _number(bars[code]["open"]),
                         _number(bars[code]["settle"]), spec, execution,
                         "spread_open" if previous is None else "spread_roll_open",
                         bars[code].get("volume"))
        same_pair = previous == legs
        previous = legs
        equity, margin = ledger.mark(date, bars, {code: spec for code in legs}, margin_mode)
        near, far = bars[legs[0]], bars[legs[1]]
        spread = _number(near["settle"]) - _number(far["settle"])
        spread_pnl = 0.0 if previous_spread is None or not same_pair else (
            spread - previous_spread
        ) * lots * _number(spec["multiplier"])
        previous_spread = spread
        nav_rows.append({
            "date": date, "equity": equity, "cash": ledger.cash, "margin": margin,
            "near_code": legs[0], "far_code": legs[1], "spread": spread,
            "gross_spread_pnl": spread_pnl,
            "margin_mode": margin_mode,
        })
    return _result(ledger, nav_rows, [], checks)
