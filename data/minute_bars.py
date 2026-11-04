"""5 分钟 Parquet 读取与品种池对齐（与具体因子/策略无关）。"""
from __future__ import annotations

from datetime import timedelta

import pandas as pd

from data import parquet_root, pool_symbols

BAR_GAP = timedelta(minutes=5)
DAY_OPEN = 9 * 60
DAY_CLOSE = 15 * 60
DEFAULT_MINUTE_POOL = "commodity_xs"


def list_minute_products() -> list[str]:
    """读取 5 分钟 Parquet 里实际出现的品种。目录为空时直接报错。"""
    folder = parquet_root() / "minute"
    files = sorted(folder.glob("*.parquet"))
    if not files:
        raise RuntimeError(f"{folder} 没有 5 分钟行情")
    names: set[str] = set()
    for path in files:
        frame = pd.read_parquet(path, columns=["symbol"])
        names.update(frame["symbol"].astype(str).str.upper())
    products = sorted(name for name in names if name.strip())
    if not products:
        raise RuntimeError(f"{folder} 没有品种")
    return products


def resolve_universe(
    minute_products: list[str],
    pool: str = DEFAULT_MINUTE_POOL,
) -> dict[str, list[str]]:
    """品种池与 5 分钟数据的交集。没有交集时停止，不补品种。"""
    pool_names = set(pool_symbols(pool))
    minute_names = {str(symbol).upper() for symbol in minute_products}
    included = sorted(pool_names & minute_names)
    if not included:
        raise RuntimeError(f"{pool} 与 5 分钟数据没有交集")
    return {
        "included": included,
        "missing_minute_bars": sorted(pool_names - minute_names),
        "outside_pool": sorted(minute_names - pool_names),
    }


def load_minute_bars(
    products: list[str],
    start: str | None = None,
    end: str | None = None,
    *,
    lookback_days: int = 45,
) -> pd.DataFrame:
    """读取指定品种的 5 分钟行情。价格是未复权 close。

    成交量、成交额、持仓按 ``configs/data.yaml`` 的 ``volume_convention``
    把 2020-01-01 之前的双边公布数折成单边。

    传入 ``start``/``end`` 时只读对应年份文件，并在 ``trade_date`` 上裁剪
    （``start`` 前额外保留 ``lookback_days`` 日历日供日频滚动窗口）。
    """
    symbols = sorted({str(symbol).upper() for symbol in products})
    if not symbols:
        raise ValueError("5 分钟加载品种为空")
    start_ts = pd.Timestamp(start) - pd.Timedelta(days=int(lookback_days)) if start else None
    end_ts = pd.Timestamp(end) if end else None
    folder = parquet_root() / "minute"
    files = sorted(folder.glob("*.parquet"))
    if not files:
        raise RuntimeError(f"{folder} 没有 5 分钟行情；禁止回退到 future_5m 或日频数据")
    frames: list[pd.DataFrame] = []
    symbol_filter = [("symbol", "in", symbols)]
    for path in files:
        if path.stem.isdigit():
            year = int(path.stem)
            if start_ts is not None and year < start_ts.year - 1:
                continue
            if end_ts is not None and year > end_ts.year + 1:
                continue
        try:
            part = pd.read_parquet(path, filters=symbol_filter)
        except Exception:
            part = pd.read_parquet(path)
            part = part.loc[part["symbol"].astype(str).str.upper().isin(symbols)]
        if part.empty:
            continue
        if start_ts is not None or end_ts is not None:
            td = pd.to_datetime(part["trade_date"], errors="coerce").dt.normalize()
            mask = td.notna()
            if start_ts is not None:
                mask &= td.ge(start_ts)
            if end_ts is not None:
                mask &= td.le(end_ts)
            part = part.loc[mask]
        if not part.empty:
            frames.append(part)
    if not frames:
        raise RuntimeError(f"{folder} 没有 {symbols} 的 5 分钟行情；禁止回退到 future_5m 或日频数据")
    frame = pd.concat(frames, ignore_index=True)
    if frame.empty:
        raise RuntimeError(f"{folder} 没有 {symbols} 的 5 分钟行情；禁止回退到 future_5m 或日频数据")
    missing = sorted({"code", "datetime", "symbol", "close", "trade_date"} - set(frame.columns))
    if missing:
        raise ValueError(f"{folder} 缺少字段: {missing}")
    frame["symbol"] = frame["symbol"].astype(str).str.upper()
    frame["code"] = frame["code"].astype(str)
    frame["datetime"] = pd.to_datetime(frame["datetime"], errors="coerce")
    frame["trade_date"] = pd.to_datetime(frame["trade_date"], errors="coerce").dt.normalize()
    frame["close"] = pd.to_numeric(frame["close"], errors="coerce")
    frame = frame.dropna(subset=["symbol", "datetime", "trade_date", "code", "close"])
    duplicated = frame.duplicated(["symbol", "datetime"])
    if duplicated.any():
        sample = frame.loc[duplicated, ["symbol", "datetime"]].head(5)
        raise ValueError(f"{folder} 存在重复 5 分钟 K 线: {sample.to_dict('records')}")
    found = set(frame["symbol"].unique())
    absent = sorted(set(symbols) - found)
    if absent:
        raise RuntimeError(f"{folder} 缺少品种: {absent}")
    from data.minute_volume import apply_minute_single_sided

    frame = apply_minute_single_sided(frame)
    return frame.sort_values(["symbol", "trade_date", "datetime"]).reset_index(drop=True)
