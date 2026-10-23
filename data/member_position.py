"""会员持仓与注册仓单宽表的清洗。

源文件结构：第 1 行是 Wind 代码，第 2 行是交易所与中文名，第 3 行起第 1 列是交易日。
输出与 ``price_data_research`` 使用同一套 ``symbol + date`` 键，日期字符串与 date_stamp 口径相同。
每个字段独立清洗，净持仓不由多头减空头推算。
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import pandas as pd

FIELDS = ("top20_long", "top20_short", "net_position", "registered_receipt")
RECEIPT_UNIT = "exchange_published"
EXCHANGE_MAP = {
    "SHF": "SHFE",
    "CZC": "CZCE",
    "GFE": "GFEX",
    "DCE": "DCE",
    "INE": "INE",
}
EPOCH = pd.Timestamp("1970-01-01")
_NAME_PREFIX = re.compile(r"^(SHFE|CZCE|DCE|INE|GFEX)\s*")


def mask_leading_zeros(series: pd.Series) -> pd.Series:
    """首次非零之前的 0 和空值置为空；首次非零之后的 0 保留。

    全部为 0 的序列整体置空，表示该品种在数据窗口内没有可用持仓。
    """
    values = pd.to_numeric(series, errors="raise").astype("float64")
    nonzero = values.index[values.notna() & (values != 0)]
    if len(nonzero) == 0:
        return values.where(values.isna())
    return values.where(values.index >= nonzero.min())


def read_wide_sheet(path: Path) -> tuple[list[str], dict[str, str], pd.DataFrame]:
    raw = pd.read_excel(path, header=None)
    codes = [str(x).strip() for x in raw.iloc[0, 1:]]
    names = ["" if pd.isna(x) else str(x).strip() for x in raw.iloc[1, 1:]]
    if len(set(codes)) != len(codes):
        raise ValueError(f"{path.name} 存在重复代码")
    dates = pd.to_datetime(raw.iloc[2:, 0], errors="raise").dt.normalize()
    if dates.isna().any() or dates.duplicated().any() or not dates.is_monotonic_increasing:
        raise ValueError(f"{path.name} 日期列为空、重复或未升序")
    data = raw.iloc[2:, 1:].apply(lambda s: pd.to_numeric(s, errors="raise")).astype("float64")
    data.columns = codes
    data.index = pd.DatetimeIndex(dates.to_numpy(), name="date")
    return codes, dict(zip(codes, names)), data


def build_member_frame(paths: dict[str, Path]) -> tuple[pd.DataFrame, int]:
    """返回长表（每行一个 symbol + date）和被丢弃的全空行数。"""
    if set(paths) != set(FIELDS):
        raise ValueError(f"需要且仅需要这些字段: {FIELDS}")
    sheets = {field: read_wide_sheet(path) for field, path in paths.items()}
    codes, names, master = sheets["net_position"]
    master_index = master.index

    aligned: dict[str, pd.DataFrame] = {}
    for field, (sheet_codes, sheet_names, frame) in sheets.items():
        if sheet_codes != codes:
            raise ValueError(f"{field} 的代码列与净持仓不一致")
        if sheet_names != names:
            raise ValueError(f"{field} 的品种名与净持仓不一致")
        extra = frame.index.difference(master_index)
        if len(extra):
            raise ValueError(f"{field} 含有净持仓没有的日期: {extra[:5].tolist()}")
        aligned[field] = frame.reindex(index=master_index, columns=codes).apply(mask_leading_zeros)

    parts = []
    seen: set[str] = set()
    for code in codes:
        base, _, suffix = code.rpartition(".")
        if suffix not in EXCHANGE_MAP:
            raise ValueError(f"未知交易所后缀: {code}")
        symbol = base.upper()
        if symbol in seen:
            raise ValueError(f"品种重复: {symbol}")
        seen.add(symbol)
        name = _NAME_PREFIX.sub("", names[code]).strip()
        if not name:
            raise ValueError(f"{code} 的中文名为空")
        part = pd.DataFrame(
            {field: aligned[field][code].to_numpy() for field in FIELDS},
            index=master_index,
        )
        part.insert(0, "symbol", symbol)
        part.insert(1, "wind_code", code)
        part.insert(2, "exchange", EXCHANGE_MAP[suffix])
        part.insert(3, "name", name)
        parts.append(part)

    out = pd.concat(parts).reset_index()
    all_null = out[list(FIELDS)].isna().all(axis=1)
    dropped = int(all_null.sum())
    return out.loc[~all_null].reset_index(drop=True), dropped


def _as_int(value: float, field: str, symbol: str, date: Any) -> int:
    if not float(value).is_integer():
        raise ValueError(f"{symbol} {date} {field} 不是整数: {value}")
    return int(value)


def to_documents(frame: pd.DataFrame, source: str, updated_at: Any) -> list[dict[str, Any]]:
    docs: list[dict[str, Any]] = []
    for row in frame.itertuples(index=False):
        date = pd.Timestamp(row.date)
        doc: dict[str, Any] = {
            "contract_object": row.symbol,
            "symbol": row.symbol,
            "date": date.strftime("%Y-%m-%d"),
            "date_stamp": int((date - EPOCH).total_seconds()),
            "wind_code": row.wind_code,
            "exchange": row.exchange,
            "name": row.name,
            "receipt_unit": RECEIPT_UNIT,
            "source": source,
            "updated_at": updated_at,
        }
        for field in FIELDS:
            value = getattr(row, field)
            doc[field] = None if pd.isna(value) else _as_int(value, field, row.symbol, doc["date"])
        docs.append(doc)
    return docs
