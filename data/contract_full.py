"""全合约日行情（load_data1.csv 口径）的纯函数。

- 合约编号 ``contract_id``：统一为 ``产品 + YY + MM``（如 ``SR2401``）。
  郑商所等 3 位数字代码（如 ``sr401``）用最后交易日推断年代，不做猜测：
  候选年份必须唯一，否则报错。
- 供应商主力：CSV 的 ``MAINCON`` / ``SMAINCON``。
- 研究主力标签 ``research_main``：与 ``price_data_research`` 同一品种同一天选定的合约比较。
  研究表无记录、或其合约在 CSV 当天找不到时为 ``None``，不记为 0。
- 空值一律保留为 ``None``，不做任何填充。
"""
from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Any

import numpy as np
import pandas as pd

EPOCH = pd.Timestamp("1970-01-01")
KEY = ["contract_object", "date"]
VENDOR_RULE = "vendor_maincon"
VENDOR_RULE_NOTE = (
    "CSV MAINCON 约等于前一交易日持仓量最大的合约（复现率 97.6%，"
    "见 reports/main_rule_inference/summary.json）；不等于 price_data_research 的主力选择。"
)
RESEARCH_SOURCE = "price_data_research"

CSV_REQUIRED = (
    "SECURITY_ID", "TICKER_SYMBOL", "EXCHANGE_CD", "SEC_SHORT_NAME", "TRADE_DATE",
    "CONTRACT_OBJECT", "CONTRACT_MARK", "PRE_SETTL_PRICE", "PRE_CLOSE_PRICE",
    "OPEN_PRICE", "HIGHEST_PRICE", "LOWEST_PRICE", "SETTL_PRICE", "CLOSE_PRICE",
    "TURNOVER_VOL", "TURNOVER_VALUE", "OPEN_INT", "CHG", "CHG1", "CHG_PCT",
    "MAINCON", "SMAINCON", "LAST_TRADE_DATE", "UPDATE_TIME", "TMSTAMP",
)
_NUMERIC_CSV = (
    "PRE_SETTL_PRICE", "PRE_CLOSE_PRICE", "OPEN_PRICE", "HIGHEST_PRICE", "LOWEST_PRICE",
    "SETTL_PRICE", "CLOSE_PRICE", "TURNOVER_VOL", "TURNOVER_VALUE", "OPEN_INT",
    "CHG", "CHG1", "CHG_PCT", "MAINCON", "SMAINCON", "TMSTAMP", "SECURITY_ID",
)
FULL_REQUIRED_FIELDS = (
    "code", "contract_id", "contract_object", "date", "date_stamp", "open", "high", "low",
    "close", "settle", "volume", "open_int", "vendor_main", "vendor_second_main",
    "research_main", "no_trade", "ohlc_missing", "source", "source_sha256", "updated_at",
)
MAIN_REQUIRED_FIELDS = (
    "contract_object", "date", "date_stamp", "main_contract", "main_contract_id",
    "research_main_contract_id", "main_agree", "rule", "source", "updated_at",
)

_CODE = re.compile(r"^([a-z]+)(\d{3}|\d{4})$")
def allowed_prefixes(product: str, trade_date: Any, aliases: Mapping[str, list[Mapping[str, Any]]] | None) -> set[str]:
    """某品种在某交易日允许的代码前缀：规范前缀（即品种名小写）始终允许；
    历史别名只在 ``valid_until`` 当日及之前允许。"""
    key = str(product).strip().upper()
    allowed = {key.lower()}
    if not aliases or key not in aliases:
        return allowed
    if trade_date is None:
        raise ValueError(f"{key} 存在代码别名，解析必须提供交易日")
    day = pd.Timestamp(trade_date)
    for item in aliases[key]:
        if day <= pd.Timestamp(item["valid_until"]):
            allowed.add(str(item["prefix"]).strip().lower())
    return allowed


def parse_contract_id(code: str, product: str, last_trade_date: Any, trade_date: Any = None,
                      aliases: Mapping[str, list[Mapping[str, Any]]] | None = None) -> str:
    """把交易所代码解析为统一的 ``产品+YY+MM`` 编号。

    4 位数字代码直接是 YYMM；3 位数字代码是 YMM，年份取满足个位数字的唯一候选，
    候选范围为最后交易日前后一年。无法唯一确定时报错。
    代码前缀必须是规范前缀，或在该交易日仍有效的历史别名（见 ``allowed_prefixes``），否则报错。
    """
    text = str(code).strip().lower()
    match = _CODE.match(text)
    product_key = str(product).strip().upper()
    allowed = allowed_prefixes(product_key, trade_date, aliases)
    if match is None or match.group(1) not in allowed:
        raise ValueError(
            f"无法解析合约代码 {code}（品种 {product}，交易日 {trade_date}）：前缀不在允许范围 {sorted(allowed)}"
        )
    digits = match.group(2)
    month = int(digits[-2:])
    if not 1 <= month <= 12:
        raise ValueError(f"合约月份非法: {code}")
    if len(digits) == 4:
        year = 2000 + int(digits[:2])
    else:
        ltd = pd.to_datetime(last_trade_date, errors="coerce")
        if pd.isna(ltd):
            raise ValueError(f"{code} 缺少最后交易日，无法推断年代")
        last_digit = int(digits[0])
        candidates = [y for y in range(ltd.year - 1, ltd.year + 2) if y % 10 == last_digit]
        if len(candidates) != 1:
            raise ValueError(f"{code} 年代无法唯一确定，候选 {candidates}，最后交易日 {ltd.date()}")
        year = candidates[0]
    return f"{str(product).strip().upper()}{year % 100:02d}{month:02d}"


def validate_vendor_flags(raw: pd.DataFrame) -> None:
    """MAINCON 每个品种每天恰好一个；SMAINCON 每天最多一个且不与主力为同一行。"""
    for col in ("MAINCON", "SMAINCON"):
        if not raw[col].isin([0, 1]).all():
            raise ValueError(f"{col} 含有 0/1 以外的取值")
    main_counts = raw.groupby(KEY)["MAINCON"].sum()
    bad_main = main_counts[main_counts != 1]
    if len(bad_main):
        raise ValueError(f"{len(bad_main)} 个品种-日的 MAINCON 数量不等于 1，例如 {bad_main.head(5).to_dict()}")
    second_counts = raw.groupby(KEY)["SMAINCON"].sum()
    bad_second = second_counts[second_counts > 1]
    if len(bad_second):
        raise ValueError(f"{len(bad_second)} 个品种-日的 SMAINCON 多于 1 个，例如 {bad_second.head(5).to_dict()}")
    if ((raw["MAINCON"] == 1) & (raw["SMAINCON"] == 1)).any():
        raise ValueError("存在同一合约既是主力又是次主力的行")


def build_full_frame(raw: pd.DataFrame, aliases: Mapping[str, list[Mapping[str, Any]]] | None = None) -> pd.DataFrame:
    """把 CSV 原始表清洗为 contract_day_full 的中间表（每行一个合约一天）。"""
    missing = sorted(set(CSV_REQUIRED) - set(raw.columns))
    if missing:
        raise ValueError(f"CSV 缺少字段: {missing}")
    df = raw.copy()
    for col in ("TICKER_SYMBOL", "CONTRACT_OBJECT", "TRADE_DATE", "LAST_TRADE_DATE"):
        if df[col].isna().any():
            raise ValueError(f"{col} 存在空值")
    for col in _NUMERIC_CSV:
        df[col] = pd.to_numeric(df[col], errors="raise")

    df["date"] = pd.to_datetime(df["TRADE_DATE"], format="%Y-%m-%d", errors="raise").dt.strftime("%Y-%m-%d")
    df["expiry"] = pd.to_datetime(df["LAST_TRADE_DATE"], format="%Y-%m-%d", errors="raise")
    df["code"] = df["TICKER_SYMBOL"].astype(str).str.strip().str.lower()
    df["contract_object"] = df["CONTRACT_OBJECT"].astype(str).str.strip().str.upper()
    df["contract_id"] = [
        parse_contract_id(c, p, e, d, aliases)
        for c, p, e, d in zip(df["code"], df["contract_object"], df["expiry"], df["date"])
    ]

    dup = df.duplicated(["contract_id", "date"], keep=False)
    if dup.any():
        sample = df.loc[dup, ["code", "date"]].head(5).to_dict("records")
        raise ValueError(f"(contract_id, date) 重复 {int(dup.sum())} 行，例如 {sample}")
    validate_vendor_flags(df)

    out = pd.DataFrame({
        "code": df["code"],
        "contract_id": df["contract_id"],
        "contract_object": df["contract_object"],
        "date": df["date"],
        "expiry": df["expiry"],
        "security_id": df["SECURITY_ID"],
        "exchange": df["EXCHANGE_CD"].astype(str).str.strip(),
        "name": df["SEC_SHORT_NAME"].astype(str).str.strip(),
        "open": df["OPEN_PRICE"],
        "high": df["HIGHEST_PRICE"],
        "low": df["LOWEST_PRICE"],
        "close": df["CLOSE_PRICE"],
        "settle": df["SETTL_PRICE"],
        "pre_settle": df["PRE_SETTL_PRICE"],
        "pre_close": df["PRE_CLOSE_PRICE"],
        "volume": df["TURNOVER_VOL"],
        "turnover_value": df["TURNOVER_VALUE"],
        "open_int": df["OPEN_INT"],
        "chg": df["CHG"],
        "chg1": df["CHG1"],
        "chg_pct": df["CHG_PCT"],
        "contract_mark": df["CONTRACT_MARK"],
        "vendor_main": df["MAINCON"].astype(int),
        "vendor_second_main": df["SMAINCON"].astype(int),
        "no_trade": df["TURNOVER_VOL"].fillna(0) <= 0,
        "ohlc_missing": df[["OPEN_PRICE", "HIGHEST_PRICE", "LOWEST_PRICE", "CLOSE_PRICE"]].isna().any(axis=1),
        "vendor_update_time": df["UPDATE_TIME"].astype(str).str.strip(),
        "vendor_tmstamp": df["TMSTAMP"],
    })
    return out.reset_index(drop=True)


def build_research_frame(docs: list[dict[str, Any]],
                        aliases: Mapping[str, list[Mapping[str, Any]]] | None = None) -> pd.DataFrame:
    """把 price_data_research 文档转为 (品种, 日期) 键的研究主力表。

    每个品种每天必须只有一条；date_stamp 与日期不一致时报错。
    expiry 缺失时 research_cid 为 None（该日研究主力无法比对）。
    """
    if not docs:
        raise ValueError("price_data_research 没有记录，无法生成研究主力标签")
    df = pd.DataFrame(docs)
    for col in ("contract_object", "date", "code", "date_stamp"):
        if col not in df.columns:
            raise ValueError(f"price_data_research 缺少字段: {col}")
    df["contract_object"] = df["contract_object"].astype(str).str.upper()
    if df.duplicated(KEY).any():
        raise ValueError("price_data_research 同一品种同一天存在多条记录")
    stamps = (pd.to_datetime(df["date"]) - EPOCH).dt.total_seconds().astype("int64")
    bad_stamp = int((stamps != df["date_stamp"].astype("int64")).sum())
    if bad_stamp:
        raise ValueError(f"price_data_research 中 {bad_stamp} 条 date_stamp 与日期不一致")
    out = pd.DataFrame({
        "contract_object": df["contract_object"],
        "date": df["date"].astype(str),
        "research_code": df["code"].astype(str).str.lower(),
        "research_cid": [
            parse_contract_id(c, p, e, d, aliases) if e is not None and not pd.isna(e) else None
            for c, p, e, d in zip(df["code"], df["contract_object"],
                                  df.get("expiry", pd.Series([None] * len(df))), df["date"])
        ],
    })
    return out.reset_index(drop=True)


def research_day_status(full: pd.DataFrame, research: pd.DataFrame) -> pd.DataFrame:
    """每个 (品种, 日期) 的研究主力状态。

    research_ok=True 表示研究表有记录，且其合约在 CSV 当天存在。
    """
    days = full[KEY].drop_duplicates()
    r = research[KEY + ["research_code", "research_cid"]]
    pairs = (
        full[KEY + ["contract_id"]].drop_duplicates()
        .rename(columns={"contract_id": "research_cid"})
    )
    hit = (
        r.merge(pairs, on=KEY + ["research_cid"], how="inner")[KEY]
        .drop_duplicates()
        .assign(research_ok=True)
    )
    out = days.merge(r, on=KEY, how="left").merge(hit, on=KEY, how="left")
    out["research_ok"] = out["research_ok"].eq(True)
    return out


def apply_research_labels(full: pd.DataFrame, research: pd.DataFrame, synced_at: str) -> pd.DataFrame:
    """为每一行生成 research_main：1 为研究主力，0 为同日其他合约，None 为无法比对。"""
    status = research_day_status(full, research)
    merged = full[KEY + ["contract_id"]].merge(status, on=KEY, how="left", sort=False)
    if len(merged) != len(full):
        raise ValueError("研究标签合并后行数改变，研究表键不唯一")
    ok = merged["research_ok"].to_numpy(dtype=bool)
    same = (merged["contract_id"] == merged["research_cid"]).to_numpy(dtype=bool)
    values = [int(s) if o else None for o, s in zip(ok, same)]
    out = full.copy()
    out["research_main"] = pd.Series(values, index=out.index, dtype=object)
    out["research_main_source"] = RESEARCH_SOURCE
    out["research_main_synced_at"] = synced_at
    return out


def build_vendor_main_frame(full: pd.DataFrame, research: pd.DataFrame) -> pd.DataFrame:
    """每个品种每天一行：CSV 主力、次主力，以及研究主力和两者是否一致。"""
    status = research_day_status(full, research)
    main = (
        full[full["vendor_main"] == 1][KEY + ["code", "contract_id"]]
        .rename(columns={"code": "main_contract", "contract_id": "main_contract_id"})
    )
    second = (
        full[full["vendor_second_main"] == 1][KEY + ["code", "contract_id"]]
        .rename(columns={"code": "second_contract", "contract_id": "second_contract_id"})
    )
    out = main.merge(second, on=KEY, how="left").merge(status, on=KEY, how="left")
    if len(out) != full[KEY].drop_duplicates().shape[0]:
        raise ValueError("vendor_main_map 行数与品种-日数不一致")
    ok = out["research_ok"].eq(True).to_numpy(dtype=bool)
    same = (out["main_contract_id"] == out["research_cid"]).to_numpy(dtype=bool)
    out["main_agree"] = pd.Series([int(s) if o else None for o, s in zip(ok, same)], index=out.index, dtype=object)
    out["research_main_contract"] = out["research_code"]
    out["research_main_contract_id"] = out["research_cid"]
    cols = KEY + [
        "main_contract", "main_contract_id", "second_contract", "second_contract_id",
        "research_main_contract", "research_main_contract_id", "main_agree",
    ]
    return out[cols].reset_index(drop=True)


def _none(value: Any) -> Any:
    return None if pd.isna(value) else value


def _num(value: Any) -> float | None:
    return None if pd.isna(value) else float(value)


def _int(value: Any, field: str, code: str) -> int | None:
    if pd.isna(value):
        return None
    if not float(value).is_integer():
        raise ValueError(f"{code} {field} 不是整数: {value}")
    return int(value)


def to_full_documents(frame: pd.DataFrame, source: str, source_sha256: str, updated_at: Any) -> list[dict[str, Any]]:
    docs: list[dict[str, Any]] = []
    for row in frame.to_dict("records"):
        date = pd.Timestamp(row["date"])
        code = row["code"]
        docs.append({
            "code": code,
            "contract_id": row["contract_id"],
            "security_id": _int(row["security_id"], "security_id", code),
            "exchange": row["exchange"],
            "name": row["name"],
            "contract_object": row["contract_object"],
            "date": row["date"],
            "date_stamp": int((date - EPOCH).total_seconds()),
            "expiry": pd.Timestamp(row["expiry"]).strftime("%Y-%m-%d"),
            "open": _num(row["open"]),
            "high": _num(row["high"]),
            "low": _num(row["low"]),
            "close": _num(row["close"]),
            "settle": _num(row["settle"]),
            "pre_settle": _num(row["pre_settle"]),
            "pre_close": _num(row["pre_close"]),
            "volume": _int(row["volume"], "volume", code),
            "turnover_value": _int(row["turnover_value"], "turnover_value", code),
            "open_int": _int(row["open_int"], "open_int", code),
            "chg": _num(row["chg"]),
            "chg1": _num(row["chg1"]),
            "chg_pct": _num(row["chg_pct"]),
            "contract_mark": _none(row["contract_mark"]),
            "vendor_main": int(row["vendor_main"]),
            "vendor_second_main": int(row["vendor_second_main"]),
            "research_main": _none(row.get("research_main")),
            "research_main_source": _none(row.get("research_main_source")),
            "research_main_synced_at": _none(row.get("research_main_synced_at")),
            "no_trade": bool(row["no_trade"]),
            "ohlc_missing": bool(row["ohlc_missing"]),
            "source": source,
            "source_sha256": source_sha256,
            "vendor_update_time": _none(row["vendor_update_time"]),
            "vendor_tmstamp": _int(row["vendor_tmstamp"], "vendor_tmstamp", code),
            "updated_at": updated_at,
        })
    return docs


def to_main_documents(day: pd.DataFrame, source: str, updated_at: Any) -> list[dict[str, Any]]:
    docs: list[dict[str, Any]] = []
    for row in day.to_dict("records"):
        date = pd.Timestamp(row["date"])
        docs.append({
            "contract_object": row["contract_object"],
            "date": row["date"],
            "date_stamp": int((date - EPOCH).total_seconds()),
            "main_contract": row["main_contract"],
            "main_contract_id": row["main_contract_id"],
            "second_contract": _none(row["second_contract"]),
            "second_contract_id": _none(row["second_contract_id"]),
            "research_main_contract": _none(row["research_main_contract"]),
            "research_main_contract_id": _none(row["research_main_contract_id"]),
            "main_agree": _none(row["main_agree"]),
            "rule": VENDOR_RULE,
            "rule_note": VENDOR_RULE_NOTE,
            "source": source,
            "updated_at": updated_at,
        })
    return docs
