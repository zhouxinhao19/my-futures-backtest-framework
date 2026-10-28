from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np
import warnings

import pandas as pd
import yaml


ROOT = Path(__file__).resolve().parents[1]


_DATA_CONFIG_CACHE: dict[str, Any] | None = None
_UNIVERSE_CACHE: dict[str, Any] | None = None


def data_config() -> dict[str, Any]:
    """读取数据配置。同一次进程里只解析一次 ``configs/data.yaml``。"""
    global _DATA_CONFIG_CACHE
    if _DATA_CONFIG_CACHE is None:
        with (ROOT / "configs" / "data.yaml").open(encoding="utf-8") as f:
            _DATA_CONFIG_CACHE = yaml.safe_load(f) or {}
    return _DATA_CONFIG_CACHE


_DATA_CONFIG = data_config()
MONGO_URI = os.getenv("QA_MONGO_URI", _DATA_CONFIG.get(
    "mongo_uri", "mongodb://127.0.0.1:27017"
))
MONGO_DB = os.getenv("QA_MONGO_DB", _DATA_CONFIG.get(
    "mongo_db", "quantaxis_price_data"
))


def config() -> dict[str, Any]:
    """读取品种池和交易规格。同一次进程里只解析一次 ``configs/universe.yaml``。"""
    global _UNIVERSE_CACHE
    if _UNIVERSE_CACHE is None:
        with (ROOT / "configs" / "universe.yaml").open(encoding="utf-8") as f:
            _UNIVERSE_CACHE = yaml.safe_load(f)
    return _UNIVERSE_CACHE


def symbols() -> list[str]:
    """返回配置中所有有主力 CSV 的品种。"""
    return [
        symbol for symbol, spec in config()["symbols"].items()
        if spec.get("file")
    ]


def pool_symbols(pool: str) -> list[str]:
    """返回配置中的规范品种池，并统一为交易所品种代码。"""
    versioned = (config().get("pools") or {}).get(pool)
    if isinstance(versioned, dict) and versioned.get("members"):
        return [str(symbol).upper() for symbol in versioned["members"]]
    pools = config().get("pool_definitions", {})
    if pool not in pools:
        raise KeyError(f"未知品种池: {pool}")
    return [str(symbol).upper() for symbol in pools[pool]]


def symbol_spec(symbol: str) -> dict[str, Any]:
    """按品种代码读取交易规格。

    品种池使用 ``AU`` 这类代码。``universe.yaml`` 里保留的 ``AU_MAIN``
    只作为乘数、保证金和手续费规格的别名。
    """
    specs = config()["symbols"]
    products = config().get("products") or {}
    key = str(symbol).upper()
    product_key = key.removesuffix("_MAIN")
    if key in specs:
        return specs[key]
    alias = key if key.endswith("_MAIN") else f"{key}_MAIN"
    if alias in specs:
        if product_key not in products:
            warnings.warn(
                f"{alias} 是过渡别名，新代码请使用 products.{product_key}",
                DeprecationWarning,
                stacklevel=2,
            )
        return specs[alias]
    product = products.get(product_key)
    if product and product.get("multiplier") is not None:
        multiplier = product["multiplier"]
        if isinstance(multiplier, list):
            multiplier = float(multiplier[-1]["value"])
        return {**product, "multiplier": float(multiplier), "variety": product_key}
    raise KeyError(f"{symbol} 不在 configs/universe.yaml 的交易规格中")


def contract_symbols() -> list[str]:
    """返回拥有全合约面板、可做具体合约账户的品种。"""
    return [
        symbol for symbol, spec in config()["symbols"].items()
        if spec.get("data_mode") == "contract" and spec.get("panel_file")
    ]


def pool_main_symbols(pool: str, require_contract: bool = False) -> list[str]:
    """将配置池产品代码映射为 universe.yaml 中的 *_MAIN 名称。"""
    cfg = config()["symbols"]
    result = []
    for product in pool_symbols(pool):
        name = product if product.endswith("_MAIN") else f"{product}_MAIN"
        if name not in cfg:
            continue
        if require_contract and cfg[name].get("data_mode") != "contract":
            continue
        result.append(name)
    return result


def main_only_symbols() -> list[str]:
    """返回只有主力序列、尚不能做真实合约账户的品种。"""
    return [
        symbol for symbol, spec in config()["symbols"].items()
        if spec.get("data_mode") == "main_only"
    ]


def load_main(symbol: str) -> pd.DataFrame:
    cfg = config()
    spec = cfg["symbols"][symbol]
    df = pd.read_csv(Path(cfg["data_root"]) / spec["file"], parse_dates=["date"])
    df = df.sort_values("date").drop_duplicates("date").set_index("date")
    return df


def validate_contract_frame(frame: pd.DataFrame, symbol: str,
                             allow_settle_fallback: bool = False) -> None:
    required = {"date", "code", "open", "close", "settle", "volume"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{symbol} 数据缺少字段: {sorted(missing)}")
    if frame.duplicated(["date", "code"]).any():
        raise ValueError(f"{symbol} 存在 date/code 重复行情")
    numeric = frame[["open", "close", "settle"]].apply(pd.to_numeric, errors="coerce")
    if numeric.isna().any().any():
        raise ValueError(f"{symbol} 存在 OHLC/settle 非数值或缺失")
    if (numeric["open"] <= 0).any() or (numeric["close"] <= 0).any():
        raise ValueError(f"{symbol} 存在非正价格")
    if not allow_settle_fallback and "settle_source" in frame:
        if (frame["settle_source"].astype(str) != "panel").any():
            raise ValueError(f"{symbol} 使用了非交易所结算价")


def load_main_map(
    symbol: str,
    start: str | None = None,
    end: str | None = None,
) -> pd.DataFrame:
    frame = load_main(symbol).reset_index()
    if start is not None:
        frame = frame[frame["date"] >= pd.Timestamp(start)]
    if end is not None:
        frame = frame[frame["date"] <= pd.Timestamp(end)]
    result = frame[["date", "contract", "adj_close"]].assign(
        contract=lambda x: x["contract"].astype(str).str.lower(),
    )
    return result.assign(
        code=lambda x: x["contract"],
        symbol=symbol,
    )


def load_main_only_dataset(symbol: str, start: str, end: str) -> pd.DataFrame:
    """读取没有全合约面板的主力序列。

    该模式只用于主力因子/价格研究；settle 使用 close 作为代理，
    不代表真实合约结算价，也不应被标记为 contract 模式。
    """
    cfg = config()
    spec = cfg["symbols"][symbol]
    frame = pd.read_csv(
        Path(cfg["data_root"]) / spec["file"],
        parse_dates=["date"],
    )
    frame = (
        frame.sort_values("date")
        .drop_duplicates("date")
        .loc[lambda x: x["date"].between(start, end)]
        .copy()
    )
    frame = normalize_market_columns(frame)
    frame["contract"] = frame["contract"].astype(str).str.lower()
    frame["code"] = frame["contract"]
    frame["main_contract"] = frame["contract"]
    frame["settle"] = pd.to_numeric(frame["close"], errors="coerce")
    frame["pre_settle"] = frame["settle"].shift(1)
    frame["position"] = pd.to_numeric(
        frame.get("open_interest", pd.Series(index=frame.index)),
        errors="coerce",
    )
    frame["trade"] = pd.to_numeric(
        frame.get("volume", pd.Series(index=frame.index)),
        errors="coerce",
    )
    return frame.dropna(subset=["date", "open", "close", "adj_close"])[
        [
            "date", "code", "main_contract", "open", "close", "adj_close",
            "settle", "pre_settle", "position", "trade", "volume",
        ]
    ].reset_index(drop=True)


def normalize_market_columns(frame: pd.DataFrame) -> pd.DataFrame:
    """统一逐合约行情的常用字段名，不填充真实价格字段。"""
    result = frame.copy()
    aliases = {
        "vol": "volume",
        "oi": "position",
        "open_interest": "position",
        "preSettlement": "pre_settle",
    }
    for source, target in aliases.items():
        if target not in result.columns and source in result.columns:
            result[target] = result[source]
    if "trade" not in result.columns and "volume" in result.columns:
        result["trade"] = result["volume"]
    for column in (
        "open", "high", "low", "close", "settle", "pre_settle",
        "volume", "position", "trade", "adj_open", "adj_high", "adj_low",
        "adj_close", "adj_settle",
        "basis_rate", "spot_price",
    ):
        if column in result.columns:
            result[column] = pd.to_numeric(result[column], errors="coerce")
    if {"open", "high", "low", "close"}.issubset(result.columns):
        result["ohlc_valid"] = result[
            ["open", "high", "low", "close"]
        ].notna().all(axis=1)
    else:
        result["ohlc_valid"] = False
    if "volume" in result.columns:
        result["no_trade"] = result["volume"].fillna(0).le(0)
    else:
        result["no_trade"] = True
    return result


def load_qa_dataset(symbol: str, start: str, end: str) -> pd.DataFrame:
    """内部兼容查询：读取 Mongo future_day 并补回 settle。

    账户回测请使用 ``load_contract_dataset``；新研究请使用
    ``load_research_price_panel`` 或 ``load_volume_oi_panel``。
    """
    from pymongo import MongoClient

    cfg = config()
    db = MongoClient(MONGO_URI)[MONGO_DB]
    etf_source = {"$or": [{"source": "etf"}, {"source": {"$exists": False}}]}
    maps = list(db.future_main.find(
        {"$and": [
            {"symbol": symbol, "date": {"$gte": start, "$lte": end}},
            etf_source,
        ]},
        {"_id": 0},
    ).sort("date_stamp", 1))
    if not maps:
        return pd.DataFrame()
    codes = sorted({item["contract"] for item in maps})
    bars = pd.DataFrame(list(db.future_day.find(
        {"$and": [
            {"code": {"$in": codes}, "date": {"$gte": start, "$lte": end}},
            etf_source,
        ]},
        {"_id": 0},
    ).sort("date_stamp", 1)))
    if bars.empty:
        return pd.DataFrame()
    bars = normalize_market_columns(bars)
    bars["date"] = pd.to_datetime(bars["date"])
    bars["volume"] = bars["volume"].fillna(bars["trade"])
    main = pd.DataFrame(maps)
    main["date"] = pd.to_datetime(main["date"])
    main["code"] = main["contract"].astype(str)
    bars["date"] = pd.to_datetime(bars["date"])
    result = bars.merge(
        main[["date", "contract", "adj_close"]].rename(columns={"contract": "main_contract"}),
        on="date", how="left", validate="many_to_one",
    )
    if result["main_contract"].isna().any():
        missing_dates = int(result["main_contract"].isna().sum())
        raise ValueError(f"{symbol} 有 {missing_dates} 行缺少主力映射")
    return result


def load_contract_dataset(
    symbol: str, start: str, end: str
) -> pd.DataFrame:
    """账户回测使用的主力序列。

    ``price_data_research`` 每个品种每天只有一条合约，该合约就是已经选定的主力。
    ``code`` 与 ``main_contract`` 都等于这一条记录上的合约代码。
    """
    frame = load_volume_oi_panel(symbol, start, end).copy()
    frame["code"] = frame["main_contract"].astype(str).str.lower()
    frame["main_contract"] = frame["code"]
    price_columns = ["settle", "adj_settle"]
    frame = frame.dropna(subset=["date", "code", *price_columns]).copy()
    if frame.duplicated(["date", "code"]).any():
        raise ValueError(f"{symbol} 的选定主力序列存在重复日期")
    if frame.empty:
        raise RuntimeError(
            f"{symbol} 在 {MONGO_DB}.price_data_research 中没有选定主力序列"
        )
    if frame["settle"].isna().any() or frame["adj_settle"].isna().any():
        raise ValueError(f"{symbol} 的 settle/adj_settle 存在缺失")
    return frame.reset_index(drop=True)


def load_research_price_panel(
    start: str | None = None,
    end: str | None = None,
    products: list[str] | None = None,
) -> pd.DataFrame:
    """唯一的价格研究面板入口，固定使用 price_data。"""
    return load_mongo_commodity_panel(start=start, end=end, products=products)


def load_volume_oi_panel(
    symbol: str, start: str, end: str, roll_window: int = 1
) -> pd.DataFrame:
    """唯一的价量仓研究入口，固定使用 price_data_research。"""
    return load_commodity_volume_oi_panel(symbol, start, end, roll_window)


def _product_code(symbol: str) -> str:
    product = str(symbol).upper()
    for suffix in ("_TS_MAIN", "_MAIN"):
        if product.endswith(suffix):
            product = product[: -len(suffix)]
            break
    return product


_RESEARCH_PROJECTION = {
    "_id": 0, "date": 1, "code": 1, "contract_object": 1,
    "expiry": 1, "open": 1, "high": 1, "low": 1, "close": 1,
    "settle": 1, "adj_open": 1, "adj_high": 1, "adj_low": 1,
    "adj_close": 1, "adj_settle": 1, "basis_rate": 1, "spot_price": 1,
    "trade_turnover_vol": 1,
    "trade_turnover_value": 1, "trade_open_int": 1,
}


def _fetch_research_rows(products: list[str], start: str, end: str | None) -> pd.DataFrame:
    """一次查出这些品种在 ``price_data_research`` 里的选定主力。"""
    wanted = sorted({_product_code(symbol) for symbol in products})
    label = ",".join(wanted)
    try:
        from pymongo import MongoClient

        with MongoClient(MONGO_URI) as client:
            rows = list(client[MONGO_DB].price_data_research.find(
                {"contract_object": {"$in": wanted}, "date": _date_query(start, end)},
                _RESEARCH_PROJECTION,
            ).sort("date", 1))
    except Exception as exc:
        raise RuntimeError(
            f"{label} 查询 {MONGO_DB}.price_data_research 失败；"
            "禁止自动回退到其它集合或本地文件"
        ) from exc
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).rename(columns={
        "trade_turnover_vol": "volume",
        "trade_turnover_value": "turnover_value",
        "trade_open_int": "position",
    })


def _rows_for_product(raw: pd.DataFrame, symbol: str) -> pd.DataFrame:
    product = _product_code(symbol)
    if raw.empty or "contract_object" not in raw.columns:
        return raw.iloc[0:0].copy()
    return raw.loc[raw["contract_object"].astype(str).str.upper() == product].copy()


def load_main_panels(products: list[str], start: str, end: str) -> pd.DataFrame:
    """一次查询多个品种的选定主力，再按请求顺序做与单品种入口相同的清洗。"""
    if not products:
        raise ValueError("品种列表为空")
    raw = _fetch_research_rows(products, start, end)
    frames = [
        _research_panel_from_rows(_rows_for_product(raw, symbol), symbol)
        for symbol in products
    ]
    return pd.concat(frames, ignore_index=True)


def load_commodity_volume_oi_panel(
    symbol: str,
    start: str,
    end: str,
    roll_window: int = 1,
) -> pd.DataFrame:
    """内部价量仓面板实现；统一研究入口请使用 ``load_volume_oi_panel``。

    读取规范 ``price_data_research`` 价量仓面板。

    该入口不允许回退到其它集合或本地 CSV。``adj_close`` 只用于收益；
    ``volume`` 和 ``position`` 保持交易所手数口径。
    """
    # price_data_research 已经是一条/品种/日期的连续序列，不能再按
    # OI 选择合约。该集合同时保存复权价格、成交量和持仓量。
    raw = _rows_for_product(_fetch_research_rows([symbol], start, end), symbol)
    return _research_panel_from_rows(raw, symbol, roll_window)


def _research_panel_from_rows(
    raw: pd.DataFrame,
    symbol: str,
    roll_window: int = 1,
) -> pd.DataFrame:
    if raw.empty:
        raise RuntimeError(
            f"{symbol} 在 {MONGO_DB}.price_data_research 中没有价量仓数据；"
            "禁止自动回退到 future_day 或主力 CSV"
        )

    x = normalize_market_columns(raw.copy())
    if "main_contract" not in x.columns:
        x["main_contract"] = x.get("code")
    if "code" not in x.columns:
        x["code"] = x["main_contract"]
    x["date"] = pd.to_datetime(x["date"], errors="coerce")
    x["code"] = x["code"].astype(str).str.lower()
    x["main_contract"] = x["main_contract"].astype(str).str.lower()
    # 研究集合中的 code 是连续序列当天使用的合约，仅用于标记换月；
    # 不做 OI 最大合约筛选。
    x = x.sort_values("date").drop_duplicates("date").reset_index(drop=True)
    required = {
        "date", "main_contract", "adj_close", "adj_settle", "volume",
        "position", "turnover_value",
    }
    if not required.issubset(x.columns):
        raise ValueError(
            f"{symbol} 价量仓面板缺少字段: "
            f"{sorted(required - set(x.columns))}"
        )
    x["volume"] = pd.to_numeric(x["volume"], errors="coerce")
    x["position"] = pd.to_numeric(x["position"], errors="coerce")
    x["adj_close"] = pd.to_numeric(x["adj_close"], errors="coerce")
    x["symbol"] = symbol
    x["rollover"] = x["main_contract"].ne(x["main_contract"].shift(1))
    rollover_dates = x.loc[x["rollover"], "date"]
    if len(rollover_dates):
        invalid = pd.Series(False, index=x.index)
        for date in rollover_dates:
            invalid |= (x["date"] - date).abs().dt.days.le(roll_window)
        x["rollover_window"] = invalid
    else:
        x["rollover_window"] = False
    return x


def load_qa_main_map(symbol: str, start: str, end: str) -> pd.DataFrame:
    """选定主力映射：每个交易日的合约代码来自主力序列本身。"""
    frame = load_contract_dataset(symbol, start, end)
    return frame.set_index("date")[["code"]].rename(columns={"code": "contract"})


FINANCIAL_FUTURES = frozenset({"IF", "IH", "IC", "IM", "TF", "T", "TS", "TL"})

# 多品种研究的统一品种池由 configs/universe.yaml 管理。
MULTI_PRODUCT_SYMBOLS = tuple(pool_symbols("commodity_xs"))


def load_mongo_commodity_panel(
    start: str | None = None,
    end: str | None = None,
    products: list[str] | None = None,
) -> pd.DataFrame:
    """内部价格面板实现；统一研究入口请使用 ``load_research_price_panel``。

    ``price_data_research`` 每个品种每天只有一条合约，该合约就是主力。
    ``adj_close`` 和 ``adj_settle`` 是后复权价格。同一天出现多条合约时直接报错。
    """
    from pymongo import MongoClient

    collection_name = data_config().get("collections", {}).get(
        "price_panel", "price_data_research"
    )
    query: dict[str, Any] = {
        "contract_object": {"$nin": sorted(FINANCIAL_FUTURES)},
    }
    if start is not None or end is not None:
        query["date"] = {}
        if start is not None:
            query["date"]["$gte"] = start
        if end is not None:
            query["date"]["$lte"] = end
    if products:
        query["contract_object"] = {
            "$in": sorted({str(product).upper() for product in products}),
            "$nin": sorted(FINANCIAL_FUTURES),
        }

    projection = {
        "_id": 0,
        "contract_object": 1,
        "code": 1,
        "date": 1,
        "expiry": 1,
        "open": 1,
        "high": 1,
        "low": 1,
        "close": 1,
        "settle": 1,
        "adj_open": 1,
        "adj_high": 1,
        "adj_low": 1,
        "adj_close": 1,
        "adj_settle": 1,
        "basis_rate": 1,
        "spot_price": 1,
        "trade_turnover_vol": 1,
        "trade_turnover_value": 1,
        "trade_open_int": 1,
    }
    with MongoClient(MONGO_URI) as client:
        rows = list(client[MONGO_DB][collection_name].find(query, projection))
    if not rows:
        raise RuntimeError(
            f"{MONGO_DB}.{collection_name} 没有符合条件的商品研究行情；"
            "禁止自动回退到其它数据源"
        )

    raw = pd.DataFrame(rows)
    raw["date"] = pd.to_datetime(raw["date"], errors="coerce")
    if "expiry" in raw.columns:
        raw["expiry"] = pd.to_datetime(raw["expiry"], errors="coerce")
    raw["contract_object"] = raw["contract_object"].astype(str).str.upper()
    raw["code"] = raw["code"].astype(str).str.lower()
    for column in (
        "adj_open", "adj_high", "adj_low", "adj_close", "adj_settle",
        "open", "high", "low", "close", "settle", "basis_rate", "spot_price",
        "trade_turnover_vol", "trade_turnover_value", "trade_open_int",
    ):
        if column in raw.columns:
            raw[column] = pd.to_numeric(raw[column], errors="coerce")
    raw = raw.dropna(subset=["date", "adj_close", "adj_settle"])
    if raw.empty:
        raise RuntimeError(f"{MONGO_DB}.{collection_name} 清洗后没有可用行情")
    duplicated = raw.duplicated(["contract_object", "date"])
    if duplicated.any():
        sample = raw.loc[duplicated, ["contract_object", "date"]].head(5)
        raise ValueError(
            f"{collection_name} 存在同一品种同一天多条合约，不能判定主力: "
            f"{sample.to_dict('records')}"
        )

    selected = (
        raw.rename(columns={
            "contract_object": "symbol",
            "code": "main_contract",
            "trade_turnover_vol": "volume",
            "trade_turnover_value": "turnover_value",
            "trade_open_int": "position",
        })
        .sort_values(["date", "symbol"])
        .reset_index(drop=True)
    )
    selected["available_at"] = selected.groupby(
        "symbol", sort=False
    )["date"].shift(-1)
    columns = [
        column for column in (
            "date", "symbol", "main_contract", "expiry", "open", "high",
            "low", "close", "settle", "adj_open", "adj_high", "adj_low",
            "adj_close", "adj_settle", "basis_rate", "spot_price",
            "volume", "position", "turnover_value", "available_at",
        )
        if column in selected.columns
    ]
    return selected[columns]


def load_mongo_basis_rates(
    start: str | None = None,
    end: str | None = None,
    products: list[str] | None = None,
) -> pd.DataFrame:
    """Load `basis_rates` for the commodity research pool. Fail closed if empty."""
    from pymongo import MongoClient

    products = [
        str(item).upper().removesuffix("_MAIN")
        for item in (products or pool_symbols("commodity_xs"))
    ]
    query: dict[str, Any] = {"symbol": {"$in": products}}
    if start is not None or end is not None:
        query["date"] = {}
        if start is not None:
            query["date"]["$gte"] = start
        if end is not None:
            query["date"]["$lte"] = end
    with MongoClient(MONGO_URI) as client:
        rows = list(
            client[MONGO_DB].basis_rates.find(
                query, {"_id": 0, "symbol": 1, "date": 1, "basis_rate": 1}
            )
        )
    if not rows:
        raise RuntimeError(
            f"{MONGO_DB}.basis_rates 没有符合条件的基差；禁止静默回退"
        )
    frame = pd.DataFrame(rows)
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    frame["symbol"] = frame["symbol"].astype(str).str.upper()
    frame["basis_rate"] = pd.to_numeric(frame["basis_rate"], errors="coerce")
    return frame.dropna(subset=["date", "symbol"]).drop_duplicates(
        ["date", "symbol"]
    )


def multiplier_map(
    symbols: list[str],
    panel: pd.DataFrame | None = None,
) -> dict[str, dict[str, Any]]:
    """确定每个品种的合约乘数。

    优先使用 ``universe.yaml`` 的交易规格。规格缺失时，用研究面板里的
    ``成交额 / (成交量 × 结算价)`` 测定。全样本稳定落在同一个整数上时用该整数；
    若按时间分成若干段、每段都稳定落在各自的整数上，则按日期使用对应整数。
    测不准就停止，不用默认乘数填上。
    """
    result: dict[str, dict[str, Any]] = {}
    unresolved: list[str] = []
    for symbol in symbols:
        configured = _configured_multiplier(symbol)
        if configured is not None:
            result[str(symbol)] = {
                "multiplier": configured,
                "source": "universe",
            }
            continue
        resolved = _implied_multiplier_spec(symbol, panel)
        if resolved is None:
            unresolved.append(str(symbol))
            continue
        result[str(symbol)] = resolved
    if unresolved:
        raise RuntimeError(
            "无法确定合约乘数，已停止手数回测: " + ", ".join(unresolved)
        )
    return result


def _configured_multiplier(symbol: str) -> float | None:
    try:
        spec = symbol_spec(symbol)
    except KeyError:
        return None
    value = spec.get("multiplier")
    if value in (None, ""):
        return None
    multiplier = float(value)
    if multiplier <= 0:
        raise ValueError(f"{symbol} 的合约乘数必须为正")
    return multiplier


def _implied_ratio(symbol: str, panel: pd.DataFrame | None) -> pd.Series | None:
    if panel is None or panel.empty:
        return None
    required = {"date", "symbol", "settle", "volume", "turnover_value"}
    if not required.issubset(panel.columns):
        return None
    group = panel.loc[panel["symbol"].astype(str).eq(str(symbol)), [
        "date", "settle", "volume", "turnover_value",
    ]].copy()
    if group.empty:
        return None
    group["date"] = pd.to_datetime(group["date"]).dt.normalize()
    settle = pd.to_numeric(group["settle"], errors="coerce")
    volume = pd.to_numeric(group["volume"], errors="coerce")
    turnover = pd.to_numeric(group["turnover_value"], errors="coerce")
    implied = turnover / (volume * settle)
    implied = implied.replace([np.inf, -np.inf], np.nan)
    valid = implied.gt(0) & implied.notna() & group["date"].notna()
    if int(valid.sum()) < 20:
        return None
    ratio = pd.Series(
        implied.loc[valid].to_numpy(dtype=float),
        index=pd.DatetimeIndex(group.loc[valid, "date"]),
        dtype=float,
    )
    return ratio.groupby(level=0).median().sort_index()


def _stable_regimes(ratio: pd.Series) -> list[dict[str, Any]] | None:
    """把连续、且各自稳定取整的日期段拆开。短段或取整不稳则返回 None。"""
    rounded = ratio.round()
    block = rounded.ne(rounded.shift()).cumsum()
    regimes: list[dict[str, Any]] = []
    for _, part in ratio.groupby(block):
        mode = float(part.round().iloc[0])
        if len(part) < 20 or mode < 1 or abs(float(part.median()) - mode) > 0.05:
            return None
        if float((part.round() == mode).mean()) < 0.95:
            return None
        regimes.append({
            "multiplier": mode,
            "start": pd.Timestamp(part.index.min()).strftime("%Y-%m-%d"),
            "end": pd.Timestamp(part.index.max()).strftime("%Y-%m-%d"),
            "days": int(len(part)),
        })
    return regimes or None


def _implied_multiplier_spec(
    symbol: str,
    panel: pd.DataFrame | None,
) -> dict[str, Any] | None:
    ratio = _implied_ratio(symbol, panel)
    if ratio is None:
        return None
    regimes = _stable_regimes(ratio)
    if regimes is None:
        return None
    source = "turnover_value/(volume*settle)"
    if len(regimes) == 1:
        return {"multiplier": regimes[0]["multiplier"], "source": source}
    schedule = pd.Series(ratio.round().to_numpy(dtype=float), index=ratio.index)
    return {
        "multiplier": schedule,
        "source": source + " by date",
        "regimes": regimes,
    }


def _implied_multiplier(symbol: str, panel: pd.DataFrame | None) -> float | None:
    spec = _implied_multiplier_spec(symbol, panel)
    if spec is None or not isinstance(spec["multiplier"], float):
        return None
    return float(spec["multiplier"])


def attach_basis_rates(panel: pd.DataFrame, basis: pd.DataFrame) -> pd.DataFrame:
    out = panel.merge(basis, on=["date", "symbol"], how="left")
    if out["basis_rate"].notna().mean() < 0.80:
        raise RuntimeError("基差覆盖率低于 80%，禁止继续研究代理回测")
    return out


MEMBER_POSITION_FIELDS = (
    "top20_long", "top20_short", "net_position", "registered_receipt",
)


def load_member_position(
    start: str | None = None,
    end: str | None = None,
    products: list[str] | None = None,
) -> pd.DataFrame:
    """读取会员持仓与注册仓单日频表，键为 date + symbol。

    查询方式与 ``load_mongo_basis_rates`` 相同。集合为空时直接报错。
    """
    from pymongo import MongoClient

    collection_name = data_config().get("collections", {}).get(
        "member_position", "member_position_daily"
    )
    query: dict[str, Any] = {}
    if products:
        query["contract_object"] = {
            "$in": sorted({str(item).upper() for item in products})
        }
    if start is not None or end is not None:
        query["date"] = {}
        if start is not None:
            query["date"]["$gte"] = start
        if end is not None:
            query["date"]["$lte"] = end
    projection = {"_id": 0, "symbol": 1, "date": 1, "receipt_unit": 1}
    projection.update({field: 1 for field in MEMBER_POSITION_FIELDS})
    with MongoClient(MONGO_URI) as client:
        rows = list(client[MONGO_DB][collection_name].find(query, projection))
    if not rows:
        raise RuntimeError(
            f"{MONGO_DB}.{collection_name} 没有符合条件的会员持仓；禁止静默回退"
        )
    frame = pd.DataFrame(rows)
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    frame["symbol"] = frame["symbol"].astype(str).str.upper()
    for field in MEMBER_POSITION_FIELDS:
        frame[field] = pd.to_numeric(frame[field], errors="coerce")
    return frame.dropna(subset=["date", "symbol"]).drop_duplicates(
        ["date", "symbol"]
    ).reset_index(drop=True)


def attach_member_position(panel: pd.DataFrame, member: pd.DataFrame) -> pd.DataFrame:
    """把会员持仓与仓单左连接到研究面板。对不齐的日期留空，不设覆盖率门槛。"""
    fields = [column for column in MEMBER_POSITION_FIELDS if column in member.columns]
    right = member[["date", "symbol", *fields]]
    return panel.merge(right, on=["date", "symbol"], how="left", validate="many_to_one")


CONTRACT_FULL_REQUIRED = (
    "code", "contract_id", "contract_object", "date", "date_stamp", "open", "high", "low",
    "close", "settle", "volume", "open_int", "vendor_main", "vendor_second_main", "research_main",
)
VENDOR_MAIN_REQUIRED = (
    "contract_object", "date", "main_contract", "main_contract_id", "second_contract",
    "research_main_contract_id", "main_agree", "rule",
)


def _read_mongo_collection(collection: str, query: dict[str, Any], label: str,
                           required: tuple[str, ...],
                           fields: tuple[str, ...] | None = None) -> pd.DataFrame:
    """读取规范集合。集合为空或缺字段时直接报错，不回退到其它集合。"""
    from pymongo import MongoClient

    projection: dict[str, int] = {"_id": 0}
    if fields is not None:
        projection.update({field: 1 for field in fields})
    rows = list(MongoClient(MONGO_URI)[MONGO_DB][collection].find(
        query, projection
    ).sort("date_stamp", 1))
    if not rows:
        raise RuntimeError(f"{MONGO_DB}.{collection} 中没有 {label} 的数据；禁止自动回退")
    frame = pd.DataFrame(rows)
    missing = sorted(set(required) - set(frame.columns))
    if missing:
        raise ValueError(f"{MONGO_DB}.{collection} 缺少字段: {missing}")
    frame["date"] = pd.to_datetime(frame["date"])
    return frame.reset_index(drop=True)


def _date_query(start: str, end: str | None) -> dict[str, str]:
    date_filter: dict[str, str] = {"$gte": start}
    if end is not None:
        date_filter["$lte"] = end
    return date_filter


def load_contract_full_panel(products: list[str], start: str, end: str | None = None,
                             fields: tuple[str, ...] | None = None) -> pd.DataFrame:
    """读取全合约原始日行情（不复权），每行一个合约一天。

    这是研究入口，不参与账户回测；账户回测仍使用 ``load_contract_dataset``。
    ``fields`` 给定时只读取这些字段，并以它们作为必需字段校验。
    """
    cfg = data_config()
    collection = cfg.get("collections", {}).get("contract_full", "contract_day_full")
    upper = [str(p).upper() for p in products]
    query = {"contract_object": {"$in": upper}, "date": _date_query(start, end)}
    return _read_mongo_collection(
        collection, query, f"{upper} 在 {start}~{end} 的全合约",
        fields or CONTRACT_FULL_REQUIRED, fields,
    )


def load_vendor_main_map(product: str, start: str, end: str | None = None) -> pd.DataFrame:
    """读取供应商（CSV MAINCON）主力映射，每个品种每天一行。

    仅用于研究与对账，不能替代 ``price_data_research`` 作为账户回测主力。
    """
    cfg = data_config()
    collection = cfg.get("collections", {}).get("vendor_main_map", "vendor_main_map")
    key = str(product).upper()
    query = {"contract_object": key, "date": _date_query(start, end)}
    return _read_mongo_collection(
        collection, query, f"{key} 在 {start}~{end} 的供应商主力映射",
        VENDOR_MAIN_REQUIRED,
    )
