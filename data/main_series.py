"""主力合约序列：以 price_data_research 中每个品种每天唯一一条合约为准（项目规则）。

- 读取 price_data_research 的 contract_object、date、code、expiry；
- 与 contract_day_full 派生的 prepared 按 (品种, 日期, code, 到期日) 匹配，得到 contract_key；
- 匹配不上的日子主力记为空，不补值，并计入 unmatched_rows，由调用方写入 run.json；
- contract_selection 中自行推导的主力列（main_*）被这里的序列覆盖；活跃近月、次近月仍按 contract_selection 计算。
"""
from __future__ import annotations

import pandas as pd
from pymongo import MongoClient

from data import MONGO_DB, MONGO_URI, data_config


def load_research_main(products: list[str]) -> pd.DataFrame:
    """读取 price_data_research 的每日主力：contract_object、date、code、expiry。"""
    coll = data_config()["collections"]["price_panel"]
    docs = list(MongoClient(MONGO_URI)[MONGO_DB][coll].find(
        {"contract_object": {"$in": products}},
        {
            "_id": 0, "contract_object": 1, "date": 1, "code": 1, "expiry": 1,
            "adj_settle": 1, "trade_turnover_value": 1,
        },
    ))
    if not docs:
        raise SystemExit(f"{MONGO_DB}.{coll} 中没有主力数据，停止")
    out = pd.DataFrame(docs)
    if "trade_turnover_value" not in out.columns or out["trade_turnover_value"].isna().all():
        raise ValueError(
            f"{MONGO_DB}.{coll} 缺少主力成交额 trade_turnover_value，停止"
        )
    out["turnover_value"] = pd.to_numeric(out["trade_turnover_value"], errors="coerce")
    out["contract_object"] = out["contract_object"].astype(str).str.upper()
    out["date"] = pd.to_datetime(out["date"])
    out["code"] = out["code"].astype(str).str.lower()
    out["expiry"] = pd.to_datetime(out["expiry"]).dt.strftime("%Y-%m-%d")
    if out.duplicated(["contract_object", "date"]).any():
        raise SystemExit("price_data_research 同一品种同一天存在多条记录，停止")
    return out


def main_code_matrix(research: pd.DataFrame, products: list[str]) -> pd.DataFrame:
    """日期 × 品种的主力合约代码。换月由相邻两日代码不同识别，缺失不填充。"""
    if "code" not in research.columns:
        raise ValueError("主力序列缺少 code 字段")
    wide = research.pivot(index="date", columns="contract_object", values="code").sort_index()
    return wide.reindex(columns=products)


def main_turnover_matrix(research: pd.DataFrame, products: list[str]) -> pd.DataFrame:
    """日期 × 品种的主力成交额（price_data_research.trade_turnover_value，元）。

    每天每个品种只有一条合约，不与其他月份相加。缺失保持为空，不填充。
    """
    if "turnover_value" not in research.columns:
        raise ValueError("主力序列缺少 trade_turnover_value 字段")
    wide = research.pivot(index="date", columns="contract_object", values="turnover_value").sort_index()
    return wide.reindex(columns=products).astype(float)


def adj_settle_matrix(research: pd.DataFrame, products: list[str]) -> pd.DataFrame:
    """日期 × 品种 的主力后复权结算价（price_data_research.adj_settle）。缺失保持为空，不填充。

    后复权序列在换月处是连续的，因此同一品种的日收益可以直接由相邻两日的价格差得到。
    """
    if "adj_settle" not in research.columns:
        raise ValueError("主力序列缺少 adj_settle 字段")
    wide = research.pivot(index="date", columns="contract_object", values="adj_settle").sort_index()
    return wide.reindex(columns=products).astype(float)


def apply_research_main(selection: pd.DataFrame, prepared: pd.DataFrame, research: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """用 price_data_research 的主力覆盖 selection 中的 main_key / main_expiry / main_contract_id。

    返回 (新的 selection, 质量统计)。质量统计包括：
    - research_rows：price_data_research 的行数；
    - unmatched_rows：在 contract_day_full 中找不到对应 (code, 到期日) 的行数（主力置空，不补值）。
    """
    lookup = prepared[["contract_object", "date", "code", "expiry", "contract_key", "contract_id"]].copy()
    lookup["code"] = lookup["code"].astype(str).str.lower()
    lookup["expiry"] = pd.to_datetime(lookup["expiry"]).dt.strftime("%Y-%m-%d")
    lookup["date"] = pd.to_datetime(lookup["date"])
    lookup = lookup.drop_duplicates(["contract_object", "date", "code", "expiry"])

    main = research.merge(
        lookup.rename(columns={"contract_key": "main_key", "contract_id": "main_contract_id"}),
        on=["contract_object", "date", "code", "expiry"], how="left", validate="one_to_one",
    )
    main["main_expiry"] = main["expiry"]
    unmatched = int(main["main_key"].isna().sum())
    main = main[["contract_object", "date", "main_key", "main_expiry", "main_contract_id"]]

    base = selection.drop(columns=["main_key", "main_expiry", "main_contract_id"], errors="ignore")
    out = base.merge(main, on=["contract_object", "date"], how="left", validate="one_to_one")
    stats = {"research_rows": int(len(research)), "unmatched_rows": unmatched,
             "selection_rows_main_missing": int(out["main_key"].isna().sum())}
    return out, stats
