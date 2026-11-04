"""流动性筛选。

正式回测使用当天那一条主力合约：过去 20 个交易日
``research_panel.parquet`` 的 trade_turnover_value 均值不低于 10 亿元。
面板里这个字段名叫 ``turnover_value``。不合格日把因子置空，不参与排序，
也不新开仓。重叠 cohort 已经建立的仓位仍持有到 H 日，不因中途不合格而强平。

``product_turnover`` 仍是全合约成交额合计，只留给对照冻结脚本。
正式入口不要再调用它。
"""
from __future__ import annotations

import pandas as pd

WINDOW = 20


def smooth_by_product(factor: pd.DataFrame, window: int) -> pd.DataFrame:
    """每个品种在自身交易日历上做 R 日滚动均值；窗口内任一日缺失则为空。"""
    if window < 1:
        raise ValueError("window 必须至少为 1")
    out = pd.DataFrame(index=factor.index, columns=factor.columns, dtype=float)
    for col in factor.columns:
        s = factor[col].dropna()
        out.loc[s.index, col] = s.rolling(window, min_periods=window).mean().to_numpy()
    return out

MIN_TURNOVER = 1_000_000_000.0  # 10 亿元
RESEARCH_MAIN_RULE = (
    "过去20个交易日，research_panel.parquet 当天主力合约 "
    "trade_turnover_value 的均值不低于10亿元；"
    "不合格日因子置空，不参与排序，也不新开仓。"
    "重叠 cohort 已有仓位持有到 H 日。"
)


def product_turnover(prepared: pd.DataFrame) -> pd.DataFrame:
    """日期 × 品种 的成交额合计（元）。"""
    if "turnover_value" not in prepared.columns:
        raise ValueError("流动性筛选需要 turnover_value 字段")
    return (prepared.groupby(["date", "contract_object"])["turnover_value"].sum()
            .unstack("contract_object").sort_index())


def liquidity_mask(turnover: pd.DataFrame, window: int = WINDOW, threshold: float = MIN_TURNOVER) -> pd.DataFrame:
    """日期 × 品种 的布尔掩码：过去 window 日均成交额 >= threshold。"""
    avg = smooth_by_product(turnover, window)
    return avg.ge(threshold).fillna(False).astype(bool)


def apply_mask(factor: pd.DataFrame, mask: pd.DataFrame) -> pd.DataFrame:
    """不合格的 (日期, 品种) 因子值置空。掩码缺失的位置视为不合格。"""
    aligned = mask.reindex(index=factor.index, columns=factor.columns).fillna(False).astype(bool)
    return factor.where(aligned)


def research_turnover_matrix(panel: pd.DataFrame) -> pd.DataFrame:
    """日期 × 品种的主力成交额。每天每个品种只接受一条记录。"""
    required = {"date", "symbol", "turnover_value"}
    missing = sorted(required - set(panel.columns))
    if missing:
        raise ValueError(
            "主力流动性筛选需要 research_panel.parquet 的 trade_turnover_value，"
            f"面板缺少字段: {missing}"
        )
    frame = panel.loc[:, ["date", "symbol", "turnover_value"]].copy()
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce").dt.normalize()
    frame["symbol"] = frame["symbol"].astype(str)
    frame["turnover_value"] = pd.to_numeric(frame["turnover_value"], errors="coerce")
    if frame["date"].isna().any() or frame["symbol"].eq("").any():
        raise ValueError("主力成交额存在空的日期或品种")
    if not frame["turnover_value"].notna().any():
        raise ValueError(
            "research_panel.parquet 的 trade_turnover_value 全部为空，不能做流动性筛选"
        )
    if frame.duplicated(["date", "symbol"]).any():
        raise ValueError("research_panel.parquet 同一品种同一天有多条成交额")
    wide = frame.pivot(index="date", columns="symbol", values="turnover_value").sort_index()
    wide.index.name = "date"
    wide.columns.name = "symbol"
    return wide


def research_main_liquidity_mask(
    panel: pd.DataFrame,
    columns: list[str] | None = None,
    window: int = WINDOW,
    threshold: float = MIN_TURNOVER,
) -> pd.DataFrame:
    """正式回测的主力成交额掩码。缺失品种视为不合格。"""
    wide = research_turnover_matrix(panel)
    if columns is not None:
        wide = wide.reindex(columns=list(columns))
    return liquidity_mask(wide, window, threshold)


def attach_research_turnover(frame: pd.DataFrame, panel: pd.DataFrame) -> pd.DataFrame:
    """因子表没有成交额时，按日期和品种从研究面板补上，不填充缺失值。"""
    if "turnover_value" in frame.columns:
        return frame
    turnover = research_turnover_matrix(panel).stack().rename("turnover_value").reset_index()
    out = frame.copy()
    out["_liq_date"] = pd.to_datetime(out["date"], errors="coerce").dt.normalize()
    out["_liq_symbol"] = out["symbol"].astype(str)
    out = out.merge(
        turnover.rename(columns={"date": "_liq_date", "symbol": "_liq_symbol"}),
        on=["_liq_date", "_liq_symbol"],
        how="left",
    )
    return out.drop(columns=["_liq_date", "_liq_symbol"])


def blank_long_signals(
    frame: pd.DataFrame,
    value_columns: list[str],
    *,
    date_col: str = "date",
    symbol_col: str = "symbol",
) -> pd.DataFrame:
    """长表上把不合格日的信号列置空。调用前长表必须已有 turnover_value。"""
    missing = [column for column in value_columns if column not in frame.columns]
    if missing:
        raise ValueError(f"缺少待筛选列: {missing}")
    renamed = frame.rename(columns={date_col: "date", symbol_col: "symbol"})
    mask = research_main_liquidity_mask(renamed)
    eligible = mask.stack().rename("_eligible").reset_index()
    out = frame.copy()
    out["_liq_date"] = pd.to_datetime(out[date_col], errors="coerce").dt.normalize()
    out["_liq_symbol"] = out[symbol_col].astype(str)
    out = out.merge(
        eligible.rename(columns={"date": "_liq_date", "symbol": "_liq_symbol"}),
        on=["_liq_date", "_liq_symbol"],
        how="left",
    )
    keep = out["_eligible"].fillna(False).astype(bool)
    for column in value_columns:
        out[column] = out[column].where(keep)
    return out.drop(columns=["_liq_date", "_liq_symbol", "_eligible"])


def main_liquidity_eligible(frame: pd.DataFrame) -> pd.Series:
    """与输入行对齐的主力流动性布尔值。没有成交额记录的行不合格。"""
    mask = research_main_liquidity_mask(frame)
    eligible = mask.stack().rename("_eligible").reset_index()
    keys = pd.DataFrame({
        "date": pd.to_datetime(frame["date"], errors="coerce").dt.normalize(),
        "symbol": frame["symbol"].astype(str),
    })
    merged = keys.merge(eligible, on=["date", "symbol"], how="left")
    result = merged["_eligible"].fillna(False).astype(bool)
    result.index = frame.index
    return result
