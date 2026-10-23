"""基于 contract_day_full 全合约日行情，逐日筛选主力合约与活跃近月合约。

主力合约：
  当日“领先合约”是 ``main_metric`` 指定指标全部为当日唯一最大值（且大于 0）的合约；
  默认 ``oi_turnover`` 要求持仓量 ``open_int`` 与成交额 ``turnover_value`` 都最大，
  两项最大值不是同一合约或存在并列时，当日没有领先合约。
  同一合约连续 ``confirm_days`` 个交易日领先才确认为主力。
  不回切：只能切换到到期日晚于当前（或上一个）主力的合约。
  当前主力当天没有行情时主力为空，直到出现新的确认合约。

活跃近月合约：
  当日候选为成交额 > ``turnover_threshold`` 且距到期 >= ``min_days_to_expiry`` 个自然日的合约中
  到期日最早者。持有合约出现以下任一情况时切换到当日候选：成交额连续 ``low_turnover_days``
  个交易日低于阈值；距到期不足 ``min_days_to_expiry`` 个自然日；当天没有行情。无候选时为空。
  不回切：候选的到期日不得早于已持有过的最晚到期日（下限），下限在持仓为空期间保留；
  下限以内的候选被跳过，取满足下限的到期最早合约。

结果是 T 日收盘后的决定，T+1 生效；本模块不做滞后。
成交额为空且成交量为 0 的行视为无成交，成交额记为 0（load_data1.csv 在 2014-11-20 前把无成交写成空值）；
成交量非 0 或也为空时成交额为空、以及持仓量或到期日为空，直接报错。
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import pandas as pd

KEY = ["contract_object", "date"]
REQUIRED_FIELDS = ("contract_object", "date", "contract_id", "code", "expiry", "open_int", "turnover_value", "volume")
OPTIONAL_FIELDS = ("vendor_main", "vendor_second_main", "settle", "close")
MAIN_METRICS = {
    "oi_turnover": ("open_int", "turnover_value"),
    "open_int": ("open_int",),
    "turnover_value": ("turnover_value",),
}


@dataclass(frozen=True)
class SelectionRule:
    main_metric: str = "oi_turnover"
    confirm_days: int = 3
    turnover_threshold: float = 20_000_000.0
    low_turnover_days: int = 3
    min_days_to_expiry: int = 30

    def __post_init__(self) -> None:
        if self.main_metric not in MAIN_METRICS:
            raise ValueError(f"未知主力指标 {self.main_metric}，可选 {sorted(MAIN_METRICS)}")
        if self.confirm_days < 1 or self.low_turnover_days < 1:
            raise ValueError("confirm_days 与 low_turnover_days 必须至少为 1")

    def metadata(self) -> dict[str, Any]:
        return {
            **asdict(self),
            "main_fields": list(MAIN_METRICS[self.main_metric]),
            "main_rule": (
                f"{'+'.join(MAIN_METRICS[self.main_metric])} 当日唯一最大，连续 {self.confirm_days} 日确认，不回切"
            ),
            "active_rule": (
                f"turnover_value > {self.turnover_threshold:.0f} 且距到期 >= {self.min_days_to_expiry} 自然日中到期最早；"
                f"成交额连续 {self.low_turnover_days} 日 < 阈值、距到期 < {self.min_days_to_expiry} 日或无行情时切换；"
                "不回切：候选到期日不早于已持有过的最晚到期日"
            ),
            "turnover_unit": "元",
            "days_to_expiry": "expiry - date，自然日",
            "timing": "T 日收盘后决定，T+1 生效",
        }


def prepare_contract_frame(frame: pd.DataFrame) -> pd.DataFrame:
    missing = sorted(set(REQUIRED_FIELDS) - set(frame.columns))
    if missing:
        raise ValueError(f"合约筛选缺少字段: {missing}")
    data = frame[list(REQUIRED_FIELDS) + [c for c in OPTIONAL_FIELDS if c in frame.columns]].copy()
    data["turnover_filled_zero"] = data["turnover_value"].isna() & data["volume"].eq(0)
    data.loc[data["turnover_filled_zero"], "turnover_value"] = 0
    for col in ("contract_id", "expiry", "open_int", "turnover_value"):
        nulls = data[col].isna()
        if nulls.any():
            sample = data.loc[nulls, ["contract_object", "date", "contract_id", "volume"]].head(5).to_dict("records")
            raise ValueError(f"{col} 有 {int(nulls.sum())} 行为空，例如 {sample}")
    data["date"] = pd.to_datetime(data["date"])
    data["expiry"] = pd.to_datetime(data["expiry"])
    data["contract_object"] = data["contract_object"].astype(str).str.upper()
    if data.duplicated(["contract_id", "date"]).any():
        raise ValueError("(contract_id, date) 存在重复行")
    # 同一编号可对应先后两个合约（如 FU 2018 年修订规格后重新挂牌 FU1901），按编号+到期日区分。
    data["contract_key"] = data["contract_id"] + "@" + data["expiry"].dt.strftime("%Y-%m-%d")
    data["days_to_expiry"] = (data["expiry"] - data["date"]).dt.days
    return data.sort_values(KEY + ["expiry"]).reset_index(drop=True)


def daily_leaders(data: pd.DataFrame, metric: str) -> pd.Series:
    """每个 (品种, 日期) 的领先合约；没有唯一领先合约的日子不出现在结果中。"""
    tops = []
    for col in MAIN_METRICS[metric]:
        peak = data.groupby(KEY)[col].transform("max")
        hit = data[col].eq(peak) & peak.gt(0)
        unique = hit.groupby([data["contract_object"], data["date"]]).transform("sum").eq(1)
        tops.append(data.loc[hit & unique].set_index(KEY)["contract_key"].rename(col))
    winners = pd.concat(tops, axis=1, join="inner")
    agree = winners.nunique(axis=1).eq(1)
    return winners.loc[agree].iloc[:, 0]


def eligible_active_candidates(data: pd.DataFrame, rule: SelectionRule) -> dict[tuple[str, pd.Timestamp], list[tuple[pd.Timestamp, str]]]:
    """每个 (品种, 日期) 满足成交额与到期天数条件的全部合约，按到期日从早到晚排列。"""
    eligible = data["turnover_value"].gt(rule.turnover_threshold) & data["days_to_expiry"].ge(rule.min_days_to_expiry)
    out: dict[tuple[str, pd.Timestamp], list[tuple[pd.Timestamp, str]]] = {}
    picked = data.loc[eligible].sort_values(["contract_object", "date", "expiry"])
    for row in picked.itertuples(index=False):
        out.setdefault((row.contract_object, row.date), []).append((row.expiry, row.contract_key))
    return out


def select_contracts(frame: pd.DataFrame, rule: SelectionRule | None = None) -> pd.DataFrame:
    """逐日输出主力合约与活跃近月合约，每个品种每天一行。"""
    rule = rule or SelectionRule()
    data = prepare_contract_frame(frame)
    leaders = daily_leaders(data, rule.main_metric).to_dict()
    candidates = eligible_active_candidates(data, rule)
    contracts = data.drop_duplicates("contract_key").set_index("contract_key")
    expiry_of = contracts["expiry"].to_dict()
    code_of = contracts["code"].to_dict()
    cid_of = contracts["contract_id"].to_dict()
    vendor = (
        data.loc[data["vendor_main"].eq(1)].set_index(KEY)["contract_key"].to_dict()
        if "vendor_main" in data.columns else {}
    )
    vendor_second = (
        data.loc[data["vendor_second_main"].eq(1)].set_index(KEY)["contract_key"].to_dict()
        if "vendor_second_main" in data.columns else {}
    )

    rows: list[dict[str, Any]] = []
    for product, group in data.groupby("contract_object", sort=True):
        main = main_expiry = streak_id = None
        streak = 0
        active = None
        active_floor = None  # 已持有过的最晚到期日：活跃近月合约不得切回到期更早的合约
        low = 0
        for date, day in group.groupby("date", sort=True):
            quotes = dict(zip(day["contract_key"], zip(day["turnover_value"], day["days_to_expiry"])))
            leader = leaders.get((product, date))
            streak = streak + 1 if leader is not None and leader == streak_id else int(leader is not None)
            streak_id = leader

            main_event = "hold" if main is not None else "none"
            if main is not None and main not in quotes:
                main, main_event = None, "missing"
            if (leader is not None and streak >= rule.confirm_days and leader != main
                    and (main_expiry is None or expiry_of[leader] > main_expiry)):
                main_event = "confirm" if main_expiry is None else "switch"
                main, main_expiry = leader, expiry_of[leader]

            reason = None
            if active is not None:
                if active not in quotes:
                    reason = "missing"
                else:
                    turnover, dte = quotes[active]
                    low = low + 1 if turnover < rule.turnover_threshold else 0
                    if dte < rule.min_days_to_expiry:
                        reason = "expiry"
                    elif low >= rule.low_turnover_days:
                        reason = "low_turnover"
            active_event = "hold"
            rollback = False
            if active is None or reason is not None:
                previous = active
                if previous is not None:
                    active_floor = expiry_of[previous]
                active = next(
                    (key for expiry, key in candidates.get((product, date), [])
                     if active_floor is None or expiry >= active_floor),
                    None,
                )
                low = 0
                if active is None:
                    active_event = "no_candidate" if reason is None else f"drop_{reason}"
                elif previous is None:
                    active_event = "init" if active_floor is None else "reenter"
                else:
                    active_event = f"switch_{reason}"
                rollback = active is not None and active_floor is not None and expiry_of[active] < active_floor

            # 活跃次近月：同日满足条件、到期日晚于活跃近月的最早合约
            active2 = None
            if active is not None:
                active2 = next(
                    (key for expiry, key in candidates.get((product, date), []) if expiry > expiry_of[active]),
                    None,
                )

            rows.append({
                "main_key": main,
                "active_key": active,
                "active2_key": active2,
                "active2_contract_id": cid_of.get(active2),
                "active2_expiry": expiry_of.get(active2),
                "vendor_second_key": vendor_second.get((product, date)),
                "contract_object": product,
                "date": date,
                "main_contract_id": cid_of.get(main),
                "main_code": code_of.get(main),
                "main_expiry": expiry_of.get(main),
                "main_event": main_event,
                "leader_contract_id": cid_of.get(leader),
                "leader_streak": streak,
                "active_contract_id": cid_of.get(active),
                "active_code": code_of.get(active),
                "active_expiry": expiry_of.get(active),
                "active_event": active_event,
                "active_rollback": rollback,
                "active_low_turnover_days": low,
                "active_days_to_expiry": quotes[active][1] if active is not None else None,
                "vendor_main_contract_id": cid_of.get(vendor.get((product, date))),
                "turnover_filled_zero_rows": int(day["turnover_filled_zero"].sum()),
            })
    return pd.DataFrame(rows)


def selection_summary(selection: pd.DataFrame) -> dict[str, Any]:
    """按品种统计覆盖率、切换次数、回切次数，以及与供应商主力的一致率。"""
    out: dict[str, Any] = {}
    for product, g in selection.groupby("contract_object", sort=True):
        has_vendor = g["vendor_main_contract_id"].notna()
        has_main = g["main_contract_id"].notna()
        both = has_vendor & has_main
        out[product] = {
            "days": int(len(g)),
            "start": g["date"].min().strftime("%Y-%m-%d"),
            "end": g["date"].max().strftime("%Y-%m-%d"),
            "main_coverage": round(float(has_main.mean()), 4),
            "main_switches": int(g["main_event"].eq("switch").sum()),
            "main_missing_events": int(g["main_event"].eq("missing").sum()),
            "main_vs_vendor_agree": (
                round(float((g.loc[both, "main_contract_id"] == g.loc[both, "vendor_main_contract_id"]).mean()), 4)
                if both.any() else None
            ),
            "active_coverage": round(float(g["active_contract_id"].notna().mean()), 4),
            "active_events": {k: int(v) for k, v in g["active_event"].value_counts().items() if k != "hold"},
            "active_rollbacks": int(g["active_rollback"].sum()),
            "main_active_same": round(float((g["main_contract_id"] == g["active_contract_id"]).mean()), 4),
            "turnover_filled_zero_rows": int(g["turnover_filled_zero_rows"].sum()),
        }
    return out
