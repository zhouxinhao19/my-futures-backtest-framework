import pandas as pd
import pytest

from data.contract_selection import SelectionRule, select_contracts, selection_summary

EXPIRY = {"SR2003": "2020-03-13", "SR2005": "2020-05-15", "SR2009": "2020-09-14"}


def _frame(days):
    """days: 每天一个 {contract_id: (open_int, turnover_value)}，从 2020-01-02 起按工作日排列。"""
    rows = []
    for date, quotes in zip(pd.bdate_range("2020-01-02", periods=len(days)), days):
        for cid, (oi, tv) in quotes.items():
            rows.append({
                "contract_object": "SR", "date": date, "contract_id": cid, "code": cid.lower(),
                "expiry": EXPIRY[cid], "open_int": oi, "turnover_value": tv, "volume": tv,
            })
    return pd.DataFrame(rows)


def test_main_requires_three_consecutive_leading_days():
    lead_a = {"SR2003": (100, 100), "SR2005": (50, 50)}
    out = select_contracts(_frame([lead_a] * 4))
    assert out["main_contract_id"].tolist() == [None, None, "SR2003", "SR2003"]
    assert out["main_event"].tolist() == ["none", "none", "confirm", "hold"]


def test_main_switches_forward_and_never_rolls_back():
    lead_a = {"SR2003": (100, 100), "SR2005": (50, 50)}
    lead_b = {"SR2003": (50, 50), "SR2005": (100, 100)}
    out = select_contracts(_frame([lead_a] * 3 + [lead_b] * 3 + [lead_a] * 3))
    assert out["main_contract_id"].tolist() == [
        None, None, "SR2003", "SR2003", "SR2003", "SR2005", "SR2005", "SR2005", "SR2005"]
    assert out.loc[5, "main_event"] == "switch"


def test_main_needs_open_int_and_turnover_on_same_contract():
    split = {"SR2003": (100, 50), "SR2005": (50, 100)}
    both = select_contracts(_frame([split] * 3))
    assert both["leader_contract_id"].isna().all()
    assert both["main_contract_id"].isna().all()
    oi_only = select_contracts(_frame([split] * 3), SelectionRule(main_metric="open_int"))
    assert oi_only["main_contract_id"].tolist()[-1] == "SR2003"


def test_main_ties_have_no_leader():
    tie = {"SR2003": (100, 100), "SR2005": (100, 100)}
    out = select_contracts(_frame([tie] * 3))
    assert out["leader_contract_id"].isna().all()


def test_main_is_empty_when_current_contract_has_no_quote():
    lead_a = {"SR2003": (100, 100), "SR2005": (50, 50)}
    only_b = {"SR2005": (50, 50)}
    out = select_contracts(_frame([lead_a] * 3 + [only_b]))
    assert out.loc[3, "main_contract_id"] is None
    assert out.loc[3, "main_event"] == "missing"


RULE = SelectionRule(turnover_threshold=100, low_turnover_days=3, min_days_to_expiry=30)


def test_active_picks_earliest_liquid_contract_and_switches_after_three_low_days():
    liquid = {"SR2003": (10, 200), "SR2005": (10, 300)}
    low = {"SR2003": (10, 50), "SR2005": (10, 300)}
    out = select_contracts(_frame([liquid, low, low, low]), RULE)
    assert out["active_contract_id"].tolist() == ["SR2003", "SR2003", "SR2003", "SR2005"]
    assert out["active_event"].tolist() == ["init", "hold", "hold", "switch_low_turnover"]
    assert not out["active_rollback"].any()


def test_active_low_turnover_streak_resets_on_liquid_day():
    liquid = {"SR2003": (10, 200), "SR2005": (10, 300)}
    low = {"SR2003": (10, 50), "SR2005": (10, 300)}
    out = select_contracts(_frame([liquid, low, low, liquid, low, low]), RULE)
    assert out["active_contract_id"].eq("SR2003").all()


def test_active_switches_when_days_to_expiry_below_threshold():
    days = [{"SR2003": (10, 200), "SR2005": (10, 300)}] * 35
    out = select_contracts(_frame(days), RULE)
    first_switch = out.index[out["active_event"].eq("switch_expiry")][0]
    assert (pd.Timestamp(EXPIRY["SR2003"]) - out.loc[first_switch, "date"]).days < 30
    assert (pd.Timestamp(EXPIRY["SR2003"]) - out.loc[first_switch - 1, "date"]).days >= 30
    assert out.loc[first_switch, "active_contract_id"] == "SR2005"


def test_active_does_not_roll_back_to_earlier_expiry():
    # SR2003 最初为活跃近月；低流动性后切到 SR2009；此后 SR2003 再次变为最早的流动合约，但不得切回。
    first = {"SR2003": (10, 200), "SR2009": (10, 300)}
    low_front = {"SR2003": (10, 50), "SR2009": (10, 300)}
    back = {"SR2003": (10, 300), "SR2009": (10, 300)}
    out = select_contracts(_frame([first, low_front, low_front, low_front, back, back]), RULE)
    assert out["active_contract_id"].tolist() == ["SR2003", "SR2003", "SR2003", "SR2009", "SR2009", "SR2009"]
    assert not out["active_rollback"].any()


def test_active_reenters_after_drop_only_at_or_after_floor():
    # 先持有 SR2009（SR2003 低流动），随后 SR2009 低流动且无候选而被丢弃；
    # SR2003 恢复流动后到期早于下限，不得重新进入，只能进入 SR2009 或更晚合约。
    start = {"SR2003": (10, 50), "SR2009": (10, 300)}
    dead = {"SR2003": (10, 50), "SR2009": (10, 50)}
    back = {"SR2003": (10, 300), "SR2009": (10, 300)}
    out = select_contracts(_frame([start, dead, dead, dead, back]), RULE)
    assert out["active_contract_id"].tolist()[:1] == ["SR2009"]
    assert out.loc[4, "active_contract_id"] == "SR2009"
    assert out.loc[4, "active_event"] == "reenter"
    assert not out["active_rollback"].any()


def test_active_skips_contracts_within_thirty_days_of_expiry():
    late = pd.bdate_range("2020-02-20", periods=1)[0]
    frame = pd.DataFrame([
        {"contract_object": "SR", "date": late, "contract_id": "SR2003", "code": "sr2003",
         "expiry": EXPIRY["SR2003"], "open_int": 10, "turnover_value": 500, "volume": 5},
        {"contract_object": "SR", "date": late, "contract_id": "SR2005", "code": "sr2005",
         "expiry": EXPIRY["SR2005"], "open_int": 10, "turnover_value": 300, "volume": 3},
    ])
    assert select_contracts(frame, RULE)["active_contract_id"].tolist() == ["SR2005"]


def test_active_is_empty_without_candidate():
    out = select_contracts(_frame([{"SR2003": (10, 50), "SR2005": (10, 60)}]), RULE)
    assert out["active_contract_id"].isna().all()
    assert out["active_event"].tolist() == ["no_candidate"]


def test_null_turnover_with_zero_volume_counts_as_no_trade():
    frame = _frame([{"SR2003": (10, 200), "SR2005": (10, 300)}])
    frame.loc[0, ["turnover_value", "volume"]] = [None, 0]
    out = select_contracts(frame, RULE)
    assert out["active_contract_id"].tolist() == ["SR2005"]
    assert out["turnover_filled_zero_rows"].tolist() == [1]


@pytest.mark.parametrize("volume", [5, None])
def test_null_turnover_with_nonzero_or_missing_volume_raises(volume):
    frame = _frame([{"SR2003": (10, 200), "SR2005": (10, 300)}])
    frame["volume"] = frame["volume"].astype(object)
    frame.loc[0, ["turnover_value", "volume"]] = [None, volume]
    with pytest.raises(ValueError, match="turnover_value 有 1 行为空"):
        select_contracts(frame)


def test_relisted_contract_id_with_new_expiry_is_a_different_contract():
    old = pd.bdate_range("2018-06-25", periods=4)
    new = pd.bdate_range("2018-07-16", periods=4)
    rows = []
    for dates, expiry, oi in ((old, "2018-06-28", 100), (new, "2018-12-28", 100)):
        for date in dates:
            rows.append({"contract_object": "FU", "date": date, "contract_id": "FU1901", "code": "fu901",
                         "expiry": expiry, "open_int": oi, "turnover_value": oi, "volume": oi})
            rows.append({"contract_object": "FU", "date": date, "contract_id": "FU1909", "code": "fu909",
                         "expiry": "2019-08-30", "open_int": 1, "turnover_value": 1, "volume": 1})
    out = select_contracts(pd.DataFrame(rows))
    assert out.loc[out["main_contract_id"].eq("FU1901"), "main_expiry"].nunique() == 2
    first_new = out.index[out["date"].eq(new[0])][0]
    assert out.loc[first_new, "main_event"] == "missing"
    assert out.loc[first_new + 2, "main_event"] == "switch"
    assert out.loc[first_new + 2, "main_expiry"] == pd.Timestamp("2018-12-28")


def test_summary_counts_switches_and_vendor_agreement():
    lead_a = {"SR2003": (100, 100), "SR2005": (50, 50)}
    frame = _frame([lead_a] * 3)
    frame["vendor_main"] = (frame["contract_id"] == "SR2003").astype(int)
    summary = selection_summary(select_contracts(frame))["SR"]
    assert summary["main_vs_vendor_agree"] == 1.0
    assert summary["main_coverage"] == pytest.approx(1 / 3, abs=1e-4)
