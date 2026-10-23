import pandas as pd

import pytest

from data.main_series import apply_research_main, main_turnover_matrix

D1, D2 = pd.Timestamp("2020-01-02"), pd.Timestamp("2020-01-03")


def _prepared():
    return pd.DataFrame([
        {"contract_object": "A", "date": D1, "code": "a2005", "expiry": "2020-05-15",
         "contract_key": "A2005@2020-05-15", "contract_id": "A2005"},
        {"contract_object": "A", "date": D1, "code": "a2009", "expiry": "2020-09-15",
         "contract_key": "A2009@2020-09-15", "contract_id": "A2009"},
        {"contract_object": "A", "date": D2, "code": "a2009", "expiry": "2020-09-15",
         "contract_key": "A2009@2020-09-15", "contract_id": "A2009"},
    ])


def _selection():
    # 选择表自己推导的主力为 A2005（与研究序列不同），用于验证会被覆盖
    return pd.DataFrame([
        {"contract_object": "A", "date": D1, "main_key": "A2005@2020-05-15",
         "main_expiry": "2020-05-15", "main_contract_id": "A2005", "active_key": "A2009@2020-09-15"},
        {"contract_object": "A", "date": D2, "main_key": "A2005@2020-05-15",
         "main_expiry": "2020-05-15", "main_contract_id": "A2005", "active_key": "A2009@2020-09-15"},
    ])


def test_research_main_overrides_selection_main_and_keeps_other_columns():
    research = pd.DataFrame([
        {"contract_object": "A", "date": D1, "code": "a2009", "expiry": "2020-09-15"},
        {"contract_object": "A", "date": D2, "code": "a2009", "expiry": "2020-09-15"},
    ])
    out, stats = apply_research_main(_selection(), _prepared(), research)
    assert out["main_key"].tolist() == ["A2009@2020-09-15", "A2009@2020-09-15"]
    assert out["main_contract_id"].tolist() == ["A2009", "A2009"]
    assert out["main_expiry"].tolist() == ["2020-09-15", "2020-09-15"]
    assert out["active_key"].tolist() == ["A2009@2020-09-15", "A2009@2020-09-15"]
    assert stats == {"research_rows": 2, "unmatched_rows": 0, "selection_rows_main_missing": 0}


def test_unmatched_research_contract_is_null_not_filled():
    # 研究序列指向 contract_day_full 中不存在的合约：主力必须为空，不能回退到选择表的主力
    research = pd.DataFrame([
        {"contract_object": "A", "date": D1, "code": "a2099", "expiry": "2020-11-15"},
        {"contract_object": "A", "date": D2, "code": "a2009", "expiry": "2020-09-15"},
    ])
    out, stats = apply_research_main(_selection(), _prepared(), research)
    assert pd.isna(out.loc[0, "main_key"])
    assert out.loc[1, "main_key"] == "A2009@2020-09-15"
    assert stats["unmatched_rows"] == 1
    assert stats["selection_rows_main_missing"] == 1


def test_main_turnover_uses_the_single_research_contract():
    research = pd.DataFrame([
        {"contract_object": "A", "date": D1, "turnover_value": 1.2e9},
        {"contract_object": "B", "date": D1, "turnover_value": 4e8},
    ])
    wide = main_turnover_matrix(research, ["A", "B", "C"])
    assert wide.loc[D1, "A"] == pytest.approx(1.2e9)
    assert wide.loc[D1, "B"] == pytest.approx(4e8)
    assert pd.isna(wide.loc[D1, "C"])


def test_main_turnover_rejects_missing_field():
    research = pd.DataFrame([{"contract_object": "A", "date": D1, "code": "a2005"}])
    with pytest.raises(ValueError, match="trade_turnover_value"):
        main_turnover_matrix(research, ["A"])


def test_expiry_distinguishes_same_code_relisted():
    # 同一 code 在不同到期日（例如 FU 重新上市）必须按 (code, expiry) 区分
    prepared = pd.DataFrame([
        {"contract_object": "A", "date": D1, "code": "a2005", "expiry": "2020-05-15",
         "contract_key": "A2005@2020-05-15", "contract_id": "A2005"},
        {"contract_object": "A", "date": D1, "code": "a2005", "expiry": "2020-05-20",
         "contract_key": "A2005@2020-05-20", "contract_id": "A2005"},
    ])
    research = pd.DataFrame([{"contract_object": "A", "date": D1, "code": "a2005", "expiry": "2020-05-20"}])
    out, _ = apply_research_main(_selection().iloc[:1], prepared, research)
    assert out.loc[0, "main_key"] == "A2005@2020-05-20"
