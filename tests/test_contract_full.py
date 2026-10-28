import pandas as pd
import pytest

import data as data_mod
from data.contract_full import (
    CSV_REQUIRED,
    EPOCH,
    apply_research_labels,
    build_full_frame,
    build_research_frame,
    build_vendor_main_frame,
    parse_contract_id,
    to_full_documents,
    to_main_documents,
)


def _stamp(date: str) -> int:
    return int((pd.Timestamp(date) - EPOCH).total_seconds())


def _row(date, code, ltd, product="SR", main=0, second=0, close=100.0, volume=10.0):
    base = {col: None for col in CSV_REQUIRED}
    base.update({
        "SECURITY_ID": 1, "TICKER_SYMBOL": code, "EXCHANGE_CD": "XZCE", "SEC_SHORT_NAME": "白糖",
        "TRADE_DATE": date, "CONTRACT_OBJECT": product, "CONTRACT_MARK": "L0",
        "PRE_SETTL_PRICE": 100.0, "PRE_CLOSE_PRICE": 100.0, "OPEN_PRICE": 100.0,
        "HIGHEST_PRICE": 101.0, "LOWEST_PRICE": 99.0, "SETTL_PRICE": 100.0,
        "CLOSE_PRICE": close, "TURNOVER_VOL": volume, "TURNOVER_VALUE": 1000.0,
        "OPEN_INT": 50.0, "CHG": 0.0, "CHG1": 0.0, "CHG_PCT": 0.0,
        "MAINCON": main, "SMAINCON": second, "LAST_TRADE_DATE": ltd,
        "UPDATE_TIME": "2019-12-31", "TMSTAMP": 162642458568,
    })
    return base


def _raw(rows):
    return pd.DataFrame(rows, columns=list(CSV_REQUIRED))


def _two_contract_csv():
    """2014-01-02 起的 SR 两条合约：sr401 为主力，sr501 为次主力，三天数据。"""
    rows = []
    for date in ("2014-01-02", "2014-01-03", "2014-01-06"):
        rows.append(_row(date, "sr401", "2014-01-15", main=1))
        rows.append(_row(date, "sr501", "2014-05-15", second=1))
    return _raw(rows)


def _research(docs):
    out = []
    for contract_object, date, code, expiry in docs:
        out.append({
            "contract_object": contract_object, "date": date, "code": code,
            "date_stamp": _stamp(date), "expiry": expiry,
            "close": 100.0, "settle": 100.0, "trade_open_int": 50, "trade_turnover_vol": 10,
        })
    return out


def test_parse_disambiguates_three_digit_codes_by_expiry_year():
    assert parse_contract_id("sr401", "SR", "2014-01-15") == "SR1401"
    assert parse_contract_id("sr401", "SR", "2024-01-15") == "SR2401"
    assert parse_contract_id("a2401", "A", "2024-01-15") == "A2401"
    assert parse_contract_id("ih312", "IH", "2023-12-15") == "IH2312"


ALIASES = {
    "MA": [{"prefix": "me", "valid_until": "2015-05-15"}],
    "ZC": [{"prefix": "tc", "valid_until": "2016-04-08"}],
}


def test_parse_accepts_alias_only_within_valid_window():
    assert parse_contract_id("me405", "MA", "2014-05-15", "2014-05-15", ALIASES) == "MA1405"
    assert parse_contract_id("ma405", "MA", "2014-05-15", "2014-05-15", ALIASES) == "MA1405"
    assert parse_contract_id("ma405", "MA", "2014-05-15", "2016-01-04", ALIASES) == "MA1405"
    assert parse_contract_id("tc401", "ZC", "2014-01-15", "2014-01-02", ALIASES) == "ZC1401"
    # 别名在有效期后出现：报错，不静默接受
    with pytest.raises(ValueError, match="前缀不在允许范围"):
        parse_contract_id("me405", "MA", "2014-05-15", "2015-06-01", ALIASES)
    # 未在配置中登记的别名：报错
    with pytest.raises(ValueError):
        parse_contract_id("me405", "ZC", "2014-05-15", "2014-05-15", ALIASES)


def test_parse_without_alias_config_rejects_alias_prefix():
    with pytest.raises(ValueError):
        parse_contract_id("me405", "MA", "2014-05-15", "2014-05-15", None)


def test_alias_requires_trade_date():
    with pytest.raises(ValueError, match="必须提供交易日"):
        parse_contract_id("me405", "MA", "2014-05-15", None, ALIASES)


def test_full_frame_applies_alias_per_row_date():
    rows = [
        _row("2015-05-15", "me505", "2015-05-15", product="MA", main=1),
        _row("2015-05-18", "ma509", "2015-09-15", product="MA", main=1),
    ]
    frame = build_full_frame(_raw(rows), ALIASES)
    assert frame["contract_id"].tolist() == ["MA1505", "MA1509"]
    late = [_row("2015-06-01", "me505", "2015-05-15", product="MA", main=1)]
    with pytest.raises(ValueError, match="前缀不在允许范围"):
        build_full_frame(_raw(late), ALIASES)


def test_research_frame_applies_alias_per_row_date():
    docs = _research([("MA", "2014-12-08", "me505", "2015-05-15")])
    assert build_research_frame(docs, ALIASES)["research_cid"].tolist() == ["MA1505"]
    with pytest.raises(ValueError):
        build_research_frame(_research([("MA", "2015-06-01", "me505", "2015-05-15")]), ALIASES)


def test_parse_rejects_wrong_product_missing_expiry_and_bad_month():
    with pytest.raises(ValueError):
        parse_contract_id("sr401", "A", "2014-01-15")
    with pytest.raises(ValueError):
        parse_contract_id("sr401", "SR", None)
    with pytest.raises(ValueError):
        parse_contract_id("sr413", "SR", "2014-01-15")


def test_full_frame_keeps_nulls_as_none_and_flags_quality():
    rows = [_row("2014-01-02", "sr401", "2014-01-15", main=1, volume=0.0)]
    rows[0]["OPEN_PRICE"] = None
    rows[0]["CLOSE_PRICE"] = None
    frame = build_full_frame(_raw(rows))
    docs = to_full_documents(frame.assign(research_main=None, research_main_source=None,
                                          research_main_synced_at=None),
                             "src.csv", "abc", pd.Timestamp("2026-10-19").to_pydatetime())
    doc = docs[0]
    assert doc["open"] is None and doc["close"] is None
    assert doc["settle"] == 100.0
    assert doc["no_trade"] is True and doc["ohlc_missing"] is True
    assert doc["contract_id"] == "SR1401"
    assert doc["date_stamp"] == _stamp("2014-01-02")
    assert doc["volume"] == 0 and isinstance(doc["volume"], int)
    assert doc["vendor_main"] == 1 and doc["research_main"] is None


def test_missing_column_and_duplicate_rows_raise():
    raw = _two_contract_csv().drop(columns=["SETTL_PRICE"])
    with pytest.raises(ValueError, match="缺少字段"):
        build_full_frame(raw)
    dup = _raw([_row("2014-01-02", "sr401", "2014-01-15", main=1)] * 2)
    with pytest.raises(ValueError, match="重复"):
        build_full_frame(dup)


def test_vendor_flags_must_be_exactly_one_main_per_product_day():
    no_main = _raw([_row("2014-01-02", "sr401", "2014-01-15"), _row("2014-01-02", "sr501", "2014-05-15")])
    with pytest.raises(ValueError, match="MAINCON"):
        build_full_frame(no_main)
    two_main = _raw([
        _row("2014-01-02", "sr401", "2014-01-15", main=1),
        _row("2014-01-02", "sr501", "2014-05-15", main=1),
    ])
    with pytest.raises(ValueError, match="MAINCON"):
        build_full_frame(two_main)


def test_second_main_cannot_be_the_same_row_as_main():
    raw = _raw([_row("2014-01-02", "sr401", "2014-01-15", main=1, second=1)])
    with pytest.raises(ValueError, match="既是主力又是次主力"):
        build_full_frame(raw)


def test_research_label_has_three_states():
    full = build_full_frame(_two_contract_csv())
    research = build_research_frame(_research([
        ("SR", "2014-01-02", "sr401", "2014-01-15"),   # 匹配 -> 1 / 0
        ("SR", "2014-01-03", "sr901", "2019-09-15"),   # 合约不在 CSV 当天 -> None
    ]))
    labeled = apply_research_labels(full, research, "t").set_index(["date", "code"])
    assert labeled.loc[("2014-01-02", "sr401"), "research_main"] == 1
    assert labeled.loc[("2014-01-02", "sr501"), "research_main"] == 0
    assert labeled.loc[("2014-01-03", "sr401"), "research_main"] is None
    assert labeled.loc[("2014-01-03", "sr501"), "research_main"] is None
    assert labeled.loc[("2014-01-06", "sr401"), "research_main"] is None  # 研究表无记录


def test_research_match_uses_contract_id_not_bare_code():
    """同名 sr401 在研究表中是 2024 年合约，CSV 当天只有 2014 年的 sr401，应判为无法比对而不是匹配。"""
    full = build_full_frame(_two_contract_csv())
    research = build_research_frame(_research([("SR", "2014-01-02", "sr401", "2024-01-15")]))
    labeled = apply_research_labels(full, research, "t")
    day1 = labeled[labeled["date"] == "2014-01-02"]
    assert day1["research_main"].isna().all()


def test_research_rejects_duplicates_and_bad_date_stamp():
    docs = _research([("SR", "2014-01-02", "sr401", "2014-01-15"),
                      ("SR", "2014-01-02", "sr501", "2014-05-15")])
    with pytest.raises(ValueError, match="多条"):
        build_research_frame(docs)
    bad = _research([("SR", "2014-01-02", "sr401", "2014-01-15")])
    bad[0]["date_stamp"] = 1
    with pytest.raises(ValueError, match="date_stamp"):
        build_research_frame(bad)
    with pytest.raises(ValueError, match="没有记录"):
        build_research_frame([])


def test_vendor_main_frame_agreement_and_documents():
    full = build_full_frame(_two_contract_csv())
    research = build_research_frame(_research([
        ("SR", "2014-01-02", "sr401", "2014-01-15"),
        ("SR", "2014-01-03", "sr501", "2014-05-15"),   # 研究主力与供应商不同
    ]))
    day = build_vendor_main_frame(full, research).set_index("date")
    assert day.loc["2014-01-02", "main_agree"] == 1
    assert day.loc["2014-01-03", "main_agree"] == 0
    assert day.loc["2014-01-06", "main_agree"] is None
    assert pd.isna(day.loc["2014-01-06", "research_main_contract_id"])
    assert day.loc["2014-01-02", "second_contract"] == "sr501"

    docs = to_main_documents(build_vendor_main_frame(full, research), "src.csv", "now")
    first = docs[0]
    assert first["rule"] == "vendor_maincon"
    assert first["main_contract_id"] == "SR1401"
    assert first["date_stamp"] == _stamp(first["date"])


def test_loader_raises_on_empty_collection(monkeypatch, tmp_path):
    empty = pd.DataFrame(columns=["contract_object", "date", "date_stamp"])
    empty.to_parquet(tmp_path / "contract_day_full.parquet", index=False)
    empty.to_parquet(tmp_path / "vendor_main_map.parquet", index=False)
    monkeypatch.setattr(data_mod, "parquet_root", lambda: tmp_path)
    with pytest.raises(RuntimeError, match="禁止自动回退"):
        data_mod.load_contract_full_panel(["SR"], "2014-01-01", "2014-12-31")
    with pytest.raises(RuntimeError, match="禁止自动回退"):
        data_mod.load_vendor_main_map("SR", "2014-01-01")

    partial = pd.DataFrame([{"contract_object": "SR", "date": "2014-01-02", "date_stamp": 1}])
    partial.to_parquet(tmp_path / "contract_day_full.parquet", index=False)
    with pytest.raises(ValueError, match="缺少字段"):
        data_mod.load_contract_full_panel(["SR"], "2014-01-01", "2014-12-31")
