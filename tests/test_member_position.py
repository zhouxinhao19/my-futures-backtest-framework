from datetime import datetime

import numpy as np
import pandas as pd
import pytest

from data import attach_member_position
from data.member_position import (
    build_member_frame,
    mask_leading_zeros,
    to_documents,
)

CODES = ["RB.SHF", "BZ.DCE"]
NAMES = ["SHFE螺纹钢", "DCE 纯苯"]


def _write(path, rows):
    table = [[""] + CODES, [""] + NAMES]
    for date, values in rows.items():
        table.append([pd.Timestamp(date), *values])
    pd.DataFrame(table).to_excel(path, header=False, index=False)


@pytest.fixture
def paths(tmp_path):
    files = {
        "net_position": ("净持仓.xlsx", {
            "2014-01-02": [0, 0], "2014-01-03": [0, 0],
            "2014-01-06": [-5, 0], "2014-01-07": [0, 0],
        }),
        "top20_long": ("多.xlsx", {
            "2014-01-02": [0, 0], "2014-01-03": [3, 0],
            "2014-01-06": [7, 0], "2014-01-07": [0, 0],
        }),
        # 2014-01-03 缺行，应保持空而不是填 0
        "top20_short": ("空.xlsx", {
            "2014-01-02": [0, 0], "2014-01-06": [4, 0], "2014-01-07": [9, 0],
        }),
        "registered_receipt": ("仓单.xlsx", {
            "2014-01-02": [None, None], "2014-01-03": [0, None],
            "2014-01-06": [4, None], "2014-01-07": [0, None],
        }),
    }
    out = {}
    for field, (name, rows) in files.items():
        path = tmp_path / name
        _write(path, rows)
        out[field] = path
    return out


def test_mask_leading_zeros_keeps_zeros_after_first_nonzero():
    idx = pd.date_range("2014-01-01", periods=5)
    out = mask_leading_zeros(pd.Series([0.0, 0.0, 5.0, 0.0, 3.0], index=idx))
    assert out.iloc[:2].isna().all()
    assert out.iloc[2:].tolist() == [5.0, 0.0, 3.0]


def test_mask_leading_zeros_blanks_all_zero_series():
    idx = pd.date_range("2014-01-01", periods=2)
    assert mask_leading_zeros(pd.Series([0.0, 0.0], index=idx)).isna().all()


def test_build_keeps_source_values_and_masks_leading_zeros(paths):
    frame, dropped = build_member_frame(paths)
    assert dropped == 5  # RB 01-02 全空 + BZ 四天全空
    assert set(frame["symbol"]) == {"RB"}

    rb = frame.set_index("date")
    row = lambda d: rb.loc[pd.Timestamp(d)]  # noqa: E731
    assert list(rb.index.strftime("%Y-%m-%d")) == ["2014-01-03", "2014-01-06", "2014-01-07"]

    # 01-03：上市后的首个非零是多头；净持仓和仓单的前导 0 置空
    assert row("2014-01-03")["top20_long"] == 3
    assert pd.isna(row("2014-01-03")["top20_short"])
    assert pd.isna(row("2014-01-03")["net_position"])
    assert pd.isna(row("2014-01-03")["registered_receipt"])

    # 01-06：净持仓直接取源数据 -5，不由多空相减推算
    assert row("2014-01-06")["net_position"] == -5
    assert row("2014-01-06")["top20_long"] == 7
    assert row("2014-01-06")["top20_short"] == 4
    assert row("2014-01-06")["registered_receipt"] == 4

    # 01-07：首次非零之后的 0 保留为 0
    assert row("2014-01-07")["net_position"] == 0
    assert row("2014-01-07")["registered_receipt"] == 0


def test_codes_map_to_bare_symbol_and_exchange(paths):
    frame, _ = build_member_frame(paths)
    assert frame["wind_code"].unique().tolist() == ["RB.SHF"]
    assert frame["exchange"].unique().tolist() == ["SHFE"]
    assert frame["name"].unique().tolist() == ["螺纹钢"]


def test_documents_use_research_date_stamp_and_int_values(paths):
    frame, _ = build_member_frame(paths)
    docs = to_documents(frame, "test", datetime(2026, 10, 16))
    first = docs[0]
    assert first["contract_object"] == "RB"
    assert first["date"] == "2014-01-03"
    assert first["date_stamp"] == int(pd.Timestamp("2014-01-03").timestamp())
    assert first["receipt_unit"] == "exchange_published"
    assert first["net_position"] is None
    assert first["top20_long"] == 3 and isinstance(first["top20_long"], int)


def test_attach_is_left_join_without_coverage_gate():
    panel = pd.DataFrame({
        "date": pd.to_datetime(["2014-01-03", "2014-01-06"]),
        "symbol": ["RB", "RB"],
        "adj_close": [1.0, 2.0],
    })
    member = pd.DataFrame({
        "date": pd.to_datetime(["2014-01-06"]),
        "symbol": ["RB"],
        "top20_long": [7],
        "top20_short": [4],
        "net_position": [-5],
        "registered_receipt": [4],
        "receipt_unit": ["exchange_published"],
    })
    out = attach_member_position(panel, member)
    assert len(out) == 2
    assert np.isnan(out.loc[0, "net_position"])
    assert out.loc[1, "net_position"] == -5
