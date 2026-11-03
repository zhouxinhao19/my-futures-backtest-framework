import pandas as pd

from data import pool_symbols, symbol_spec


def test_canonical_database_and_pools():
    all_symbols = set(pool_symbols("all_symbols"))
    commodity_xs = set(pool_symbols("commodity_xs"))
    assert {"AU", "AG", "JD", "ZC", "EC"} <= all_symbols
    assert commodity_xs == all_symbols - {"AU", "AG", "JD", "ZC", "EC"}
    account_core = pool_symbols("account_core")
    assert account_core == ["AU", "SC", "TA", "CU", "RB"]
    for product in account_core:
        spec = symbol_spec(product)
        assert spec["variety"] == product
        assert spec["multiplier"] > 0


def test_research_panel_parquet_is_read_once_per_process(monkeypatch, tmp_path):
    import data as data

    frame = pd.DataFrame([
        {
            "contract_object": "RB",
            "code": "rb1",
            "date": "2024-01-02",
            "expiry": "2025-01-01",
            "open": 1,
            "high": 1,
            "low": 1,
            "close": 1,
            "settle": 1,
            "adj_close": 1,
            "adj_settle": 1,
            "trade_turnover_vol": 1.0,
            "trade_turnover_value": 2e9,
            "trade_open_int": 1.0,
        },
        {
            "contract_object": "CU",
            "code": "cu1",
            "date": "2024-01-02",
            "expiry": "2025-01-01",
            "open": 1,
            "high": 1,
            "low": 1,
            "close": 1,
            "settle": 1,
            "adj_close": 1,
            "adj_settle": 1,
            "trade_turnover_vol": 1.0,
            "trade_turnover_value": 2e9,
            "trade_open_int": 1.0,
        },
    ])
    frame.to_parquet(tmp_path / "research_panel.parquet", index=False)
    monkeypatch.setattr(data, "parquet_root", lambda: tmp_path)
    data.clear_research_panel_cache()
    reads = {"n": 0}
    real_read = data.read_market_parquet

    def counted(name: str):
        reads["n"] += 1
        return real_read(name)

    monkeypatch.setattr(data, "read_market_parquet", counted)
    data.load_contract_dataset("RB", "2024-01-01", "2024-12-31")
    data.load_contract_dataset("CU", "2024-01-01", "2024-12-31")
    assert reads["n"] == 1


def test_protocol_uses_account_settle_book():
    from research.protocol import (
        book_mode,
        execution_price_column,
        ledger_execution_column,
        protocol_metadata,
    )

    meta = protocol_metadata()
    assert book_mode() == "account"
    assert execution_price_column() == "adj_settle"
    assert ledger_execution_column() == "settle"
    assert meta["execution_price"] == "settle"
    assert meta["mark_price"] == "settle"
    assert meta["book"] == "account"


def test_available_at_does_not_cross_products(monkeypatch, tmp_path):
    import data as data

    frame = pd.DataFrame([
        {"contract_object": "A", "code": "a1", "date": "2024-01-01",
         "expiry": "2025-01-01", "open": 1, "high": 1, "low": 1,
         "close": 1, "settle": 1, "adj_close": 1, "adj_settle": 1,
         "basis_rate": 0.1},
        {"contract_object": "B", "code": "b1", "date": "2024-01-01",
         "expiry": "2025-01-01", "open": 1, "high": 1, "low": 1,
         "close": 1, "settle": 1, "adj_close": 1, "adj_settle": 1,
         "basis_rate": 0.2},
    ])
    frame.to_parquet(tmp_path / "research_panel.parquet", index=False)
    monkeypatch.setattr(data, "parquet_root", lambda: tmp_path)
    data.clear_research_panel_cache()
    result = data.load_mongo_commodity_panel()
    assert pd.isna(
        result.loc[result["symbol"] == "B", "available_at"].iloc[0]
    )
