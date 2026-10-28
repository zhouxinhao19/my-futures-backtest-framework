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
        assert spec["data_mode"] == "contract"


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
    result = data.load_mongo_commodity_panel()
    assert pd.isna(
        result.loc[result["symbol"] == "B", "available_at"].iloc[0]
    )
