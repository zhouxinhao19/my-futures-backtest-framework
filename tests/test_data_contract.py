import pandas as pd

from data import (
    MONGO_DB, load_mongo_commodity_panel, pool_symbols, symbol_spec,
)


def test_canonical_database_and_pools():
    assert MONGO_DB == "quantaxis_price_data"
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


def test_available_at_does_not_cross_products(monkeypatch):
    import data as data

    rows = [
        {"contract_object": "A", "code": "a1", "date": "2024-01-01",
         "expiry": "2025-01-01", "open": 1, "high": 1, "low": 1,
         "close": 1, "settle": 1, "adj_close": 1, "adj_settle": 1,
         "basis_rate": 0.1},
        {"contract_object": "B", "code": "b1", "date": "2024-01-01",
         "expiry": "2025-01-01", "open": 1, "high": 1, "low": 1,
         "close": 1, "settle": 1, "adj_close": 1, "adj_settle": 1,
         "basis_rate": 0.2},
    ]

    class Collection:
        def find(self, *_args, **_kwargs):
            return self

        def sort(self, *_args, **_kwargs):
            return rows

        def __iter__(self):
            return iter(rows)

    class Db:
        price_data_research = Collection()

        def __getitem__(self, name):
            return getattr(self, name)

    class Client:
        def __getitem__(self, _name):
            return Db()

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

    monkeypatch.setattr(data, "MONGO_URI", "mongodb://test")
    monkeypatch.setattr(data, "MONGO_DB", "test")
    monkeypatch.setattr("pymongo.MongoClient", lambda *_a, **_k: Client())
    result = load_mongo_commodity_panel()
    assert pd.isna(
        result.loc[result["symbol"] == "B", "available_at"].iloc[0]
    )
