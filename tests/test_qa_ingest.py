import os

import pytest


@pytest.mark.mongo
def test_qa_future_day_has_settle():
    try:
        from pymongo import MongoClient
        db = MongoClient("mongodb://127.0.0.1:27017", serverSelectionTimeoutMS=1000).quantaxis
        db.command("ping")
        if "future_day" not in db.list_collection_names():
            pytest.skip("quantaxis.future_day 不存在")
    except Exception:
        pytest.skip("MongoDB 不可用")
    doc = db.future_day.find_one({"code": "au1406"})
    if doc is None:
        pytest.skip("quantaxis.future_day 没有 au1406")
    assert doc is not None
    assert "settle" in doc
    assert {"code", "date", "date_stamp", "open", "close", "position", "trade"} <= set(doc)
