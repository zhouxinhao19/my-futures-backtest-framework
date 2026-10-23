import json

import pandas as pd
import pytest

from backtest.ledger import ResearchResult
from backtest.records import write_research_run
from research.repro import research_run_contract, validate_run_contract


def test_run_contract_rejects_missing_fields():
    with pytest.raises(ValueError, match="run.json"):
        validate_run_contract({"database": "quantaxis_price_data"})


def test_research_run_contract_contains_protocol_fields():
    metadata = research_run_contract(
        database="quantaxis_price_data",
        collection="price_data",
        universe_pool="commodity_xs",
        symbols=["A"],
        main_rule="nearest_unexpired",
        quality="research_panel",
        cost_model="bps_proxy",
        sample_split={"start": "2024-01-01"},
        signal_price="adj_close",
        execution_price="open",
        mark_price="settle",
        signal_lag=1,
        annualization=243,
    )
    assert metadata["universe_pool"] == "commodity_xs"
    assert metadata["sample_split"]["oos_start"] == "2025-01-01"
    with pytest.raises(ValueError, match="2025-01-01"):
        research_run_contract(
            database="quantaxis_price_data",
            collection="price_data",
            universe_pool="commodity_xs",
            symbols=["A"],
            main_rule="nearest_unexpired",
            quality="research_panel",
            cost_model="bps_proxy",
            sample_split={"start": "2024-01-01", "oos_start": "2023-01-01"},
            signal_price="adj_close",
            execution_price="open",
            mark_price="settle",
            signal_lag=1,
            annualization=243,
        )


def test_research_run_writes_fixed_sample_summary(tmp_path):
    dates = pd.to_datetime(["2024-12-31", "2025-01-02", "2025-01-03"])
    daily = pd.DataFrame(
        {
            "equity_after": [100.0, 110.0, 121.0],
            "ret": [0.0, 0.10, 0.10],
        },
        index=dates,
    )
    result = ResearchResult(
        daily=daily,
        trades=pd.DataFrame({"symbol": ["A"]}),
        positions=pd.DataFrame({"symbol": ["A"]}),
    )
    written = write_research_run(tmp_path / "run", {"strategy": "demo"}, result)
    files = {path.name for path in written.iterdir()}
    assert files == {"run.json", "daily.csv", "trades.csv", "positions.csv", "summary.csv"}
    metadata = json.loads((written / "run.json").read_text(encoding="utf-8"))
    assert metadata["sample_split"]["oos_start"] == "2025-01-01"
    summary = pd.read_csv(written / "summary.csv")
    assert summary["sample"].tolist() == ["全样本", "样本内", "样本外"]
    assert summary.loc[summary["sample"] == "样本外", "start"].iloc[0] == "2025-01-02"
