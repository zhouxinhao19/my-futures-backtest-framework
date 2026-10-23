import json
from pathlib import Path

import pandas as pd

from backtest.reconcile import raw_change_without_roll_gap, replay_adj_pnl
from research.grid import run_grid
from research.metrics import ic_summary, information_coefficient, performance_metrics, quantile_mean_return
from validation.reports import write_validation


def test_raw_roll_replay_drops_the_gap_from_pnl():
    dates = pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05"])
    settle = pd.DataFrame({"RB": [100.0, 100.0, 200.0, 210.0]}, index=dates)
    adj = pd.DataFrame({"RB": [100.0, 100.0, 110.0, 115.5]}, index=dates)
    code = pd.DataFrame({"RB": ["a", "a", "b", "b"]}, index=dates)
    change = raw_change_without_roll_gap(settle, adj, code)
    lots = pd.DataFrame({"RB": [1.0, 1.0, 1.0, 1.0]}, index=dates)
    specs = {"RB": {"multiplier": 1.0}}
    raw = replay_adj_pnl(lots, settle, specs, holding=2, lag=1, cost_bps=0.0, price_change=change)
    naive = replay_adj_pnl(lots, settle, specs, holding=2, lag=1, cost_bps=0.0)
    assert change.loc[dates[2], "RB"] == 10.0
    assert raw["pnl"].abs().sum() < naive["pnl"].abs().sum()


def test_ic_and_quantiles_use_the_same_panel():
    frame = pd.DataFrame({
        "date": list(pd.bdate_range("2024-01-02", periods=4)) * 3,
        "factor": [1, 2, 3, 4, 0, 0, 0, 0, -1, -2, -3, -4],
        "forward": [0.03, 0.02, 0.01, 0.0, 0.0, 0.0, 0.0, 0.0, -0.03, -0.02, -0.01, 0.0],
    })
    summary = ic_summary(information_coefficient(frame))
    assert summary["ic"] > 0
    buckets = quantile_mean_return(frame, quantiles=3)
    assert buckets.iloc[-1] > buckets.iloc[0]
    metrics = performance_metrics(pd.Series([1.0, 1.1]), pd.Series([0.1, -0.05]))
    assert "calmar" in metrics and "win_rate" in metrics


def test_grid_checkpoint_skips_finished_configs(tmp_path: Path):
    path = tmp_path / "grid.jsonl"
    calls = []

    def worker(config):
        calls.append(config["name"])
        return config["name"] + "-ok"

    first = run_grid([{"name": "a"}, {"name": "b"}], worker, max_workers=1, checkpoint=path)
    assert first == ["a-ok", "b-ok"]
    second = run_grid([{"name": "a"}, {"name": "b"}], worker, max_workers=1, checkpoint=path)
    assert second == ["a-ok", "b-ok"]
    assert calls == ["a", "b"]
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 2


def test_validation_writes_the_three_tables(tmp_path: Path):
    directory = write_validation(
        tmp_path / "run",
        robustness=pd.DataFrame([{"name": "cost", "value": 1}]),
        diagnostics=pd.DataFrame([{"name": "overlap", "value": 0.2}]),
        trial={"factor": "M1"},
        registry=tmp_path / "index.jsonl",
    )
    assert (directory / "robustness.csv").is_file()
    assert (directory / "diagnostics.csv").is_file()
    assert (directory / "trials.csv").is_file()
    assert (tmp_path / "index.jsonl").is_file()
