from pathlib import Path

from research.repro import build_manifest
from research.runtime import ExecutionConfig, load_runtime_config


def test_runtime_config_loads_execution_controls():
    execution, universe = load_runtime_config()
    assert isinstance(execution, ExecutionConfig)
    assert execution.slippage_ticks >= 0
    assert 0 < execution.margin_cap <= 1
    assert "symbols" in universe


def test_manifest_hash_is_stable(tmp_path: Path):
    source = tmp_path / "data.csv"
    source.write_text("date,close\n2024-01-01,1\n", encoding="utf-8")
    first = build_manifest([source], {"start": "2024-01-01"})
    second = build_manifest([source], {"start": "2024-01-01"})
    assert first == second
    assert len(first["files"][str(source)]["sha256"]) == 64
