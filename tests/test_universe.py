import hashlib
import json
from pathlib import Path

from data import pool_symbols, symbol_spec
from data.specs import multiplier_on

ROOT = Path(__file__).resolve().parents[1]


def test_versioned_pool_matches_definition_and_exclusions():
    import yaml
    payload = yaml.safe_load((ROOT / "configs" / "universe.yaml").read_text(encoding="utf-8"))
    pool = payload["pools"]["commodity_xs"]
    assert pool["version"] == 1
    assert pool["members"] == payload["pool_definitions"]["commodity_xs"]
    assert set(pool["exclusions"]) == {"AU", "AG", "JD", "ZC", "EC"}
    assert set(payload["pools"]["account_core"]["members"]) - set(pool["members"]) == {"AU"}


def test_commodity_xs_snapshot_and_multipliers():
    members = pool_symbols("commodity_xs")
    assert len(members) == 55
    assert len(set(members)) == 55
    assert set(members) == set(pool_symbols("all_symbols")) - {"AU", "AG", "JD", "ZC", "EC"}
    assert "AU" in pool_symbols("account_core")
    assert set(pool_symbols("account_core")) - set(members) == {"AU"}
    snapshot = json.loads((ROOT / "configs" / "universe" / "commodity_xs_v1.json").read_text(encoding="utf-8"))
    body = {key: snapshot[key] for key in ("pool", "version", "as_of", "members")}
    digest = hashlib.sha256(json.dumps(body, ensure_ascii=False, separators=(",", ":")).encode()).hexdigest()
    assert snapshot["sha256"] == digest
    assert snapshot["members"] == members
    for symbol in members:
        assert multiplier_on(symbol, "2024-06-03") > 0
        assert symbol_spec(symbol)["multiplier"] > 0


def test_multiplier_schedule_is_contiguous_for_changed_contracts():
    assert multiplier_on("FU", "2018-07-13") == 50
    assert multiplier_on("FU", "2018-07-16") == 10
    assert multiplier_on("MA", "2014-12-08") == 50
    assert multiplier_on("MA", "2014-12-09") == 10
