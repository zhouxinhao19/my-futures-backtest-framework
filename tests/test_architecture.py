from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PACKAGES = ("backtest", "data", "research", "validation")
FORBIDDEN = (
    "from factors",
    "import factors",
    "from strategies",
    "import strategies",
    "research.signals",
    "alpha_library",
)


def test_research_content_is_not_in_the_tree():
    assert not (ROOT / "factors").exists()
    assert not (ROOT / "strategies").exists()
    assert not (ROOT / "research" / "signals").exists()
    assert not (ROOT / "alpha_library").exists()
    assert not (ROOT / "runs").exists()
    assert not (ROOT / "reports").exists()


def test_engine_does_not_import_factors_or_strategies():
    for name in PACKAGES:
        for path in (ROOT / name).rglob("*.py"):
            source = path.read_text(encoding="utf-8")
            for token in FORBIDDEN:
                assert token not in source, f"{path.name} 引用了 {token}"
