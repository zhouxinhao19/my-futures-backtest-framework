from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PACKAGES = ("backtest", "data", "research", "validation")
ALLOWED_FACTOR_FILES = {
    ROOT / "factors" / "__init__.py",
    ROOT / "factors" / "signals" / "__init__.py",
    ROOT / "factors" / "signals" / "external_factor.py",
    ROOT / "factors" / "signals" / "liquidity.py",
}
ALLOWED_STRATEGY_FILES = {
    ROOT / "strategies" / "__init__.py",
    ROOT / "strategies" / "factor_file.py",
    ROOT / "strategies" / "schedule.py",
}
FORBIDDEN = (
    "from factors",
    "import factors",
    "from strategies",
    "import strategies",
    "research.signals",
    "alpha_library",
)


def test_research_content_is_not_in_the_tree():
    factor_files = {
        path for path in (ROOT / "factors").rglob("*.py")
    } if (ROOT / "factors").exists() else set()
    strategy_files = {
        path for path in (ROOT / "strategies").rglob("*.py")
    } if (ROOT / "strategies").exists() else set()
    assert factor_files <= ALLOWED_FACTOR_FILES
    assert strategy_files <= ALLOWED_STRATEGY_FILES
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
