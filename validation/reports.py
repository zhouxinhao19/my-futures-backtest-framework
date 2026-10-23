from __future__ import annotations

from pathlib import Path

import pandas as pd

from .trials import TrialRegistry


def write_validation(
    directory: str | Path,
    *,
    robustness: pd.DataFrame,
    diagnostics: pd.DataFrame,
    trial: dict,
    registry: str | Path | None = None,
) -> Path:
    path = Path(directory)
    path.mkdir(parents=True, exist_ok=True)
    robustness.to_csv(path / "robustness.csv", index=False, encoding="utf-8-sig")
    diagnostics.to_csv(path / "diagnostics.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame([trial]).to_csv(path / "trials.csv", index=False, encoding="utf-8-sig")
    if registry is not None:
        TrialRegistry(registry).register(trial)
    return path
