from __future__ import annotations

import hashlib
import json
import platform
from pathlib import Path
from typing import Any


def sha256_file(path: Path) -> str | None:
    if not path.exists():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_manifest(files: list[Path], query: dict[str, Any] | None = None) -> dict[str, Any]:
    return {
        "files": {
            str(path): {
                "sha256": sha256_file(path),
                "bytes": path.stat().st_size if path.exists() else None,
            }
            for path in sorted(set(files))
        },
        "query": query or {},
        "python": platform.python_version(),
        "platform": platform.platform(),
    }


def write_manifest(path: Path, manifest: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )


RUN_CONTRACT_FIELDS = (
    "database", "collection", "universe_pool", "symbols", "main_rule",
    "signal_price", "execution_price", "mark_price", "signal_lag",
    "cost_model", "annualization", "sample_split", "quality",
)


def validate_run_contract(metadata: dict[str, Any]) -> None:
    missing = [
        field for field in RUN_CONTRACT_FIELDS
        if field not in metadata or metadata[field] in (None, "")
    ]
    if missing:
        raise ValueError(f"run.json 缺少数据契约字段: {missing}")


def write_run_metadata(path: Path, metadata: dict[str, Any]) -> None:
    validate_run_contract(metadata)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )


def research_run_contract(
    *,
    database: str,
    collection: str | list[str],
    universe_pool: str,
    symbols: list[str],
    main_rule: str,
    quality: str,
    cost_model: str,
    sample_split: dict[str, Any],
    **metadata: Any,
) -> dict[str, Any]:
    """构造并校验研究脚本统一的运行元数据。样本外起点以协议为准。"""
    from research.protocol import canonical_sample_split

    result = {
        "database": database,
        "collection": collection,
        "universe_pool": universe_pool,
        "symbols": symbols,
        "main_rule": main_rule,
        "quality": quality,
        "cost_model": cost_model,
        "sample_split": sample_split,
        **metadata,
    }
    result["sample_split"] = canonical_sample_split(result.get("sample_split"))
    validate_run_contract(result)
    return result
