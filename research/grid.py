from __future__ import annotations

import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any, Callable, Iterable


def _key(config: dict[str, Any]) -> str:
    return json.dumps(config, ensure_ascii=False, sort_keys=True, default=str)


def run_grid(
    configs: Iterable[dict[str, Any]],
    worker: Callable[[dict[str, Any]], Any],
    *,
    max_workers: int | None = None,
    checkpoint: str | Path | None = None,
) -> list[Any]:
    items = list(configs)
    done: dict[str, Any] = {}
    path = Path(checkpoint) if checkpoint else None
    if path and path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            done[row["key"]] = row["result"]
    pending = [item for item in items if _key(item) not in done]
    fresh: list[Any] = []
    if pending:
        if max_workers == 1 or max_workers is None and len(pending) == 1:
            fresh = [worker(item) for item in pending]
        else:
            with ProcessPoolExecutor(max_workers=max_workers) as pool:
                fresh = list(pool.map(worker, pending))
        if path:
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a", encoding="utf-8") as handle:
                for item, result in zip(pending, fresh):
                    handle.write(json.dumps({"key": _key(item), "result": result}, ensure_ascii=False, default=str) + "\n")
    return [done[_key(item)] if _key(item) in done else fresh.pop(0) for item in items]
