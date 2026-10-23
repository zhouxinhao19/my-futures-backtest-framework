from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any


class TrialRegistry:
    def __init__(self, path: str | Path = "runs/_trials/index.jsonl"):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def register(self, config: dict[str, Any], *, result: str = "started") -> dict[str, Any]:
        trial = {
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "result": result,
            "config": config,
        }
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(trial, ensure_ascii=False, default=str) + "\n")
        return trial
