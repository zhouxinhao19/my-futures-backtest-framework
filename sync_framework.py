"""把源仓库已提交的框架文件同步到本仓库。

用法（在本仓库根目录）::

    python sync_framework.py
    python sync_framework.py --source D:\\回测框架测试

只读取源仓库的提交，不复制工作区里尚未提交的修改。
因子、策略、研究信号、回测入口和结果目录不在同步范围内。
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

DEST = Path(__file__).resolve().parent
STATE = DEST / "source-sync.json"
DEFAULT_SOURCE = Path(r"D:\回测框架测试")

BACKTEST_FILES = {"__init__.py", "ledger.py", "records.py", "reconcile.py"}
RESEARCH_EXCLUDE = {"__init__.py", "experiments.py"}
CONFIGS = {
    "configs/data.yaml",
    "configs/universe.yaml",
    "configs/backtest.yaml",
    "configs/costs.yaml",
    "configs/universe/commodity_xs_v1.json",
}
RESEARCH_TOKENS = ("factors", "strategies", "research.signals", "alpha_library")
HELD_TOKENS = ("backtest.run", "backtest.research_cases", "profile_backtest_speed")


def git(repo: Path, *args: str) -> str:
    completed = subprocess.run(
        ["git", "-C", str(repo), *args],
        check=True,
        capture_output=True,
    )
    return completed.stdout.decode("utf-8")


def allowed(path: str) -> bool:
    if path.startswith(("research/signals/", "factors/", "strategies/", "scripts/", "runs/", "reports/")):
        return False
    if path.startswith("data/") or path.startswith("validation/"):
        return True
    if path.startswith("research/") and path.count("/") == 1 and path.endswith(".py"):
        return Path(path).name not in RESEARCH_EXCLUDE
    if path.startswith("backtest/") and path.count("/") == 1:
        return Path(path).name in BACKTEST_FILES
    if path in CONFIGS or path == "pytest.ini":
        return True
    return path.startswith("tests/") and path.endswith(".py")


def parse_diff(text: str) -> list[tuple[str, str]]:
    changes: list[tuple[str, str]] = []
    for line in text.splitlines():
        if not line.strip():
            continue
        parts = line.split("\t")
        code = parts[0]
        if code.startswith(("R", "C")):
            changes.append(("D", parts[1]))
            changes.append(("A", parts[2]))
        else:
            changes.append((code[:1], parts[-1]))
    return changes


def blocked_reason(repo: Path, rev: str, path: str) -> str | None:
    """返回跳过原因。研究内容返回 ``ignore``，不写入同步状态。"""
    if not path.startswith("tests/"):
        return None
    try:
        text = git(repo, "show", f"{rev}:{path}")
    except subprocess.CalledProcessError:
        return "源提交中没有这个测试"
    for token in RESEARCH_TOKENS:
        if token in text:
            return "ignore"
    for token in HELD_TOKENS:
        if token in text:
            return f"测试引用了 {token}"
    return None


def clean(repo: Path, rev: str, path: str) -> None:
    status = git(repo, "status", "--porcelain", "--", path).strip()
    if status:
        raise SystemExit(f"源仓库有未提交修改，已停止: {path}")
    head = git(repo, "rev-parse", "HEAD").strip()
    if head != rev:
        raise SystemExit(f"源仓库 HEAD 在读取期间变化: {head} != {rev}")


def apply_file(source: Path, rev: str, path: str) -> None:
    clean(source, rev, path)
    target = DEST / path
    source_file = source / path
    if not source_file.is_file():
        raise SystemExit(f"源提交包含 {path}，但工作区没有对应文件")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(source_file.read_bytes())


def remove_file(path: str) -> None:
    target = DEST / path
    if target.is_file():
        target.unlink()


def load_state() -> dict:
    if not STATE.exists():
        raise SystemExit("缺少 source-sync.json。首次同步请传入 --from <源提交>")
    return json.loads(STATE.read_text(encoding="utf-8"))


def save_state(rev: str, skipped: list[dict[str, str]]) -> None:
    payload = {"source": rev, "skipped": skipped}
    STATE.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="同步源仓库中的框架文件")
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--from", dest="start", help="首次同步时的源提交，该提交本身不重复套用")
    args = parser.parse_args()
    source = args.source.resolve()
    rev = git(source, "rev-parse", "HEAD").strip()
    state = {"source": args.start, "skipped": []} if args.start else load_state()
    previous = str(state.get("source") or "").strip()
    if not previous:
        raise SystemExit("source-sync.json 没有 source")
    if git(source, "cat-file", "-t", previous).strip() != "commit":
        raise SystemExit(f"源仓库找不到提交 {previous}")

    seen: dict[str, str] = {}
    if previous != rev:
        diff = git(source, "diff", "--name-status", "--find-renames", f"{previous}..{rev}")
        for code, path in parse_diff(diff):
            if allowed(path):
                seen[path] = code
    for item in state.get("skipped") or []:
        seen.setdefault(item["path"], "M")

    applied: list[str] = []
    deleted: list[str] = []
    skipped: list[dict[str, str]] = []
    for path in sorted(seen):
        code = seen[path]
        if code == "D":
            remove_file(path)
            deleted.append(path)
            continue
        reason = blocked_reason(source, rev, path)
        if reason == "ignore":
            continue
        if reason:
            skipped.append({"path": path, "reason": reason})
            continue
        apply_file(source, rev, path)
        applied.append(path)

    save_state(rev, skipped)
    print(f"源提交 {previous} -> {rev}")
    for path in applied:
        print(f"更新 {path}")
    for path in deleted:
        print(f"删除 {path}")
    for item in skipped:
        print(f"跳过 {item['path']}: {item['reason']}")
    if not applied and not deleted and not skipped and previous == rev:
        print("框架文件已与源提交一致")


if __name__ == "__main__":
    main()
