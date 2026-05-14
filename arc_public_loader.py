"""
Loader for public ARC-AGI-2 dataset (arcprize/ARC-AGI-2).

The dataset is baked into the Docker image at /app/data/arc_agi2_public/
during build (via Dockerfile COPY). 1000 training + 120 evaluation tasks.

Public dataset format (per task JSON):
{
  "train": [{"input": [[...]], "output": [[...]]}, ...],   # variable count, usually 3
  "test":  [{"input": [[...]], "output": [[...]]}, ...]    # usually 1 pair
}

Mapping to SN5 training format:
- Use `train` field as-is for TTT (LoRA learns from input→output pairs)
- We don't need `test` outputs for training (TTT is unsupervised on train pairs)

Env vars:
  PUBLIC_ARC_DIR   (default /app/data/arc_agi2_public) — where dataset lives
  PUBLIC_ARC_N     (default 0)   — how many tasks to load. 0 = all training.
  PUBLIC_ARC_FILTER (default "") — "hard" / "easy" filter (heuristic)
"""
from __future__ import annotations

import json
import os
import random
from pathlib import Path
from typing import Iterable, List, Dict, Optional


DEFAULT_DATA_DIR = Path(__file__).parent / "data" / "arc_agi2_public"


def _heuristic_difficulty(task: Dict) -> int:
    """
    Approximate difficulty by grid sizes + color count + train-pair count.
    Larger / more colorful / fewer train examples → harder. Returns 0-10.
    """
    train = task.get("train", [])
    if not train:
        return 5
    max_h = max((len(ex["input"]) for ex in train), default=0)
    max_w = max((len(ex["input"][0]) for ex in train if ex["input"]), default=0)
    colors = set()
    for ex in train:
        for row in ex["input"]:
            colors.update(row)
        for row in ex["output"]:
            colors.update(row)
    size_score = min(5, (max_h * max_w) // 50)  # 0-5
    color_score = min(3, len(colors) // 3)       # 0-3
    pairs_score = max(0, 3 - len(train))         # 0-3
    return size_score + color_score + pairs_score  # 0-11


def _iter_task_files(data_dir: Path, split: str = "training") -> Iterable[Path]:
    d = data_dir / split
    if not d.exists():
        return
    yield from sorted(d.glob("*.json"))


def load_public_arc(
    n: int = 0,
    filter_by: str = "",
    split: str = "training",
    seed: int = 42,
    data_dir: Optional[Path] = None,
) -> List[Dict]:
    """
    Load up to `n` public ARC-AGI-2 tasks.

    Returns: list of dicts {"task_id", "train_examples", "test_input"?}
    Format compatible with SN5 task structure consumed by arc_ttt.py.

    n=0       → load all
    filter_by → "hard" / "easy" / "" (no filter)
    seed      → shuffle seed for stable selection across runs
    """
    data_dir = data_dir or Path(os.getenv("PUBLIC_ARC_DIR", str(DEFAULT_DATA_DIR)))

    files = list(_iter_task_files(data_dir, split))
    if not files:
        print(f"[arc_public_loader] WARN: no tasks at {data_dir}/{split}")
        return []

    tasks: List[Dict] = []
    for f in files:
        try:
            raw = json.loads(f.read_text())
        except Exception:
            continue
        train = raw.get("train") or []
        test = raw.get("test") or []
        if not train:
            continue
        # SN5 convention: train_examples + test_input (single test pair canonical)
        first_test = test[0] if test else {}
        tasks.append({
            "task_id": f.stem,
            "train_examples": train,
            "test_input": first_test.get("input"),
            "test_output": first_test.get("output"),
            "_difficulty": _heuristic_difficulty(raw),
        })

    if filter_by == "hard":
        tasks.sort(key=lambda t: -t["_difficulty"])
    elif filter_by == "easy":
        tasks.sort(key=lambda t: t["_difficulty"])
    else:
        rng = random.Random(seed)
        rng.shuffle(tasks)

    if n > 0:
        tasks = tasks[:n]

    return tasks


def stats(tasks: List[Dict]) -> Dict:
    """Quick stats about a loaded subset (sanity check before TTT)."""
    if not tasks:
        return {"n": 0}
    diffs = [t["_difficulty"] for t in tasks]
    pair_counts = [len(t["train_examples"]) for t in tasks]
    return {
        "n": len(tasks),
        "difficulty_range": (min(diffs), max(diffs)),
        "difficulty_avg": sum(diffs) / len(diffs),
        "train_pairs_per_task": {
            "min": min(pair_counts),
            "max": max(pair_counts),
            "avg": sum(pair_counts) / len(pair_counts),
        },
    }


if __name__ == "__main__":
    n = int(os.getenv("PUBLIC_ARC_N", "10"))
    tasks = load_public_arc(n=n, filter_by=os.getenv("PUBLIC_ARC_FILTER", ""))
    print(f"Loaded {len(tasks)} tasks")
    print(json.dumps(stats(tasks), indent=2))
    if tasks:
        t = tasks[0]
        print(f"\nFirst task: {t['task_id']} (diff={t['_difficulty']})")
        print(f"  train pairs: {len(t['train_examples'])}")
        print(f"  test input shape: {len(t['test_input'])}x{len(t['test_input'][0])}")
