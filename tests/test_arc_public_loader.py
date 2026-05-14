"""Unit tests for arc_public_loader."""
import json
import tempfile
from pathlib import Path

import pytest

from arc_public_loader import (
    _heuristic_difficulty,
    load_public_arc,
    stats,
)


@pytest.fixture
def fake_dataset_dir(tmp_path: Path) -> Path:
    """Build a small fake dataset matching arcprize/ARC-AGI-2 layout."""
    train = tmp_path / "training"
    train.mkdir()
    eval_dir = tmp_path / "evaluation"
    eval_dir.mkdir()

    # Easy task: 2x2 single color
    (train / "easy01.json").write_text(json.dumps({
        "train": [
            {"input": [[1, 0], [0, 1]], "output": [[1, 0], [0, 1]]},
            {"input": [[2, 0], [0, 2]], "output": [[2, 0], [0, 2]]},
            {"input": [[3, 0], [0, 3]], "output": [[3, 0], [0, 3]]},
        ],
        "test": [{"input": [[5, 0], [0, 5]], "output": [[5, 0], [0, 5]]}],
    }))
    # Medium task: 5x5 multi-color
    (train / "med02.json").write_text(json.dumps({
        "train": [
            {"input": [[i for i in range(5)] for _ in range(5)],
             "output": [[i for i in range(5)] for _ in range(5)]},
            {"input": [[1,2,3,4,5]] * 5,
             "output": [[5,4,3,2,1]] * 5},
        ],
        "test": [{"input": [[1,2,3,4,5]], "output": [[5,4,3,2,1]]}],
    }))
    # Hard task: 15x15 many colors
    (train / "hard03.json").write_text(json.dumps({
        "train": [
            {"input": [[i % 9 + 1 for i in range(15)] for _ in range(15)],
             "output": [[i % 9 + 1 for i in range(15)] for _ in range(15)]},
        ],
        "test": [],
    }))
    return tmp_path


class TestLoadPublicArc:
    def test_loads_all(self, fake_dataset_dir):
        tasks = load_public_arc(n=0, data_dir=fake_dataset_dir)
        assert len(tasks) == 3

    def test_caps_at_n(self, fake_dataset_dir):
        tasks = load_public_arc(n=2, data_dir=fake_dataset_dir)
        assert len(tasks) == 2

    def test_format_matches_sn5(self, fake_dataset_dir):
        tasks = load_public_arc(n=1, data_dir=fake_dataset_dir)
        t = tasks[0]
        assert "task_id" in t
        assert "train_examples" in t
        assert all("input" in ex and "output" in ex for ex in t["train_examples"])

    def test_filter_hard_first(self, fake_dataset_dir):
        # With filter_by="hard", hardest task should be first
        tasks = load_public_arc(n=3, filter_by="hard", data_dir=fake_dataset_dir)
        difficulties = [t["_difficulty"] for t in tasks]
        assert difficulties == sorted(difficulties, reverse=True)

    def test_filter_easy_first(self, fake_dataset_dir):
        tasks = load_public_arc(n=3, filter_by="easy", data_dir=fake_dataset_dir)
        difficulties = [t["_difficulty"] for t in tasks]
        assert difficulties == sorted(difficulties)

    def test_seed_stability(self, fake_dataset_dir):
        """Same seed → same task order."""
        a = load_public_arc(n=3, seed=42, data_dir=fake_dataset_dir)
        b = load_public_arc(n=3, seed=42, data_dir=fake_dataset_dir)
        assert [t["task_id"] for t in a] == [t["task_id"] for t in b]

    def test_skips_tasks_without_train(self, tmp_path: Path):
        d = tmp_path / "training"
        d.mkdir()
        # task with empty train field
        (d / "empty.json").write_text(json.dumps({"train": [], "test": []}))
        (d / "good.json").write_text(json.dumps({
            "train": [{"input": [[1]], "output": [[1]]}],
            "test": []
        }))
        tasks = load_public_arc(n=0, data_dir=tmp_path)
        assert len(tasks) == 1
        assert tasks[0]["task_id"] == "good"

    def test_missing_dir_returns_empty(self, tmp_path: Path):
        tasks = load_public_arc(n=10, data_dir=tmp_path / "nonexistent")
        assert tasks == []


class TestHeuristicDifficulty:
    def test_small_simple_grid_is_easy(self):
        task = {"train": [{"input": [[1]], "output": [[1]]}] * 3}
        assert _heuristic_difficulty(task) <= 3

    def test_large_complex_grid_is_hard(self):
        big = [[i % 9 + 1 for i in range(20)] for _ in range(20)]
        task = {"train": [{"input": big, "output": big}]}
        assert _heuristic_difficulty(task) >= 6

    def test_no_train_returns_default(self):
        assert _heuristic_difficulty({"train": []}) == 5


class TestStats:
    def test_empty(self):
        assert stats([]) == {"n": 0}

    def test_aggregates(self, fake_dataset_dir):
        tasks = load_public_arc(n=0, data_dir=fake_dataset_dir)
        s = stats(tasks)
        assert s["n"] == 3
        assert s["train_pairs_per_task"]["min"] >= 1
