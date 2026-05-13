"""
Unit test for historical cache (Phase 1 / #3).

Tests:
  1. Load — empty file → empty dict
  2. Load — valid file → dict
  3. Load — corrupted file → empty dict (graceful)
  4. Update — merges new without overwriting old
  5. Validation — filters invalid grids (wrong type, too big, wrong values)
"""
import json
import shutil
from pathlib import Path

import arc_prep_phase as prep


def test_load_empty_file():
    """Empty file or missing file → empty dict."""
    prep.HISTORICAL_CACHE_FILE = Path("/tmp/test_cache_empty.json")
    if prep.HISTORICAL_CACHE_FILE.exists():
        prep.HISTORICAL_CACHE_FILE.unlink()
    # Missing file
    result = prep._load_historical_cache()
    assert result == {}, f"Missing file should return {{}}, got {result}"
    # Empty content
    prep.HISTORICAL_CACHE_FILE.write_text("{}")
    result = prep._load_historical_cache()
    assert result == {}, f"Empty {{}} should return {{}}, got {result}"
    print("  ✅ test_load_empty_file")


def test_load_valid():
    """Valid cache loads correctly."""
    prep.HISTORICAL_CACHE_FILE = Path("/tmp/test_cache_valid.json")
    sample = {
        "task_hash_1": [[0, 1, 2], [3, 4, 5]],
        "task_hash_2": [[9, 9], [0, 0]],
    }
    prep.HISTORICAL_CACHE_FILE.write_text(json.dumps(sample))
    result = prep._load_historical_cache()
    assert result == sample, f"Mismatch: {result} != {sample}"
    print("  ✅ test_load_valid")


def test_load_filters_invalid():
    """Invalid entries are filtered out."""
    prep.HISTORICAL_CACHE_FILE = Path("/tmp/test_cache_invalid.json")
    sample = {
        "valid": [[0, 1], [2, 3]],
        "not_grid": "string instead",
        "wrong_values": [[10, 11], [0, 0]],     # 10/11 out of 0-9
        "too_big": [[0] * 50 for _ in range(50)],  # 50×50 > 30
        "empty": [],
        "ragged": [[0, 1], [2]],                 # different row lengths — should still pass our basic check
    }
    prep.HISTORICAL_CACHE_FILE.write_text(json.dumps(sample))
    result = prep._load_historical_cache()
    assert "valid" in result, "Valid entry must survive"
    assert "not_grid" not in result, "String value must be filtered"
    assert "wrong_values" not in result, "Values >9 must be filtered"
    assert "too_big" not in result, "Grid >30×30 must be filtered"
    assert "empty" not in result, "Empty list must be filtered"
    print(f"  ✅ test_load_filters_invalid (kept: {list(result.keys())})")


def test_load_corrupted():
    """Corrupted JSON → empty dict (graceful)."""
    prep.HISTORICAL_CACHE_FILE = Path("/tmp/test_cache_corrupt.json")
    prep.HISTORICAL_CACHE_FILE.write_text("{not valid json")
    result = prep._load_historical_cache()
    assert result == {}, f"Corrupted should return {{}}, got {result}"
    print("  ✅ test_load_corrupted")


def test_update_merges_without_overwrite():
    """Update adds new entries, preserves old."""
    prep.HISTORICAL_CACHE_FILE = Path("/tmp/test_cache_update.json")
    if prep.HISTORICAL_CACHE_FILE.exists():
        prep.HISTORICAL_CACHE_FILE.unlink()

    old = {"hash_a": [[0]], "hash_b": [[1]]}
    fresh = {
        "hash_a": [[99]],    # would overwrite — must NOT
        "hash_c": [[2]],     # new — should add
        "hash_d": None,      # None — must skip
    }
    prep._update_historical_cache(old, fresh)
    saved = json.loads(prep.HISTORICAL_CACHE_FILE.read_text())

    assert saved["hash_a"] == [[0]], "Old must not be overwritten"
    assert saved["hash_b"] == [[1]], "Old must remain"
    assert saved["hash_c"] == [[2]], "Fresh must be added"
    assert "hash_d" not in saved, "None must be skipped"
    print(f"  ✅ test_update_merges_without_overwrite (saved: {list(saved.keys())})")


def test_pipeline_integration():
    """Simulate: prep finds 3 cache hits + 7 fresh, merged cache contains all 10."""
    prep.HISTORICAL_CACHE_FILE = Path("/tmp/test_cache_pipe.json")
    historical = {f"t{i}": [[i]] for i in range(3)}  # historical: t0,t1,t2
    prep.HISTORICAL_CACHE_FILE.write_text(json.dumps(historical))

    loaded = prep._load_historical_cache()
    assert len(loaded) == 3

    # Simulate today's 10 tasks
    today_tasks = [{"task_hash": f"t{i}"} for i in range(10)]
    hist_hits = sum(1 for t in today_tasks if t["task_hash"] in loaded)
    assert hist_hits == 3, f"Should have 3 hits, got {hist_hits}"

    # Tasks to solve fresh
    to_solve = [t for t in today_tasks if t["task_hash"] not in loaded]
    assert len(to_solve) == 7

    # Simulate fresh solves succeed for 5/7
    fresh_cache = {f"t{i}": [[i + 100]] for i in range(3, 8)}  # t3-t7 solved
    fresh_cache.update({"t8": None, "t9": None})                # t8,t9 failed

    # Merged result
    merged: dict = {}
    for t in today_tasks:
        h = t["task_hash"]
        if h in loaded:
            merged[h] = loaded[h]
        elif h in fresh_cache:
            merged[h] = fresh_cache[h]

    solved = sum(1 for v in merged.values() if v is not None)
    assert solved == 8, f"Should solve 8/10 (3 hist + 5 fresh), got {solved}"

    # Update historical
    prep._update_historical_cache(loaded, fresh_cache)
    updated = json.loads(prep.HISTORICAL_CACHE_FILE.read_text())
    assert len(updated) == 3 + 5, f"Historical should grow to 8, got {len(updated)}"
    print(f"  ✅ test_pipeline_integration (cache grew 3 → {len(updated)})")


def main():
    print("Running Phase 1 unit tests (historical cache):")
    test_load_empty_file()
    test_load_valid()
    test_load_filters_invalid()
    test_load_corrupted()
    test_update_merges_without_overwrite()
    test_pipeline_integration()
    print("\n✅ All Phase 1 tests passed")
    # cleanup
    for f in Path("/tmp").glob("test_cache_*.json"):
        f.unlink()


if __name__ == "__main__":
    main()
