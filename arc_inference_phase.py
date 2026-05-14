"""
Inference phase — NO internet access.
Strategy:
  1. Load pre-computed answers from cache (set in prep phase by OpenRouter ensemble)
  2. Self-consistent program synthesis via QwQ-32B (K=3, time-budgeted)
  3. Final fallback — identity

Time budget: 3600s total. We dynamically allocate remaining time across
uncached tasks. Cached tasks take ~0s, so remaining time goes to vLLM.

BFS removed: benchmark showed 1.1% real accuracy (wrong architecture).

Telemetry: log_event() writes to /output/inference_telemetry.jsonl
(separate from prep buffer). No TG send — sandbox has no internet.
"""

import json
import os
import sys
import time
from pathlib import Path
from typing import List, Dict, Optional
from arc_utils import load_input_data, save_output_data
from arc_solver_llm import ARCSolver

# Use inference-specific event log path. Buffer-only (no TG: no internet).
os.environ.setdefault("EVENT_LOG_PATH", "/output/inference_telemetry.jsonl")

# Logger is optional — degrade gracefully if module unavailable
try:
    from tg_logger import log_event
except Exception:
    def log_event(event, level="INFO", data=None):  # type: ignore
        return {}

CACHE_FILE = Path("/app/cache.json")

# Hard limit: validator gives us 3600s for inference. Reserve 5% for startup/teardown.
INFERENCE_TIME_BUDGET = int(os.getenv("INFERENCE_TIME_BUDGET", "3420"))


def _is_identity(predicted: List[List[int]], test_input: List[List[int]]) -> bool:
    """Check if solver returned the input unchanged (identity fallback)."""
    if predicted is None or test_input is None:
        return False
    if len(predicted) != len(test_input):
        return False
    return all(predicted[i] == test_input[i] for i in range(len(predicted)))


def load_cache() -> Dict:
    if CACHE_FILE.exists():
        try:
            cache = json.loads(CACHE_FILE.read_text())
            print(f"✅ Cache loaded: {len(cache)} pre-solved tasks")
            return cache
        except Exception as e:
            print(f"⚠️ Cache load error: {e}")
    else:
        print("⚠️ No cache found — using vLLM only")
    return {}


def run_inference(input_dir: str, output_dir: str) -> None:
    print("=" * 60)
    print("INFERENCE PHASE: Cache-first + vLLM fallback")
    print("=" * 60)

    data = load_input_data(input_dir)
    tasks = data.get("tasks", [])
    print(f"Tasks to solve: {len(tasks)}")

    # Load pre-computed answers
    cache = load_cache()
    cache_size = len(cache)
    cache_non_null = sum(1 for v in cache.values() if v is not None)

    log_event("inference_start", "INFO", {
        "n_tasks": len(tasks),
        "cache_size": cache_size,
        "cache_non_null": cache_non_null,
        "time_budget_sec": INFERENCE_TIME_BUDGET,
    })

    # Init vLLM solver for uncached tasks
    solver = ARCSolver(use_vllm=True)
    log_event("vllm_init", "INFO" if solver.vllm_available else "WARN", {
        "available": solver.vllm_available,
        "model": solver.vllm_model,
    })

    predictions = []
    cache_hits = 0
    vllm_hits  = 0
    fallbacks  = 0
    inference_start = time.monotonic()

    for i, task in enumerate(tasks):
        task_hash   = task.get("task_hash", "")
        train       = task.get("train_examples", [])
        test_input  = task.get("test_input", [])

        # Calculate per-task budget: remaining_time / remaining_uncached_tasks
        elapsed = time.monotonic() - inference_start
        remaining_time = max(0, INFERENCE_TIME_BUDGET - elapsed)
        remaining_tasks = len(tasks) - i
        per_task_budget = remaining_time / max(1, remaining_tasks)

        print(f"\n[{i+1}/{len(tasks)}] {task_hash[:12]}... budget={per_task_budget:.0f}s elapsed={elapsed:.0f}s")

        predicted = None
        source    = "unknown"

        # 1. Cache (pre-computed by OpenRouter ensemble in prep phase) — takes ~0s
        if task_hash in cache and cache[task_hash] is not None:
            candidate = cache[task_hash]
            if solver._is_valid(candidate):
                predicted = candidate
                cache_hits += 1
                source = "cache"
                print(f"  ✅ Cache hit!")
            else:
                print(f"  ⚠️ Cache entry invalid, falling through to vLLM")

        # 2. vLLM program synthesis — only if enough budget (need >20s for one synthesis call)
        if predicted is None and per_task_budget > 20:
            print(f"  🤖 vLLM solving (K=3, budget={per_task_budget:.0f}s)...")
            predicted = solver.solve(train, test_input, time_budget_sec=per_task_budget)
            if predicted and not _is_identity(predicted, test_input):
                vllm_hits += 1
                source = "vllm"
            else:
                predicted = None  # treat identity from solver as fallback

        # 3. Identity fallback — last resort
        if predicted is None:
            predicted = [row[:] for row in test_input]
            fallbacks += 1
            source = "identity"
            print(f"  ⚠️ Identity fallback (budget exhausted or no solution)")

        predictions.append({
            "problem_index": i,
            "task_hash": task_hash,
            "predicted_output": predicted,
            "metadata": {"source": source},
        })

        log_event("task_done", "INFO", {
            "i": i + 1,
            "task_hash": task_hash,
            "source": source,
            "elapsed_sec": round(time.monotonic() - inference_start - elapsed, 1),
            "budget_left_sec": round(per_task_budget, 1),
        })

        # Incremental durability: save after every task so that a hard-kill
        # (validator timeout) doesn't lose all progress. Validator scoring
        # reads /output/ — we want SOMETHING there even if partial.
        try:
            partial = {
                "phase": "inference",
                "status": "in_progress",  # marker that we're not yet done
                "num_problems_solved": len(predictions),
                "vllm_available": solver.vllm_available,
                "cache_hits": cache_hits,
                "vllm_hits": vllm_hits,
                "fallbacks": fallbacks,
                "predictions": predictions,
            }
            save_output_data(partial, output_dir)
        except Exception as e:
            print(f"⚠️ Incremental output save failed: {e}")

    # Save final results
    results = {
        "phase": "inference",
        "status": "success",
        "num_problems_solved": len(predictions),
        "vllm_available": solver.vllm_available,
        "cache_hits": cache_hits,
        "vllm_hits": vllm_hits,
        "fallbacks": fallbacks,
        "predictions": predictions,
    }
    save_output_data(results, output_dir)

    total_time = time.monotonic() - inference_start
    print(f"\n{'='*60}")
    print(f"DONE: {len(predictions)} predictions  ({total_time:.0f}s of {INFERENCE_TIME_BUDGET}s)")
    print(f"  Cache hits:    {cache_hits}/{len(tasks)}")
    print(f"  vLLM (K=3):    {vllm_hits}/{len(tasks)}")
    print(f"  Fallbacks:     {fallbacks}/{len(tasks)}")
    print(f"{'='*60}")

    log_event("inference_done", "OK", {
        "n_tasks": len(tasks),
        "cache_hits": cache_hits,
        "vllm_hits": vllm_hits,
        "fallbacks": fallbacks,
        "total_time_sec": round(total_time, 1),
        "avg_time_per_task": round(total_time / max(1, len(tasks)), 1),
    })


def run_inference_phase(input_dir, output_dir):
    run_inference(str(input_dir), str(output_dir))
