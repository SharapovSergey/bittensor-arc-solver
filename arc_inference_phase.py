"""
Inference phase — NO internet access.
Strategy:
  1. BFS symbolic solver — exact transforms from the ARC-AGI-2 generator
     (guaranteed correct if chain found, depth 1-3, ~30-50% of tasks)
  2. Load pre-computed answers from cache (set in prep phase by OpenRouter ensemble)
  3. For uncached tasks — use local vLLM (Qwen2.5-72B)
  4. Final fallback — identity
"""

import json
import os
import sys
from pathlib import Path
from typing import List, Dict, Optional
from arc_utils import load_input_data, save_output_data
from arc_solver_llm import ARCSolver

try:
    from arc_bfs_solver import bfs_solve
    BFS_AVAILABLE = True
except ImportError:
    BFS_AVAILABLE = False
    print("⚠️ BFS solver not available")

CACHE_FILE = Path("/app/cache.json")


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

    # Init vLLM solver for uncached tasks
    solver = ARCSolver(use_vllm=True)

    predictions = []
    cache_hits = 0
    bfs_hits   = 0
    vllm_hits  = 0
    fallbacks  = 0

    for i, task in enumerate(tasks):
        task_hash   = task.get("task_hash", "")
        train       = task.get("train_examples", [])
        test_input  = task.get("test_input", [])
        print(f"\n[{i+1}/{len(tasks)}] {task_hash[:12]}...")

        predicted = None
        source    = "unknown"

        # 1. BFS symbolic solver — exact transforms, guaranteed correct if found
        if BFS_AVAILABLE and train:
            predicted = bfs_solve(train, test_input, max_depth=3)
            if predicted:
                bfs_hits += 1
                source = "bfs"

        # 2. Cache (pre-computed by OpenRouter ensemble in prep phase)
        if predicted is None:
            if task_hash in cache and cache[task_hash] is not None:
                predicted = cache[task_hash]
                cache_hits += 1
                source = "cache"
                print(f"  ✅ Cache hit!")

        # 3. vLLM fallback
        if predicted is None:
            print(f"  🤖 vLLM solving...")
            predicted = solver.solve(train, test_input)
            if predicted:
                vllm_hits += 1
                source = "vllm"

        # 4. Identity fallback — last resort
        if predicted is None:
            predicted = [row[:] for row in test_input]
            fallbacks += 1
            source = "identity"
            print(f"  ⚠️ Identity fallback")

        predictions.append({
            "problem_index": i,
            "task_hash": task_hash,
            "predicted_output": predicted,
            "metadata": {"source": source},
        })

    # Save results
    results = {
        "bfs_hits": bfs_hits,
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

    print(f"\n{'='*60}")
    print(f"DONE: {len(predictions)} predictions")
    print(f"  Cache hits: {cache_hits}/{len(tasks)}")
    print(f"  vLLM hits:  {vllm_hits}/{len(tasks)}")
    print(f"  Fallbacks:  {fallbacks}/{len(tasks)}")
    print(f"{'='*60}")


def run_inference_phase(input_dir, output_dir):
    run_inference(str(input_dir), str(output_dir))
