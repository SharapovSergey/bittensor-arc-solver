"""
Inference phase — NO internet access.
Strategy:
  1. Load pre-computed answers from cache (set in prep phase by OpenRouter ensemble)
  2. Self-consistent program synthesis via QwQ-32B:
     - Sample K=20 programs at 4 temperatures
     - Keep only programs that pass ALL training pairs
     - Majority-vote on test output
  3. Final fallback — identity

BFS removed: benchmark showed 1.1% real accuracy (wrong architecture —
chain applies to base_fn(input), not input directly).
"""

import json
import os
import sys
from pathlib import Path
from typing import List, Dict, Optional
from arc_utils import load_input_data, save_output_data
from arc_solver_llm import ARCSolver

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
    vllm_hits  = 0
    fallbacks  = 0

    for i, task in enumerate(tasks):
        task_hash   = task.get("task_hash", "")
        train       = task.get("train_examples", [])
        test_input  = task.get("test_input", [])
        print(f"\n[{i+1}/{len(tasks)}] {task_hash[:12]}...")

        predicted = None
        source    = "unknown"

        # 1. Cache (pre-computed by OpenRouter ensemble in prep phase)
        if task_hash in cache and cache[task_hash] is not None:
            candidate = cache[task_hash]
            if solver._is_valid(candidate):
                predicted = candidate
                cache_hits += 1
                source = "cache"
                print(f"  ✅ Cache hit!")
            else:
                print(f"  ⚠️ Cache entry invalid, falling through to vLLM")

        # 2. Self-consistent program synthesis via QwQ-32B (K=20, majority vote)
        if predicted is None:
            print(f"  🤖 vLLM solving (K=20 synthesis)...")
            predicted = solver.solve(train, test_input)
            if predicted:
                vllm_hits += 1
                source = "vllm"

        # 3. Identity fallback — last resort
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
    print(f"  Cache hits:    {cache_hits}/{len(tasks)}")
    print(f"  vLLM (K=20):   {vllm_hits}/{len(tasks)}")
    print(f"  Fallbacks:     {fallbacks}/{len(tasks)}")
    print(f"{'='*60}")


def run_inference_phase(input_dir, output_dir):
    run_inference(str(input_dir), str(output_dir))
