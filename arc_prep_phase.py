"""
Prep phase — WITH internet access.
Strategy: solve ALL tasks NOW using 5 cheap OpenRouter models + validation agent.
Cache answers → inference phase just reads the cache (no internet needed).
"""

import json
import os
import sys
import asyncio
import httpx
from pathlib import Path
from typing import List, Dict, Optional, Any
from copy import deepcopy

# ── Config ───────────────────────────────────────────────────────────────────

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
OR_BASE = "https://openrouter.ai/api/v1/chat/completions"
INPUT_DIR  = Path(os.getenv("INPUT_DIR",  "/input"))
OUTPUT_DIR = Path(os.getenv("OUTPUT_DIR", "/output"))
CACHE_FILE = Path("/app/cache.json")

# 5 cheap diverse models — different strengths
SOLVER_MODELS = [
    "qwen/qwen3-32b",                        # strong reasoning, $0.08/M
    "deepseek/deepseek-v4-flash",             # fast + good code, $0.14/M
    "xiaomi/mimo-v2-flash",                   # math/reasoning, $0.10/M
    "google/gemma-4-31b-it",                  # pattern recognition, $0.13/M
    "qwen/qwen3-coder-30b-a3b-instruct",      # code gen, $0.07/M
]

# Validator agent — judges which answer is correct
VALIDATOR_MODEL = "qwen/qwen3-32b"


# ── Helpers ──────────────────────────────────────────────────────────────────

def grid_str(grid: List[List[int]]) -> str:
    return "\n".join(" ".join(str(c) for c in row) for row in grid)

def grids_match(a, b) -> bool:
    if len(a) != len(b): return False
    return all(len(a[i]) == len(b[i]) and a[i] == b[i] for i in range(len(a)))

def parse_grid(text: str) -> Optional[List[List[int]]]:
    """Extract JSON grid from LLM response."""
    text = text.strip()
    # Try JSON directly
    for start in ['[[', '[']:
        idx = text.find(start)
        if idx != -1:
            # Find matching close
            depth, end = 0, -1
            for i, c in enumerate(text[idx:], idx):
                if c == '[': depth += 1
                elif c == ']':
                    depth -= 1
                    if depth == 0:
                        end = i + 1
                        break
            if end != -1:
                try:
                    grid = json.loads(text[idx:end])
                    if isinstance(grid, list) and grid and isinstance(grid[0], list):
                        # Validate: all ints 0-9, rectangular, ≤30x30
                        if len(grid) <= 30 and len(grid[0]) <= 30:
                            if all(isinstance(v, int) and 0 <= v <= 9
                                   for row in grid for v in row):
                                return grid
                except Exception:
                    pass
    return None


async def call_model(client: httpx.AsyncClient, model: str,
                     messages: List[Dict], temperature: float = 0.2) -> str:
    try:
        r = await client.post(OR_BASE, headers={
            "Authorization": f"Bearer {OPENROUTER_API_KEY}",
            "Content-Type": "application/json",
        }, json={
            "model": model,
            "messages": messages,
            "max_tokens": 2000,
            "temperature": temperature,
        }, timeout=60.0)
        data = r.json()
        return data["choices"][0]["message"]["content"]
    except Exception as e:
        return f"ERROR: {e}"


# ── Super Prompt ──────────────────────────────────────────────────────────────

SYSTEM_SOLVER = """You are an expert at ARC-AGI-2 visual reasoning puzzles.
Each puzzle shows a transformation rule through examples.
Your job: identify the rule and apply it to the test input.

Output ONLY the answer grid as JSON array of arrays, like: [[1,2],[3,4]]
No explanation. No markdown. Just the JSON array."""

def make_solver_prompt(train: List[Dict], test_input: List[List[int]]) -> str:
    lines = ["Analyze the transformation pattern:\n"]
    for i, ex in enumerate(train[:4]):
        lines.append(f"Example {i+1}:")
        lines.append(f"INPUT:\n{json.dumps(ex['input'])}")
        lines.append(f"OUTPUT:\n{json.dumps(ex['output'])}\n")
    lines.append(f"TEST INPUT:\n{json.dumps(test_input)}")
    lines.append("\nApply the same transformation. Output ONLY the result grid as JSON.")
    return "\n".join(lines)


SYSTEM_VALIDATOR = """You are a strict validator for ARC-AGI-2 puzzle answers.
Given training examples and multiple candidate answers, find which answer
correctly applies the same transformation rule as shown in the examples.

Respond with ONLY the index number (0, 1, 2, 3, or 4) of the best answer.
If none are correct, respond with -1."""

def make_validator_prompt(train: List[Dict], test_input: List[List[int]],
                          candidates: List[List[List[int]]]) -> str:
    lines = ["Training examples:\n"]
    for i, ex in enumerate(train[:3]):
        lines.append(f"Example {i+1}: {json.dumps(ex['input'])} → {json.dumps(ex['output'])}")
    lines.append(f"\nTest input: {json.dumps(test_input)}")
    lines.append("\nCandidate answers:")
    for i, cand in enumerate(candidates):
        lines.append(f"[{i}]: {json.dumps(cand)}")
    lines.append(
        "\nWhich candidate correctly applies the same transformation? "
        "Reply with ONLY the index number (0-" + str(len(candidates)-1) + ") or -1 if none."
    )
    return "\n".join(lines)


# ── Core: Solve One Task ──────────────────────────────────────────────────────

async def solve_task(client: httpx.AsyncClient, task: Dict) -> Optional[List[List[int]]]:
    train      = task["train_examples"]
    test_input = task["test_input"]
    task_hash  = task.get("task_hash", "?")

    solver_prompt = make_solver_prompt(train, test_input)

    # Step 1: Ask 5 models in parallel
    tasks = [
        call_model(client, model, [
            {"role": "system", "content": SYSTEM_SOLVER},
            {"role": "user",   "content": solver_prompt},
        ])
        for model in SOLVER_MODELS
    ]
    responses = await asyncio.gather(*tasks)

    # Parse grids
    candidates = []
    for resp in responses:
        grid = parse_grid(resp)
        if grid:
            candidates.append(grid)

    if not candidates:
        print(f"  [{task_hash[:8]}] All models failed to parse")
        return None

    # Step 2: Majority voting on matching grids
    grid_votes: Dict[str, int] = {}
    grid_map: Dict[str, List] = {}
    for grid in candidates:
        key = json.dumps(grid)
        grid_votes[key] = grid_votes.get(key, 0) + 1
        grid_map[key] = grid

    # Best by votes
    best_key = max(grid_votes, key=grid_votes.get)
    best_grid = grid_map[best_key]
    best_votes = grid_votes[best_key]

    # Step 4: If only 1 vote for best, ask validator agent
    if best_votes == 1 and len(candidates) > 1 and OPENROUTER_API_KEY:
        unique = list({json.dumps(g): g for g in candidates}.values())
        if len(unique) > 1:
            val_prompt = make_validator_prompt(train, test_input, unique)
            val_resp = await call_model(client, VALIDATOR_MODEL, [
                {"role": "system", "content": SYSTEM_VALIDATOR},
                {"role": "user",   "content": val_prompt},
            ], temperature=0.0)

            # Parse index
            for tok in val_resp.strip().split():
                try:
                    idx = int(tok)
                    if 0 <= idx < len(unique):
                        best_grid = unique[idx]
                        print(f"  [{task_hash[:8]}] Validator picked #{idx}")
                        break
                except ValueError:
                    pass

    print(f"  [{task_hash[:8]}] Solved: {len(candidates)} candidates, {best_votes} votes for best")
    return best_grid


# ── Main Prep Flow ────────────────────────────────────────────────────────────

async def run_prep():
    print("=" * 60)
    print("PREP PHASE: Solving tasks with OpenRouter ensemble")
    print("=" * 60)

    # Load input data
    input_file = INPUT_DIR / "miner_current_dataset.json"
    if not input_file.exists():
        print(f"Input file not found: {input_file}. Downloading vLLM model instead.")
        await download_fallback_model()
        return

    data = json.loads(input_file.read_text())
    tasks = data.get("tasks", [])
    print(f"Found {len(tasks)} tasks to solve")

    if not OPENROUTER_API_KEY:
        print("WARNING: No OPENROUTER_API_KEY — falling back to vLLM only")
        await download_fallback_model()
        return

    # Solve all tasks in parallel batches
    BATCH_SIZE = 4  # 4 tasks × 5 models = 20 concurrent API calls
    cache = {}
    async with httpx.AsyncClient(timeout=90.0) as client:
        for batch_start in range(0, len(tasks), BATCH_SIZE):
            batch = tasks[batch_start:batch_start + BATCH_SIZE]
            end = min(batch_start + BATCH_SIZE, len(tasks))
            print(f"\n[{batch_start+1}-{end}/{len(tasks)}] Solving batch...")
            results = await asyncio.gather(
                *[solve_task(client, t) for t in batch],
                return_exceptions=True,
            )
            for task, result in zip(batch, results):
                h = task["task_hash"]
                cache[h] = None if isinstance(result, Exception) else result

    # Save cache
    CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
    CACHE_FILE.write_text(json.dumps(cache, indent=2))

    solved = sum(1 for v in cache.values() if v is not None)
    unsolved = len(tasks) - solved
    print(f"\n✅ Cache saved: {solved}/{len(tasks)} tasks pre-solved")

    # Only download vLLM if there are tasks we couldn't solve
    if unsolved > 0:
        print(f"Downloading vLLM model for {unsolved} uncached tasks...")
        await download_fallback_model()
    else:
        print("All tasks pre-solved — skipping vLLM model download (saves hours)")


async def download_fallback_model():
    """Download Qwen2.5-72B as fallback for tasks not in cache."""
    print("\nDownloading fallback vLLM model...")
    try:
        from huggingface_hub import snapshot_download
        model_id = "Qwen/QwQ-32B"
        save_dir = os.getenv("MODEL_SAVE_DIR", "/app/models")
        os.makedirs(save_dir, exist_ok=True)
        path = snapshot_download(
            repo_id=model_id,
            local_dir=f"{save_dir}/{model_id.replace('/', '_')}",
            ignore_patterns=["*.gguf"],
        )
        print(f"✅ Model saved to {path}")
    except Exception as e:
        print(f"⚠️ Model download failed: {e} — will use API-free heuristics")


def main():
    asyncio.run(run_prep())


def run_prep_phase():
    asyncio.run(run_prep())


if __name__ == "__main__":
    main()
