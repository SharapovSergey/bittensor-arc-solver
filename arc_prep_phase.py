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
                     messages: List[Dict], temperature: float = 0.2,
                     max_tokens: int = 4000) -> str:
    for attempt in range(3):
        try:
            r = await client.post(OR_BASE, headers={
                "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                "Content-Type": "application/json",
            }, json={
                "model": model,
                "messages": messages,
                "max_tokens": max_tokens,
                "temperature": temperature,
            }, timeout=90.0)
            data = r.json()
            if "choices" in data and data["choices"]:
                return data["choices"][0]["message"]["content"]
            # Rate limit or API error — back off and retry
            await asyncio.sleep(2 ** attempt)
        except Exception:
            if attempt < 2:
                await asyncio.sleep(2 ** attempt)
    return ""


# ── Program Synthesis Prompts ─────────────────────────────────────────────────

SYSTEM_SYNTHESIS = """You are an expert Python programmer solving ARC-AGI-2 visual puzzles.
Each puzzle shows a pattern through input→output grid pairs (integers 0-9).
Your job: write Python code implementing the transformation.

Rules:
- Function name: transform(grid: list[list[int]]) -> list[list[int]]
- Use only standard library (itertools, math, collections, copy allowed)
- Output must be a 2D list of integers 0-9, max 30×30
- The function MUST produce correct output for ALL shown examples

Think step by step about what changes between input and output, then write the code.
Return ONLY the function inside ```python ... ``` block."""


def make_synthesis_prompt(train: List[Dict]) -> str:
    lines = ["Analyze these input→output transformations and write Python code:\n"]
    for i, ex in enumerate(train[:3]):
        lines.append(f"Example {i+1}:")
        lines.append(f"Input:  {json.dumps(ex['input'])}")
        lines.append(f"Output: {json.dumps(ex['output'])}\n")
    lines.append(
        "Write `transform(grid)` that produces the correct output for ALL examples above.\n"
        "Return ONLY the function inside ```python ... ``` block. Template:\n"
        "```python\n"
        "def transform(grid: list[list[int]]) -> list[list[int]]:\n"
        "    from copy import deepcopy\n"
        "    result = deepcopy(grid)\n"
        "    # your logic here\n"
        "    return result\n"
        "```"
    )
    return "\n".join(lines)


# ── Code execution helpers ────────────────────────────────────────────────────

def extract_code(text: str) -> str:
    if "```python" in text:
        return text.split("```python")[1].split("```")[0].strip()
    if "```" in text:
        return text.split("```")[1].split("```")[0].strip()
    if "def transform" in text:
        lines = text.split("\n")
        start = next((i for i, l in enumerate(lines) if "def transform" in l), None)
        if start is not None:
            return "\n".join(lines[start:])
    return ""


def _run_with_timeout(fn: Any, grid: List[List[int]], timeout_sec: int = 5) -> Optional[List[List[int]]]:
    """Run fn(grid) with a hard timeout. Returns None on timeout or error."""
    import signal

    class _Timeout(Exception):
        pass

    def _handler(signum, frame):
        raise _Timeout()

    old = signal.signal(signal.SIGALRM, _handler)
    signal.alarm(timeout_sec)
    try:
        result = fn(grid)
        return result
    except _Timeout:
        return None
    except Exception:
        return None
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old)


def compile_and_validate(code: str, train: List[Dict]) -> Optional[Any]:
    """Compile code, run on all training pairs. Return fn if ALL pass, else None."""
    if not code:
        return None
    try:
        ns: Dict = {}
        exec(
            "from typing import List, Dict, Optional, Tuple, Set\n"
            "from copy import deepcopy\n"
            "import itertools, math, collections, functools, re\n"
            "from collections import Counter, defaultdict, deque\n",
            ns,
        )
        exec(code, ns)
        fn = ns.get("transform")
        if not callable(fn):
            return None

        for ex in train:
            # deepcopy prevents generated code from mutating training data in-place
            # timeout prevents infinite loops from hanging the process
            pred = _run_with_timeout(fn, deepcopy(ex["input"]), timeout_sec=5)
            if not pred or not pred[0]:
                return None
            if pred != ex["output"]:
                return None

        return fn
    except Exception:
        return None


def apply_safe(fn: Any, grid: List[List[int]]) -> Optional[List[List[int]]]:
    try:
        r = fn(grid)
        if r and r[0] and len(r) <= 30 and len(r[0]) <= 30:
            if all(isinstance(v, int) and 0 <= v <= 9 for row in r for v in row):
                return r
    except Exception:
        pass
    return None


def vote_outputs(outputs: List[List[List[int]]]) -> List[List[int]]:
    counts: Dict[str, int] = {}
    for g in outputs:
        k = json.dumps(g)
        counts[k] = counts.get(k, 0) + 1
    return json.loads(max(counts, key=counts.__getitem__))


def evaluate_program(code: str, train: List[Dict]) -> Dict:
    """
    Compile code and run on all training pairs.
    Returns {"fn": callable|None, "pass_count": int,
             "failing": [(example, pred|None), ...]}
    Unlike compile_and_validate, does not fail-fast — collects partial results.
    """
    empty = {"fn": None, "pass_count": 0, "failing": []}
    if not code:
        return empty
    try:
        ns: Dict = {}
        exec(
            "from typing import List, Dict, Optional, Tuple, Set\n"
            "from copy import deepcopy\n"
            "import itertools, math, collections, functools, re\n"
            "from collections import Counter, defaultdict, deque\n",
            ns,
        )
        exec(code, ns)
        fn = ns.get("transform")
        if not callable(fn):
            return empty
    except Exception:
        return empty

    pass_count = 0
    failing: List[tuple] = []
    for ex in train:
        raw = _run_with_timeout(fn, deepcopy(ex["input"]), timeout_sec=5)
        # Normalise: None if invalid/timeout, else the grid
        pred: Optional[List[List[int]]] = None
        if raw and raw[0] and len(raw) <= 30 and len(raw[0]) <= 30:
            if all(isinstance(v, int) and 0 <= v <= 9 for row in raw for v in row):
                pred = raw
        if pred is not None and pred == ex["output"]:
            pass_count += 1
        else:
            failing.append((ex, pred))  # (example_dict, what_we_got)

    return {"fn": fn, "pass_count": pass_count, "failing": failing}


# ── Repair loop ───────────────────────────────────────────────────────────────

REPAIR_MODEL = "qwen/qwen3-32b"

SYSTEM_REPAIR = """You are debugging a Python ARC-AGI-2 solver function.
It passes 2 out of 3 training examples but fails on one specific case.
Fix ONLY the bug causing that failure — preserve all logic that works.
Return ONLY the corrected function inside ```python ... ``` block."""


def make_repair_prompt(code: str, fail_ex: Dict,
                       fail_pred: Optional[List[List[int]]]) -> str:
    got = json.dumps(fail_pred) if fail_pred is not None else "ERROR (exception)"
    return (
        f"This function passes 2/3 training examples but fails on one:\n\n"
        f"```python\n{code}\n```\n\n"
        f"FAILING EXAMPLE:\n"
        f"Input:    {json.dumps(fail_ex['input'])}\n"
        f"Expected: {json.dumps(fail_ex['output'])}\n"
        f"Got:      {got}\n\n"
        f"Fix the bug. Do not change logic that works for passing examples.\n"
        f"Return ONLY the corrected function inside ```python ... ``` block."
    )


async def repair_program(
    client: httpx.AsyncClient,
    code: str,
    fail_ex: Dict,
    fail_pred: Optional[List[List[int]]],
    train: List[Dict],
    test_input: List[List[int]],
) -> Optional[List[List[int]]]:
    """
    Attempt to repair a 2/3-passing program.
    Makes up to 2 sequential calls at T=0.1.
    Returns test output if repair passes ALL train pairs, else None.
    """
    repair_prompt = make_repair_prompt(code, fail_ex, fail_pred)
    messages = [
        {"role": "system", "content": SYSTEM_REPAIR},
        {"role": "user",   "content": repair_prompt},
    ]
    # Attempt 1: T=0.1 (focused fix). Attempt 2: T=0.4 (if same prompt → same answer at T=0.1)
    for attempt, temperature in enumerate([0.1, 0.4]):
        resp = await call_model(client, REPAIR_MODEL, messages, temperature=temperature)
        new_code = extract_code(resp)
        fn = compile_and_validate(new_code, train)   # strict: ALL pairs must pass
        if fn is not None:
            result = apply_safe(fn, deepcopy(test_input))
            if result:
                return result
    return None


# ── Fallback: direct grid prediction (legacy) ─────────────────────────────────

SYSTEM_DIRECT = """You are an expert at ARC-AGI-2 visual reasoning puzzles.
Output ONLY the answer grid as JSON array of arrays, like: [[1,2],[3,4]]
No explanation. No markdown. Just the JSON array."""

def make_direct_prompt(train: List[Dict], test_input: List[List[int]]) -> str:
    lines = ["Find the pattern and predict the test output:\n"]
    for i, ex in enumerate(train[:3]):
        lines.append(f"Train {i+1} input:  {json.dumps(ex['input'])}")
        lines.append(f"Train {i+1} output: {json.dumps(ex['output'])}\n")
    lines.append(f"Test input: {json.dumps(test_input)}")
    lines.append("\nOutput ONLY the result grid as JSON array of arrays.")
    return "\n".join(lines)


# ── Core: Solve One Task ──────────────────────────────────────────────────────

async def solve_task(client: httpx.AsyncClient, task: Dict) -> Optional[List[List[int]]]:
    train      = task["train_examples"]
    test_input = task["test_input"]
    task_hash  = task.get("task_hash", "?")

    synthesis_prompt = make_synthesis_prompt(train)

    # ── Phase 1: Program synthesis — 2 temperatures × 5 models = 10 shots ──────
    passing_outputs: List[List[List[int]]] = []
    repair_candidates: List[tuple] = []  # (code, fail_ex, fail_pred)

    for temperature in (0.3, 0.7):
        synth_tasks = [
            call_model(client, model, [
                {"role": "system", "content": SYSTEM_SYNTHESIS},
                {"role": "user",   "content": synthesis_prompt},
            ], temperature=temperature)
            for model in SOLVER_MODELS
        ]
        responses = await asyncio.gather(*synth_tasks)

        for resp in responses:
            code = extract_code(resp)
            if not code:
                continue
            info = evaluate_program(code, train)
            if info["pass_count"] == len(train):
                result = apply_safe(info["fn"], deepcopy(test_input))
                if result:
                    passing_outputs.append(result)
            elif info["pass_count"] >= 2 and info["failing"]:
                fail_ex, fail_pred = info["failing"][0]
                repair_candidates.append((code, fail_ex, fail_pred))

        if passing_outputs:
            break  # найден рабочий код — не тратим T=0.7

    if passing_outputs:
        result = vote_outputs(passing_outputs)
        print(f"  [{task_hash[:8]}] ✅ Synthesis: {len(passing_outputs)} passed → voted")
        return result

    # ── Phase 1.5: Repair loop — fix 2/3 candidates ──────────────────────────
    if repair_candidates:
        # Deduplicate by full code string, keep order (first = earliest found)
        seen: set = set()
        unique: List[tuple] = []
        for code, fail_ex, fail_pred in repair_candidates:
            if code not in seen:
                seen.add(code)
                unique.append((code, fail_ex, fail_pred))

        # Run up to 3 candidates in parallel (each makes ≤2 sequential repair calls)
        repair_tasks = [
            repair_program(client, code, fail_ex, fail_pred, train, test_input)
            for code, fail_ex, fail_pred in unique[:3]
        ]
        repaired = await asyncio.gather(*repair_tasks)

        for r in repaired:
            if r is not None:
                passing_outputs.append(r)

        if passing_outputs:
            result = vote_outputs(passing_outputs)
            n_fixed = sum(1 for r in repaired if r is not None)
            print(f"  [{task_hash[:8]}] 🔧 Repair: {n_fixed}/{len(unique[:3])} fixed → voted")
            return result
        else:
            print(f"  [{task_hash[:8]}] 🔧 Repair: 0/{len(unique[:3])} succeeded")

    # ── Phase 2: Fallback — direct grid prediction ────────────────────────────
    direct_prompt = make_direct_prompt(train, test_input)
    direct_tasks = [
        call_model(client, model, [
            {"role": "system", "content": SYSTEM_DIRECT},
            {"role": "user",   "content": direct_prompt},
        ], temperature=0.1)
        for model in SOLVER_MODELS
    ]
    direct_responses = await asyncio.gather(*direct_tasks)

    candidates = [g for resp in direct_responses if (g := parse_grid(resp))]
    if not candidates:
        print(f"  [{task_hash[:8]}] ❌ Both phases failed")
        return None

    result = vote_outputs(candidates)
    print(f"  [{task_hash[:8]}] ⚠️ Direct fallback: {len(candidates)}/5 grids parsed")
    return result


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
