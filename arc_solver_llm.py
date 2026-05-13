"""
ARC-AGI-2 Solver — Self-Consistent Program Synthesis
Strategy:
  1. Sample K=20 programs from QwQ-32B at 4 temperatures
  2. Keep only programs that pass ALL training pairs (pixel-exact)
  3. Majority-vote test outputs across passing programs
  4. Fall back to direct LLM prediction if no program validates

No network calls during inference — only local vLLM.
"""

import json
import os
from typing import List, Dict, Optional, Callable
from copy import deepcopy


# Downloaded in prep phase to /app/models or /tmp/models
# Model: da-fr/Mistral-NeMo-Minitron-8B-ARChitects-Full-bnb-4bit (3.5GB, 4-bit)
# Decision (2026-05-13, Opus 4.7): 2024 ARChitects winner has the TTT recipe
# written specifically for it; 4-bit fits comfortably in H200 with room for TTT.
# Source: /Users/sharapov/Cloude/Project X/nvarc_implementation_plan.md
model_name = "da-fr/Mistral-NeMo-Minitron-8B-ARChitects-Full-bnb-4bit"


def grid_to_str(grid: List[List[int]]) -> str:
    """Human-readable grid representation."""
    return "\n".join(" ".join(str(c) for c in row) for row in grid)


def grids_match(a: List[List[int]], b: List[List[int]]) -> bool:
    if len(a) != len(b): return False
    return all(len(a[i]) == len(b[i]) and a[i] == b[i] for i in range(len(a)))


class ARCSolver:
    def __init__(self, use_vllm: bool = True):
        self.use_vllm = use_vllm
        self.vllm_client = None
        self.vllm_model = None
        self.vllm_available = False
        self.strategies = [
            self._try_identity,
            self._try_color_mapping,
            self._try_rotation,
            self._try_flip,
            self._try_crop,
        ]
        if use_vllm:
            self._init_vllm()

    def _init_vllm(self):
        try:
            from openai import OpenAI
            base = os.environ.get("VLLM_API_BASE", "http://vllm-container:8000")
            self.vllm_client = OpenAI(base_url=f"{base}/v1", api_key="local")
            models = self.vllm_client.models.list()
            if models.data:
                # vLLM serves the merged model (Mistral-NeMo-8B with TTT weights baked in).
                # See sn5_miner_server.py /info — vllm_config.model = "mistral-ttt-merged"
                self.vllm_model = models.data[0].id
                self.vllm_available = True
                print(f"✅ vLLM ready: {self.vllm_model}")
        except Exception as e:
            print(f"⚠️ vLLM unavailable: {e}")

    def solve(self, train_examples: List[Dict], test_input: List[List[int]],
              time_budget_sec: Optional[float] = None) -> List[List[int]]:
        """
        Main entry point.
        time_budget_sec: soft limit on total time for this task. If exceeded,
                        skip remaining steps and return best so far (or identity).
        """
        import time
        start = time.monotonic()
        deadline = (start + time_budget_sec) if time_budget_sec else None

        def time_left() -> float:
            return float("inf") if deadline is None else max(0.0, deadline - time.monotonic())

        # 0. Chain inversion (zero LLM, ~1s) — symbolic solve if base_fn=identity
        if os.getenv("ENABLE_CHAIN_INVERSION", "1") == "1":
            try:
                from arc_chain_inverter import (
                    find_inverse_chain, forward_chain_from_inverse, apply_forward_chain
                )
                inv_chain = find_inverse_chain(train_examples, max_depth=4,
                                               require_identity_match=True)
                if inv_chain:
                    forward_fns = forward_chain_from_inverse(inv_chain)
                    predicted = apply_forward_chain(test_input, forward_fns)
                    if predicted and self._is_valid(predicted):
                        print(f"⚡ Chain inversion: {inv_chain} ({time.monotonic()-start:.0f}s)")
                        return predicted
            except Exception as e:
                print(f"  chain inversion error: {e}")

        # 1. Self-consistent program synthesis (K=3 with diversity, majority vote)
        if self.vllm_available and time_left() > 5:
            result = self._solve_with_program_synthesis(
                train_examples, test_input, deadline=deadline
            )
            if result and self._is_valid(result):
                print(f"✅ Solved via program synthesis ({time.monotonic()-start:.0f}s)")
                return result

        # 2. Direct LLM prediction fallback (only if budget remains)
        if self.vllm_available and time_left() > 15:
            result = self._solve_direct_llm(train_examples, test_input)
            if result and self._is_valid(result):
                print(f"✅ Solved via direct LLM ({time.monotonic()-start:.0f}s)")
                return result

        # 3. Last resort: return input unchanged
        print(f"⚠️ Identity fallback ({time.monotonic()-start:.0f}s, budget_left={time_left():.0f}s)")
        return [row[:] for row in test_input]

    # ── Program Synthesis ────────────────────────────────────────────────────

    def _solve_with_program_synthesis(
        self,
        train: List[Dict],
        test_input: List[List[int]],
        deadline: Optional[float] = None,
    ) -> Optional[List[List[int]]]:
        """
        Self-consistent program synthesis: sample K=3 programs at diverse temperatures,
        keep only those that pass ALL training pairs, then majority-vote on test output.

        K reduced from 20 to 3 to fit inference budget: at QwQ-32B speed (~30 tok/s
        with thinking), K=3 × 2048 tokens ≈ 60-90s per task. Diversity preserved via
        temperature spread (0.3, 0.6, 0.9).

        deadline: absolute time.monotonic() value. If exceeded, stop early.
        """
        import time
        prompt = self._build_synthesis_prompt(train)

        # K=3 with temperature diversity (replaces K=20 which was 5-10× over budget)
        batches = [(0.3, 1), (0.6, 1), (0.9, 1)]
        passing_outputs: List[List[List[int]]] = []
        repair_candidates: List[tuple] = []  # (code, fail_ex, fail_pred)
        total_attempts = 0

        for temp, n in batches:
            # Stop early if deadline exceeded
            if deadline is not None and time.monotonic() >= deadline:
                print(f"  Deadline reached, stopping synthesis at {total_attempts} attempts")
                break
            # Stop early if we already have a passing program (no need to vote with 1 candidate)
            if passing_outputs:
                break
            choices = []
            try:
                resp = self.vllm_client.chat.completions.create(
                    model=self.vllm_model,
                    messages=[
                        {"role": "system", "content": SYNTHESIS_SYSTEM},
                        {"role": "user", "content": prompt},
                    ],
                    temperature=temp,
                    max_tokens=2048,  # reduced from 4096 — limits reasoning model thinking
                    n=n,
                )
                choices = resp.choices
            except Exception as e:
                if n > 1:
                    for _ in range(n):
                        try:
                            r = self.vllm_client.chat.completions.create(
                                model=self.vllm_model,
                                messages=[
                                    {"role": "system", "content": SYNTHESIS_SYSTEM},
                                    {"role": "user", "content": prompt},
                                ],
                                temperature=temp,
                                max_tokens=2048,
                            )
                            choices.extend(r.choices)
                        except Exception:
                            pass
                else:
                    print(f"  Synthesis batch T={temp} failed: {e}")

            for choice in choices:
                total_attempts += 1
                code = self._extract_code(choice.message.content)
                if not code:
                    continue
                fn = self._compile_transform(code)
                if not fn:
                    continue

                info = self._evaluate_program(fn, train)
                if info["pass_count"] == len(train):
                    result = self._safe_apply(fn, deepcopy(test_input))
                    if result and self._is_valid(result):
                        passing_outputs.append(result)
                elif info["pass_count"] >= 2 and info["failing"]:
                    fail_ex, fail_pred = info["failing"][0]
                    repair_candidates.append((code, fail_ex, fail_pred))

        print(f"  Synthesis: {len(passing_outputs)}/{total_attempts} programs passed all train pairs")

        # ── LOO generalization filter (inference) ─────────────────────────────
        # DEFAULT DISABLED (false-positive rate too high — see prep phase comment).
        # To re-enable: ENABLE_LOO=1
        if passing_outputs and os.getenv("ENABLE_LOO", "0") == "1":
            if not self._loo_verify(train, deadline=deadline):
                print(f"  LOO filter: rejected {len(passing_outputs)} overfit candidates")
                passing_outputs = []

        # ── Repair loop (inference: 1 best candidate, 2 attempts) ────────────
        # Skip if deadline tight — repair takes 30-60s for 2 vLLM calls
        budget_ok = deadline is None or (deadline - time.monotonic()) > 40
        if not passing_outputs and repair_candidates and self.vllm_available and budget_ok:
            seen: set = set()
            unique: List[tuple] = []
            for code, fail_ex, fail_pred in repair_candidates:
                if code not in seen:
                    seen.add(code)
                    unique.append((code, fail_ex, fail_pred))

            # Only repair the first unique candidate — inference is time-constrained
            code, fail_ex, fail_pred = unique[0]
            print(f"  Repair: attempting to fix 2/3 program ({len(unique)} candidates, using first)...")
            fn = self._attempt_repair(code, fail_ex, fail_pred, train)
            if fn is not None:
                result = self._safe_apply(fn, deepcopy(test_input))
                if result and self._is_valid(result):
                    passing_outputs.append(result)
                    print("  Repair: ✅ success")
            if not passing_outputs:
                print("  Repair: ❌ failed")

        if not passing_outputs:
            return None
        if len(passing_outputs) == 1:
            return passing_outputs[0]

        # Majority vote on test outputs
        vote: Dict[str, int] = {}
        for out in passing_outputs:
            key = json.dumps(out)
            vote[key] = vote.get(key, 0) + 1

        best_key = max(vote, key=vote.__getitem__)
        print(f"  Vote: {vote[best_key]}/{len(passing_outputs)} for winning output")
        return json.loads(best_key)

    def _safe_apply(self, fn: Callable, grid: List[List[int]]) -> Optional[List[List[int]]]:
        try:
            result = fn(grid)
            if not result or not result[0]:
                return None
            if len(result) > 30 or len(result[0]) > 30:
                return None
            if not all(isinstance(v, int) and 0 <= v <= 9 for row in result for v in row):
                return None
            return result
        except Exception:
            return None

    def _loo_verify(self, train: List[Dict], deadline: Optional[float] = None) -> bool:
        """
        LOO check: synthesize 3 programs each seeing only 2 train pairs,
        verify each correctly predicts the held-out 3rd pair.
        Returns True if all 3 LOO programs pass their held-out test.
        Sequential (3 vLLM calls) — uses ~30-60s budget.
        """
        if len(train) != 3 or not self.vllm_available:
            return True
        import time

        for hold_idx in range(3):
            if deadline is not None and time.monotonic() >= deadline:
                return True   # out of time, give benefit of doubt
            loo_train = [train[i] for i in range(3) if i != hold_idx]
            held_out = train[hold_idx]
            prompt = self._build_synthesis_prompt(loo_train)
            try:
                resp = self.vllm_client.chat.completions.create(
                    model=self.vllm_model,
                    messages=[
                        {"role": "system", "content": SYNTHESIS_SYSTEM},
                        {"role": "user", "content": prompt},
                    ],
                    temperature=0.3, max_tokens=2048, n=1,
                )
                code = self._extract_code(resp.choices[0].message.content)
                fn = self._compile_transform(code)
                if fn is None:
                    return False
                # Run on held-out — must produce held_out.output
                pred = self._safe_apply(fn, deepcopy(held_out["input"]))
                if pred is None or not grids_match(pred, held_out["output"]):
                    return False
            except Exception:
                return False
        return True

    def _evaluate_program(self, fn: Callable, train: List[Dict]) -> Dict:
        """
        Run fn on all training pairs without fail-fast.
        Returns {"pass_count": int, "failing": [(ex, pred|None), ...]}
        """
        pass_count = 0
        failing: List[tuple] = []
        for ex in train:
            pred = self._safe_apply(fn, deepcopy(ex["input"]))
            if pred is not None and grids_match(pred, ex["output"]):
                pass_count += 1
            else:
                failing.append((ex, pred))
        return {"pass_count": pass_count, "failing": failing}

    def _attempt_repair(self, code: str, fail_ex: Dict,
                        fail_pred: Optional[List[List[int]]],
                        train: List[Dict]) -> Optional[Callable]:
        """
        Try to repair a 2/3-passing program via vLLM.
        Makes up to 2 sequential calls at T=0.1.
        Returns fn that passes ALL train pairs, or None.
        """
        repair_prompt = _make_repair_prompt(code, fail_ex, fail_pred)
        messages = [
            {"role": "system", "content": REPAIR_SYSTEM},
            {"role": "user",   "content": repair_prompt},
        ]
        # Attempt 1: T=0.1 (focused). Attempt 2: T=0.4 (avoid same output as attempt 1)
        for attempt, temperature in enumerate([0.1, 0.4]):
            try:
                resp = self.vllm_client.chat.completions.create(
                    model=self.vllm_model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=4096,
                )
                new_code = self._extract_code(resp.choices[0].message.content)
                fn = self._compile_transform(new_code)
                if fn is None:
                    continue
                info = self._evaluate_program(fn, train)
                if info["pass_count"] == len(train):
                    return fn
            except Exception as e:
                print(f"  Repair attempt {attempt + 1} failed: {e}")
        return None

    def _build_synthesis_prompt(self, train: List[Dict]) -> str:
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

    def _extract_code(self, text: str) -> str:
        """Extract Python code from LLM response."""
        if "```python" in text:
            return text.split("```python")[1].split("```")[0].strip()
        if "```" in text:
            return text.split("```")[1].split("```")[0].strip()
        # Look for def transform
        if "def transform" in text:
            lines = text.split("\n")
            start = next((i for i, l in enumerate(lines) if "def transform" in l), None)
            if start is not None:
                return "\n".join(lines[start:])
        return ""

    def _compile_transform(self, code: str) -> Optional[Callable]:
        """Compile and return the transform function."""
        if not code:
            return None
        try:
            namespace: Dict = {}
            exec(
                "from typing import List, Dict, Optional, Tuple, Set\n"
                "from copy import deepcopy\n"
                "import itertools, math, collections, functools, re\n"
                "from collections import Counter, defaultdict, deque\n",
                namespace,
            )
            exec(code, namespace)
            fn = namespace.get("transform")
            if callable(fn):
                return fn
        except Exception as e:
            print(f"  Compile error: {e}")
        return None

    # ── Direct LLM Prediction ────────────────────────────────────────────────

    def _solve_direct_llm(self, train: List[Dict], test_input: List[List[int]]) -> Optional[List[List[int]]]:
        """Direct prediction without code generation."""
        lines = ["ARC puzzle. Find the output for the test input.\n"]
        for i, ex in enumerate(train[:3]):
            lines.append(f"Train {i+1} input:\n{json.dumps(ex['input'])}")
            lines.append(f"Train {i+1} output:\n{json.dumps(ex['output'])}\n")
        lines.append(f"Test input:\n{json.dumps(test_input)}")
        lines.append("\nReturn ONLY the output grid as JSON array of arrays. No explanation.")

        try:
            resp = self.vllm_client.chat.completions.create(
                model=self.vllm_model,
                messages=[
                    {"role": "system", "content": "You solve ARC-AGI puzzles. Return only valid JSON grids."},
                    {"role": "user", "content": "\n".join(lines)},
                ],
                temperature=0.0,
                max_tokens=4096,
            )
            content = resp.choices[0].message.content.strip()
            # Extract JSON
            if "```" in content:
                content = content.split("```")[1].split("```")[0]
                if content.startswith("json"):
                    content = content[4:]
            grid = json.loads(content.strip())
            if isinstance(grid, list) and all(isinstance(r, list) for r in grid):
                return grid
        except Exception as e:
            print(f"  Direct LLM failed: {e}")
        return None

    # ── Heuristic Solvers ────────────────────────────────────────────────────

    def _solve_with_heuristics(self, train: List[Dict], test_input: List[List[int]]) -> Optional[List[List[int]]]:
        """Try all heuristic strategies, verify against training."""
        for strategy in self.strategies:
            try:
                pred = strategy(test_input, train)
                if pred and self._is_valid(pred):
                    # Verify on training
                    correct = sum(1 for ex in train if grids_match(strategy(ex["input"], train), ex["output"]))
                    if correct == len(train):
                        return pred
            except Exception:
                pass
        return None

    def _try_identity(self, grid, examples=None):
        return [r[:] for r in grid]

    def _try_color_mapping(self, grid, examples=None):
        if not examples: return None
        mapping: Dict[int, int] = {}
        for ex in examples:
            flat_in  = [v for r in ex["input"]  for v in r]
            flat_out = [v for r in ex["output"] for v in r]
            if len(flat_in) != len(flat_out): return None
            for a, b in zip(flat_in, flat_out):
                if a in mapping and mapping[a] != b: return None
                mapping[a] = b
        return [[mapping.get(v, v) for v in r] for r in grid]

    def _try_rotation(self, grid, examples=None):
        h, w = len(grid), len(grid[0])
        return [[grid[h-1-j][i] for j in range(h)] for i in range(w)]

    def _try_flip(self, grid, examples=None):
        return [r[::-1] for r in grid]

    def _try_crop(self, grid, examples=None):
        if not examples: return None
        out = examples[0]["output"]
        th, tw = len(out), len(out[0])
        if th <= len(grid) and tw <= len(grid[0]):
            return [r[:tw] for r in grid[:th]]
        return None

    # ── Validation ───────────────────────────────────────────────────────────

    def _is_valid(self, grid) -> bool:
        if not grid or not grid[0]: return False
        if len(grid) > 30 or len(grid[0]) > 30: return False
        w = len(grid[0])
        return all(len(r) == w and all(isinstance(v, int) and 0 <= v <= 9 for v in r) for r in grid)


# ── Prompts ──────────────────────────────────────────────────────────────────

SYNTHESIS_SYSTEM = """You will be given some number of paired example inputs and outputs.
The outputs were produced by applying a transformation rule to the inputs.

Start your response by carefully reasoning in <reasoning></reasoning> tags about
what the transformation does. Then write the transformation in Python code.

You should write a function called `transform` which takes a single argument,
the input grid as `list[list[int]]`, and returns the transformed grid.

Don't write tests in your python code, just output the `transform` function.

You are creative and accomplished at solving puzzles.
When you write `transform`, do NOT hardcode the solution for each example —
your function must apply the same logic to any input grid following the same rule.

Common ARC patterns to consider:
- Spatial: rotate 90/180/270, flip horizontal/vertical/diagonal
- Scale: zoom 2x/3x (repeat pixels), downsample
- Gravity: move non-zero cells to edge (up/down/left/right)
- Shift: translate grid by N cells in direction
- Color ops: remap, swap, remove, highlight
- Object-level: find connected components, move/copy/recolor objects

Wrap the function in ```python ... ``` block. Standard library only."""


REPAIR_SYSTEM = """You are refining a Python ARC-AGI-2 solver.
The function passes 2/3 training examples but fails on one.

First, reflect on what was correct and what was wrong in <reflection></reflection> tags.
Then write the corrected function.

DO NOT hardcode output into your `transform` function and return it for each example —
the function must apply the same logic to any input grid.

Wrap the corrected function in ```python ... ``` block."""


def _make_repair_prompt(code: str, fail_ex: Dict,
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
