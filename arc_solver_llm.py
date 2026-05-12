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
model_name = "Qwen/QwQ-32B"


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
                self.vllm_model = models.data[0].id
                self.vllm_available = True
                print(f"✅ vLLM ready: {self.vllm_model}")
        except Exception as e:
            print(f"⚠️ vLLM unavailable: {e}")

    def solve(self, train_examples: List[Dict], test_input: List[List[int]]) -> List[List[int]]:
        """Main entry point."""
        # 1. Self-consistent program synthesis (K=20, majority vote)
        if self.vllm_available:
            result = self._solve_with_program_synthesis(train_examples, test_input)
            if result and self._is_valid(result):
                print("✅ Solved via program synthesis")
                return result

        # 2. Direct LLM prediction fallback
        if self.vllm_available:
            result = self._solve_direct_llm(train_examples, test_input)
            if result and self._is_valid(result):
                print("✅ Solved via direct LLM")
                return result

        # 3. Last resort: return input unchanged
        print("⚠️ Using identity fallback")
        return [row[:] for row in test_input]

    # ── Program Synthesis ────────────────────────────────────────────────────

    def _solve_with_program_synthesis(self, train: List[Dict], test_input: List[List[int]]) -> Optional[List[List[int]]]:
        """
        Self-consistent program synthesis: sample K=20 programs at 4 temperatures,
        keep only those that pass ALL training pairs, then majority-vote on test output.
        """
        prompt = self._build_synthesis_prompt(train)

        # (temperature, n_samples) — 4 batches × 5 = 20 total candidates
        batches = [(0.2, 5), (0.5, 5), (0.7, 5), (0.9, 5)]
        passing_outputs: List[List[List[int]]] = []
        total_attempts = 0

        for temp, n in batches:
            # Try batched first; fall back to n individual calls if n>1 unsupported
            choices = []
            try:
                resp = self.vllm_client.chat.completions.create(
                    model=self.vllm_model,
                    messages=[
                        {"role": "system", "content": SYNTHESIS_SYSTEM},
                        {"role": "user", "content": prompt},
                    ],
                    temperature=temp,
                    max_tokens=4096,
                    n=n,
                )
                choices = resp.choices
            except Exception as e:
                if n > 1:
                    # n parameter not supported — fall back to individual calls
                    for _ in range(n):
                        try:
                            r = self.vllm_client.chat.completions.create(
                                model=self.vllm_model,
                                messages=[
                                    {"role": "system", "content": SYNTHESIS_SYSTEM},
                                    {"role": "user", "content": prompt},
                                ],
                                temperature=temp,
                                max_tokens=4096,
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

                # Validate on ALL training pairs — deepcopy to prevent in-place mutation
                correct = 0
                for ex in train:
                    pred = self._safe_apply(fn, deepcopy(ex["input"]))
                    if pred is not None and grids_match(pred, ex["output"]):
                        correct += 1

                if correct == len(train):
                    result = self._safe_apply(fn, deepcopy(test_input))
                    if result and self._is_valid(result):
                        passing_outputs.append(result)

        print(f"  Synthesis: {len(passing_outputs)}/{total_attempts} programs passed all train pairs")

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

SYNTHESIS_SYSTEM = """You are an expert at ARC-AGI-2 visual reasoning puzzles.
Analyze the examples, identify the transformation rule, write Python code.

Function: transform(grid: List[List[int]]) -> List[List[int]]
- Standard library only (itertools, math, collections, copy, re)
- Output: 2D list of ints 0-9, size ≤ 30×30
- Must pass ALL shown examples

Common ARC patterns to check:
- Spatial: rotate 90/180/270, flip horizontal/vertical/diagonal
- Scale: zoom 2x/3x (repeat pixels), downsample
- Gravity: move non-zero cells to edge (up/down/left/right)
- Shift: translate grid by N cells in direction
- Color ops: remap color A→B, remove color, highlight one color
- Crop/pad: extract subgrid or add border
- Object-level: find connected components, move/copy objects

Think: same size or different? which colors appear/disappear? spatial shift?
Then write the function inside ```python ... ``` block."""
