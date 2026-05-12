"""
ARC-AGI-2 Solver — Program Synthesis + Heuristics
Strategy:
  1. Ask Qwen2.5-72B to write Python code that transforms grids
  2. Execute the generated code and verify against training examples
  3. If code works on train → apply to test
  4. Fall back to heuristics for simple patterns

No network calls during inference — only local vLLM.
"""

import json
import os
import sys
import traceback
from typing import List, Dict, Optional, Callable
from copy import deepcopy


# Downloaded in prep phase to /app/models or /tmp/models
model_name = "Qwen/Qwen2.5-72B-Instruct"


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
        # 1. Try program synthesis with LLM (most powerful)
        if self.vllm_available:
            result = self._solve_with_program_synthesis(train_examples, test_input)
            if result and self._is_valid(result):
                print("✅ Solved via program synthesis")
                return result

        # 2. Try heuristic transformations
        result = self._solve_with_heuristics(train_examples, test_input)
        if result:
            print("✅ Solved via heuristics")
            return result

        # 3. Fallback: direct LLM prediction
        if self.vllm_available:
            result = self._solve_direct_llm(train_examples, test_input)
            if result and self._is_valid(result):
                print("✅ Solved via direct LLM")
                return result

        # 4. Last resort: return input unchanged
        print("⚠️ Using identity fallback")
        return [row[:] for row in test_input]

    # ── Program Synthesis ────────────────────────────────────────────────────

    def _solve_with_program_synthesis(self, train: List[Dict], test_input: List[List[int]]) -> Optional[List[List[int]]]:
        """Ask LLM to write Python code for the transformation."""
        prompt = self._build_synthesis_prompt(train)

        for attempt in range(3):
            try:
                resp = self.vllm_client.chat.completions.create(
                    model=self.vllm_model,
                    messages=[
                        {"role": "system", "content": SYNTHESIS_SYSTEM},
                        {"role": "user", "content": prompt},
                    ],
                    temperature=0.1 + attempt * 0.2,
                    max_tokens=1500,
                )
                code_text = resp.choices[0].message.content

                # Extract Python code
                code = self._extract_code(code_text)
                if not code:
                    continue

                # Test on training examples
                transform_fn = self._compile_transform(code)
                if not transform_fn:
                    continue

                correct = 0
                for ex in train:
                    try:
                        pred = transform_fn(ex["input"])
                        if grids_match(pred, ex["output"]):
                            correct += 1
                    except Exception:
                        pass

                print(f"  Synthesis attempt {attempt+1}: {correct}/{len(train)} correct")

                if correct == len(train):
                    # Perfect match — apply to test
                    return transform_fn(test_input)
                elif correct >= len(train) * 0.8 and len(train) >= 3:
                    # Good enough — try it
                    try:
                        return transform_fn(test_input)
                    except Exception:
                        pass

            except Exception as e:
                print(f"  Synthesis attempt {attempt+1} failed: {e}")

        return None

    def _build_synthesis_prompt(self, train: List[Dict]) -> str:
        lines = ["Study these input→output transformations:\n"]
        for i, ex in enumerate(train[:4]):
            lines.append(f"Example {i+1}:")
            lines.append(f"Input ({len(ex['input'])}×{len(ex['input'][0])}):")
            lines.append(grid_to_str(ex["input"]))
            lines.append(f"Output ({len(ex['output'])}×{len(ex['output'][0])}):")
            lines.append(grid_to_str(ex["output"]))
            lines.append("")

        lines.append(
            "Write a Python function `transform(grid: List[List[int]]) -> List[List[int]]` "
            "that applies the transformation. The function must work correctly for all examples.\n"
            "Rules: no imports beyond standard library, no recursion deeper than 100, "
            "output must be a 2D list of ints 0-9, max 30×30.\n"
            "Return ONLY the function code inside ```python ... ``` block."
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
        try:
            namespace: Dict = {"List": List, "Dict": Dict, "Optional": Optional}
            # Add common imports
            exec(
                "from typing import List, Dict, Optional, Tuple, Set\n"
                "from copy import deepcopy\n"
                "import itertools, math, collections, functools\n"
                "from collections import Counter, defaultdict\n",
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
                max_tokens=2000,
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

SYNTHESIS_SYSTEM = """You are an expert programmer solving ARC-AGI-2 puzzles.
Your task: write a Python function that transforms input grids to output grids.

Rules for the function:
- Name: transform(grid: List[List[int]]) -> List[List[int]]
- No external imports (only Python standard library)
- Output must be 2D list of integers 0-9
- Output size ≤ 30×30
- Must handle all the shown examples correctly

Common ARC transformations: rotation (90/180/270°), reflection, color remapping,
scaling (2x, 3x, crop), tiling, pattern completion, connected components, etc.

Analyze the examples carefully before writing code."""
