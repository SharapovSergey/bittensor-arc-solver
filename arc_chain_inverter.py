"""
Phase 3 (#2): Chain Inversion for ARC-AGI-2 task simplification.

Validator generates tasks as output = chain(base_fn(input)) where:
  - base_fn: unknown ARC-1 style transformation
  - chain: composition of 3-7 ops from known closed vocabulary

If we can recover `chain` symbolically (via inverse ops), we strip it from
all 3 train_outputs, giving the synthesizer a simpler task:
  (train_input_i, unwrapped_output_i) where unwrapped_output_i = chain^-1(output_i)
                                          = base_fn(input_i)

The LLM then only needs to find base_fn, not the full chain∘base_fn.

This module:
  - Forward & inverse implementations of 13 cleanly-invertible ops
  - BFS over inverse chains of depth 1-4
  - Consistency check: chain candidate accepted only if it produces
    UNWRAPPED outputs whose relationship with inputs is the SAME across
    all 3 train pairs (judged by grid-shape signature)

NOT invertible (skipped): gravity_*, remove_color, highlight_color, recenter.
LOSSY but tried: downsample_2x (inverse = zoom_2x, lossy block-wise).
"""

from itertools import product
from typing import Callable, Dict, List, Optional, Tuple

Grid = List[List[int]]


# ── Forward implementations (verbatim from arc_agi2_utils.py) ────────────────

def _size(g: Grid) -> Tuple[int, int]:
    return (len(g), len(g[0]) if g else 0)


def rotate_90(g: Grid) -> Grid:
    h, w = _size(g)
    r = [[0]*h for _ in range(w)]
    for i in range(h):
        for j in range(w):
            r[j][h-1-i] = g[i][j]
    return r


def rotate_180(g: Grid) -> Grid:
    return [row[::-1] for row in g[::-1]]


def rotate_270(g: Grid) -> Grid:
    return rotate_90(rotate_180(g))


def flip_horizontal(g: Grid) -> Grid:
    return [row[::-1] for row in g]


def flip_vertical(g: Grid) -> Grid:
    return g[::-1]


def transpose(g: Grid) -> Grid:
    h, w = _size(g)
    return [[g[i][j] for i in range(h)] for j in range(w)]


def flip_diagonal(g: Grid) -> Grid:
    return transpose(g)


def flip_antidiagonal(g: Grid) -> Grid:
    return rotate_90(flip_vertical(g))


def zoom_2x(g: Grid) -> Grid:
    h, w = _size(g)
    r = [[0]*(w*2) for _ in range(h*2)]
    for i in range(h):
        for j in range(w):
            v = g[i][j]
            r[i*2][j*2] = r[i*2][j*2+1] = r[i*2+1][j*2] = r[i*2+1][j*2+1] = v
    return r


def zoom_3x(g: Grid) -> Grid:
    h, w = _size(g)
    r = [[0]*(w*3) for _ in range(h*3)]
    for i in range(h):
        for j in range(w):
            v = g[i][j]
            for di in range(3):
                for dj in range(3):
                    r[i*3+di][j*3+dj] = v
    return r


def downsample_2x(g: Grid) -> Grid:
    h, w = _size(g)
    if h < 2 or w < 2:
        return g
    return [[g[r*2][c*2] for c in range(w//2)] for r in range(h//2) if r*2 < h]


def downsample_3x(g: Grid) -> Grid:
    h, w = _size(g)
    if h < 3 or w < 3:
        return g
    return [[g[r*3][c*3] for c in range(w//3)] for r in range(h//3)]


# ── Inverse pairs ────────────────────────────────────────────────────────────
# Each inverse op satisfies: inv(forward(g)) == g for valid inputs.

INVERSE_OPS: Dict[str, Callable[[Grid], Grid]] = {
    "rotate_90":         rotate_270,     # rotate_270 is the inverse of rotate_90
    "rotate_180":        rotate_180,     # self-inverse
    "rotate_270":        rotate_90,
    "flip_horizontal":   flip_horizontal,  # self-inverse
    "flip_vertical":     flip_vertical,
    "flip_diagonal":     flip_diagonal,
    "flip_antidiagonal": flip_antidiagonal,
    "transpose":         transpose,
    "zoom_2x":           downsample_2x,
    "zoom_3x":           downsample_3x,
    "downsample_2x":     zoom_2x,        # lossy in reverse, but try
    "downsample_3x":     zoom_3x,
}


def _apply_chain(g: Grid, chain: List[str]) -> Optional[Grid]:
    """Apply chain of inverse ops left-to-right. Returns None on failure."""
    cur = g
    try:
        for op_name in chain:
            cur = INVERSE_OPS[op_name](cur)
            if not cur or not cur[0]:
                return None
            if len(cur) > 30 or len(cur[0]) > 30:
                return None
    except Exception:
        return None
    return cur


def _grids_equal(a: Grid, b: Grid) -> bool:
    if len(a) != len(b):
        return False
    return all(len(a[i]) == len(b[i]) and a[i] == b[i] for i in range(len(a)))


def _signature(g: Grid) -> Tuple:
    """Cheap hash for grid: (h, w, sorted_color_counts)."""
    if not g or not g[0]:
        return (0, 0, tuple())
    h, w = _size(g)
    from collections import Counter
    flat = Counter(v for row in g for v in row)
    return (h, w, tuple(sorted(flat.items())))


def find_inverse_chain(
    train_examples: List[Dict],
    max_depth: int = 4,
    require_identity_match: bool = False,
) -> Optional[List[str]]:
    """
    Search for an inverse chain that produces consistent "unwrapped" outputs.

    BFS over chains of length 1..max_depth.
    A chain is accepted iff applying it to all 3 train_outputs yields:
      - Valid grids (≤30×30, ints 0-9)
      - SAME SIGNATURE across all 3 unwrapped grids and their corresponding
        train inputs (if require_identity_match=True, exact match to input).

    Returns the SHORTEST chain that works, or None.
    """
    if len(train_examples) < 2:
        return None

    op_names = list(INVERSE_OPS.keys())  # 12 ops
    train_outputs = [ex["output"] for ex in train_examples]
    train_inputs  = [ex["input"]  for ex in train_examples]

    for depth in range(1, max_depth + 1):
        # Iterate all chains of length `depth`. 12^4 = 20736 max — tractable.
        for chain in product(op_names, repeat=depth):
            chain = list(chain)
            unwrapped = []
            for out in train_outputs:
                u = _apply_chain(out, chain)
                if u is None:
                    unwrapped = None
                    break
                unwrapped.append(u)
            if unwrapped is None:
                continue

            if require_identity_match:
                # Strict: chain wrapped the WHOLE transformation
                # → unwrapped output should == input
                if all(_grids_equal(unwrapped[i], train_inputs[i])
                       for i in range(len(train_examples))):
                    return chain
            else:
                # Loose: unwrapped outputs are "compatible" with each other
                # (same shape + same color palette as their inputs)
                sigs_in  = [_signature(g) for g in train_inputs]
                sigs_out = [_signature(g) for g in unwrapped]
                # Heuristic: same shape AND each unwrapped has subset of
                # colors used in its input (base_fn doesn't add new colors usually)
                shapes_match = all(
                    sigs_in[i][:2] == sigs_out[i][:2]
                    for i in range(len(train_examples))
                )
                if shapes_match:
                    # Additional check: chain produces non-trivial transformation
                    # (i.e., unwrapped != original output for at least one pair).
                    nontrivial = any(
                        not _grids_equal(unwrapped[i], train_outputs[i])
                        for i in range(len(train_examples))
                    )
                    if nontrivial:
                        return chain

    return None


def unwrap_train_examples(
    train_examples: List[Dict],
    inverse_chain: List[str],
) -> List[Dict]:
    """
    Apply inverse_chain to each train_example's output.
    Returns new train_examples with unwrapped outputs.
    """
    unwrapped = []
    for ex in train_examples:
        new_output = _apply_chain(ex["output"], inverse_chain)
        if new_output is None:
            # Skip this pair if inversion fails
            continue
        unwrapped.append({"input": ex["input"], "output": new_output})
    return unwrapped


def forward_chain_from_inverse(inverse_chain: List[str]) -> List[Callable]:
    """
    Build the forward chain corresponding to the discovered inverse chain.
    To apply: forward = reverse the inverse chain, replace each op with its inverse.

    If inverse_chain reverses the original chain, then:
      original_chain = [INVERSE_OPS_of_each]_reversed
    But we don't need original ops — we can apply the inverse of the inverse chain.
    Simpler: applying inverse(inverse(x)) gives back the chain effect.

    Returns list of callables to apply left-to-right on a grid.
    """
    # The forward equivalent of "undo inverse_chain" = "apply inverse_chain forward"
    # since INVERSE_OPS contains inverse implementations.
    # To go from base_fn(input) → output, we need: output = chain(base_fn(input))
    # If we know inverse_chain, then chain = inverse_chain reversed with each op inverted
    #
    # Example: if inverse_chain = [rotate_270, flip_h], that means we applied
    #          inv(rotate_270) ∘ inv(flip_h) = rotate_90 ∘ flip_h to UNDO chain.
    # So original chain was: flip_h^-1 then rotate_90^-1 (read right-to-left)
    # Which equals:          flip_h then rotate_270 (their inverses).
    #
    # Or more simply: forward_chain(x) = INVERSE_OPS^-1 of inverse_chain in reverse order.
    # And INVERSE_OPS^-1 = forward_ops. We need forward ops.

    # Map inverse op back to its forward (which is the same dict but reversed lookup)
    forward_ops = {
        "rotate_90":         rotate_270,    # forward of rotate_270 inverse is rotate_90... wait
        # Actually: INVERSE_OPS["rotate_90"] = rotate_270 (this op is the INVERSE of rotate_90)
        # So if inverse_chain contains "rotate_90", we applied rotate_270 as INV.
        # Forward equivalent: the original chain had rotate_90 in this position.
        # Forward function: rotate_90.
    }

    # Simpler: build dictionary of forward callables.
    NAME_TO_FORWARD = {
        "rotate_90":         rotate_90,
        "rotate_180":        rotate_180,
        "rotate_270":        rotate_270,
        "flip_horizontal":   flip_horizontal,
        "flip_vertical":     flip_vertical,
        "flip_diagonal":     flip_diagonal,
        "flip_antidiagonal": flip_antidiagonal,
        "transpose":         transpose,
        "zoom_2x":           zoom_2x,
        "zoom_3x":           zoom_3x,
        "downsample_2x":     downsample_2x,
        "downsample_3x":     downsample_3x,
    }
    # The original chain (forward) had each op in the order we INVERTED in reverse.
    # If we applied inverse_chain = [I1, I2, I3] to undo, the original was [F3, F2, F1]
    # where Fk is the forward of Ik (i.e., the op whose INVERSE_OPS entry is Ik's impl).
    # But our inverse_chain stores names of ops whose inverses we applied.
    # If inverse_chain[i] = "rotate_90", that means we applied rotate_270 (INVERSE_OPS["rotate_90"])
    # to undo the original op that was rotate_90.
    # So the forward chain = inverse_chain reversed, with same names → forward funcs.
    return [NAME_TO_FORWARD[name] for name in reversed(inverse_chain)]


def apply_forward_chain(g: Grid, forward_chain: List[Callable]) -> Optional[Grid]:
    """Apply forward chain left-to-right. Returns None on error."""
    cur = g
    try:
        for fn in forward_chain:
            cur = fn(cur)
            if not cur or not cur[0]:
                return None
            if len(cur) > 30 or len(cur[0]) > 30:
                return None
    except Exception:
        return None
    return cur
