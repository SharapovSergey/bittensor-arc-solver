# Inference Phase — Decision Flow

How `inference_phase.py` produces a prediction for each task.

## Entry point

```
arc_main.py --phase inference --input <dir> --output <dir>
  → arc_inference_phase.run_inference_phase()
    → for task in tasks:
        predict(task) — see below
```

## Per-task decision tree

```
┌──────────────────────────────────────────────────────────────────┐
│  task: { task_hash, train_examples, test_input }                  │
└────────────────────────────────┬─────────────────────────────────┘
                                 ↓
                  ┌──────────────────────────────┐
                  │  STEP 1: Cache lookup        │
                  │  cache[task_hash] valid?     │
                  └──┬───────────────────────┬───┘
              YES   │                       │  NO / invalid
                    ↓                       ↓
            ┌───────────────┐    ┌──────────────────────────┐
            │ return cache  │    │  STEP 2: solver.solve()  │
            │ source=cache  │    └──────────────┬───────────┘
            └───────────────┘                   ↓
                                    ┌────────────────────────────┐
                                    │ Phase 0: Chain inversion   │
                                    │ (strict mode, ~1s, 0 LLM)  │
                                    │ if found → return          │
                                    └─────────┬──────────────────┘
                                              ↓ not found
                                    ┌────────────────────────────┐
                                    │ Phase 1: Program synthesis │
                                    │ K=3 at T=0.3, 0.6, 0.9     │
                                    │ Sequential, early stop on  │
                                    │ first passing program.     │
                                    │ Repair loop (1 cand×2 try) │
                                    │ if no full-pass yet.       │
                                    │ Majority vote across       │
                                    │ passing outputs.           │
                                    └─────────┬──────────────────┘
                                              ↓ no passing
                                    ┌────────────────────────────┐
                                    │ Phase 2: Direct LLM        │
                                    │ Single call, grid output   │
                                    │ (if budget > 15s)          │
                                    └─────────┬──────────────────┘
                                              ↓ failed
                                    ┌────────────────────────────┐
                                    │ Phase 3: Identity fallback │
                                    │ return test_input copy     │
                                    │ source=identity            │
                                    └────────────────────────────┘
```

## Key facts

- **vLLM serves the MERGED TTT model.** `arc_solver_llm.py:60` —
  `vllm_model = models.data[0].id`, whatever vLLM has loaded. The merged
  model (base + TTT LoRA) is produced by `arc_ttt.py` during prep phase
  and saved as `mistral-ttt-merged`. There is no "TTT-vs-base" branching
  on inference side — it's TTT-merged ALWAYS (or base if TTT failed and
  was copied as fallback).

- **Cache wins over vLLM.** If prep phase solved this task_hash via
  OpenRouter ensemble, we use that cached answer. `_is_valid` checks
  shape and value ranges before accepting cache.

- **Chain inversion only at strict mode** (`require_identity_match=True`).
  Per CLAUDE.md "0 hits on ARC-AGI-2 generator" — generator always wraps
  with non-identity base_fn. Loose mode not yet implemented.

- **K=3, not K=20.** Reduced from NVARC's K=20 to fit time budget.
  Diversity via temperature spread (0.3, 0.6, 0.9). Early-stop on first
  passing program — so often only 1-2 LLM calls happen on easy tasks.

- **Per-task budget is dynamic.** `remaining_time / remaining_tasks` so
  early easy cached tasks free up time for later hard ones.

- **No BFS solver.** Comment in `arc_inference_phase.py:11` —
  "BFS removed: benchmark showed 1.1% real accuracy (wrong architecture)".

- **Identity fallback ALWAYS returns something.** Even if vLLM is down,
  we return `test_input` unchanged. Never 0 predictions.

## Output metadata

Each prediction carries `metadata.source ∈ {cache, vllm, identity}`. This
lets us post-mortem the distribution: how many tasks each path actually
served. Should be logged via the JSONL buffer (#2.5 — TODO wire into
inference_phase).

## Time budget

Hard cap: `INFERENCE_TIME_BUDGET=3420s` (validator gives 3600s, we reserve
~5% for startup/teardown).

Per-task allocation: `remaining_time / remaining_tasks`. So:
- Cached task: ~0s (instant) → frees budget for others
- Synthesis success: 30-90s typical
- Direct fallback: 15-30s
- Identity: instant

If pipeline has 100 tasks and 50 are cached, 50 uncached get ~68s each.

## Gap analysis (where decision logic could be better)

### Gap A — No cache validation vs vLLM cross-check
If cache says A and vLLM says B, we use A. No way to detect
prep-phase mistakes. Future: optionally re-validate cached answer
on vLLM for high-stakes tasks (chain ≥ 6).

### Gap B — Chain inversion loose mode not wired
Loose mode (shape-match heuristic) is implemented but not enabled
because we never built "unwrapped synthesis" (LLM gets unwrapped
train → synthesizes base_fn → applies forward chain). High potential
but ~2-3 day risky implementation. See CLAUDE.md.

### Gap C — Single-shot per temperature
Phase 1 calls vLLM once per temp (T=0.3, 0.6, 0.9). Could batch
n=2-3 per call for free diversity. NVARC's K=20 used n=5 batches.

### Gap D — Repair only 1 candidate, 2 attempts
On hard tasks (chain ≥ 5) where Phase 1 yields multiple 2/3-passing
programs, we only repair the first. Hidden value in repairing others.
Time budget is real constraint though.

### Gap E — No adaptive K based on task difficulty
Easy chain=3 tasks get same K=3 as hard chain=7. Could spend less
on easy ones, more on hard.

### Gap F — Multi-adapter not explored
Single TTT-merged model serves all tasks. Hypothesis from work plan
#5.7: train per-task-type LoRAs (spatial, color, object ops). Would
require training infrastructure overhaul.

## Improvement hypotheses (ranked by ROI estimate)

| # | Hypothesis | Effort | Expected lift | Risk |
|---|---|---|---|---|
| 1 | **Batch n=3 per temperature** (Gap C) — more diversity for free | 1h | +1-3pp | Low |
| 2 | **Inference telemetry to JSONL** (Gap A foundation) — wire into 2.5 buffer | 2h | 0 directly, enables all | Zero |
| 3 | **Loose chain inversion** with unwrapped synthesis (Gap B) | 2-3 days | +3-8pp on chain≥5 | High — regression risk if guess wrong |
| 4 | **Adaptive K by chain length** (Gap E) — K=2 for chain≤3, K=5 for chain≥6 | 3h | +1-3pp on chain≥5 | Low |
| 5 | **Multi-adapter** (Gap F) — train 3 LoRAs by task type, vote | 1 week | +5-10pp | High — infra rewrite |
| 6 | **Cache cross-check** (Gap A) — vLLM revalidates cached chain≥6 | 4h | -1 to +2pp | Medium — costs budget |

## Recommendation

Cheapest first: implement #1 and #2 (~3h total).
Then evaluate impact via #2 (telemetry shows where time goes).
Defer #3-6 until after failure analysis (#1 of work plan).
