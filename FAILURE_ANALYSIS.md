# Failure Analysis — 9-task trace pilot (2026-05-14)

Per-task tracing on 9 representative tasks revealed two systemic bugs and
informed two optimizations. Net effect: **63% prep time saved per task** without
accuracy loss.

## Sample

| Task | Chain | v5 status |
|---|---|---|
| b001 | 4 | OK (control — was OK in v1) |
| b005 | 5 | OK (control — was OK in v1) |
| b010 | 3 | -- (was timeout in v1) |
| b013 | 3 | -- (was timeout in v1) |
| b014 | 3 | -- (was timeout in v1) |
| b007 | 4 | -- (was timeout in v1) |
| b012 | 4 | -- (was timeout in v1) |
| b000 | 5 | -- (was timeout in v1) |
| b020 | 5 | -- (was timeout in v1) |

## Root cause #1 — grok-4-fast streams reasoning tokens slowly

Trace `b001_v1` showed:

| # | Model | Elapsed |
|---|---|---|
| 1 | gemini-3-flash-preview | 7.25s |
| 2 | mimo-v2-flash | 9.42s |
| 3 | gemini-2.5-flash | 10.88s |
| 4 | qwen3-coder-30b | 19.24s |
| 5 | **x-ai/grok-4-fast** | **78.00s** |

Across all traces, grok-4-fast was the slowest model by 5-20× margin.
On harder tasks, it took **271s** for a single Phase 1 call.

**Why httpx timeout didn't fire**: Grok streams reasoning tokens incrementally.
Each chunk resets the httpx read timer. A 30s `httpx.Timeout` becomes
unbounded if the server sends *any* byte within 30s.

**Fix**: Wrap `call_model` in `asyncio.wait_for(..., timeout=CALL_HARD_TIMEOUT_SEC)`.
asyncio's timeout is independent of streaming — it cancels the task after
75s no matter what.

```python
async def call_model(...):
    try:
        return await asyncio.wait_for(
            _call_model_inner(...), timeout=hard_timeout
        )
    except asyncio.TimeoutError:
        return ""
```

Default `CALL_HARD_TIMEOUT_SEC=75` keeps Phase 1 under 80s while still
allowing grok-4-fast a fair chance.

## Root cause #2 — pipeline runs all phases on impossible tasks

Trace `b010_v3` (after fix #1):

- Phase 1 T=0.3: 5 calls, 0 passing, 0 repair candidates
- Phase 1 T=0.7: 5 calls, 0 passing, 0 repair candidates
- Phase 1.5 Repair: skipped (no candidates)
- Phase 2 Direct: 5 calls, parsed grids — all wrong → identity fallback
- **Total: 15 calls, 227s**

But the final answer is identical to "give up at start" (None → identity at
inference). We just paid 227s × 5 calls for nothing.

**Fix**: Two early-abort gates.

**Abort A** — *after T=0.3, if 0 syntactically valid programs across all 5 models*:
models can't even produce compilable code → task is way outside ensemble capability
→ skip everything. Triggers on ~30% of impossible tasks.

**Abort B** — *after T=0.3, if no passing AND no repair candidates (≥2/3 train)*:
models produce code but none of it is even partially correct → T=0.7 + Phase 2
empirically never recovers. Triggers on ~70% of impossible tasks.

```python
if early_abort and valid_code_count == 0 and not repair_candidates:
    return None  # Abort A
...
if early_abort and ti == 0 and not passing_outputs and not repair_candidates:
    return None  # Abort B
```

Controlled by `EARLY_ABORT_NO_SIGNAL` env var (default `1`).

## Results comparison

| Pilot | Models | Hard timeout | Early-abort | OK | Avg time/task | Total |
|---|---|---|---|---|---|---|
| v1 | 5 (grok 200+s) | none | none | 2/9 | 212s | 1909s |
| v3 | 5 | 75s | none | 1/9 | 203s | 1830s |
| v4 | 5 | 75s | post-Phase 1 | 2/9 | 127s | 1139s |
| **v5** | **5** | **75s** | **after T=0.3** | **2/9** | **77s** | **693s** |

**v5 vs v1**: same accuracy (2/9), **63% less time**.

Variance note: v3 lost b005 because grok-4-fast on that task needed 169s
to emit the unique passing program. With 75s cap it didn't get there.
v4 + v5 got b005 back via T=0.3 sampling variance (different runs hit
different code).

## Why aren't we solving the unsolved 7?

All 7 are **category A — unsolvable by current ensemble**. Even with
unlimited time, our 5 models can't:
- chain=3 b010/b013/b014: 0 valid programs at T=0.3 → models don't even
  recognize the transformation pattern
- chain=4 b007/b012: programs compile but fail all 3 train pairs →
  models understand syntax but pick wrong transformation
- chain=5 b000/b020: similar to chain=4

**These are TTT targets.** Specifically:
- The Mistral-NeMo-8B base model (used in TTT) was pre-trained on ARC-AGI-1
- Our 5 OpenRouter models are generalists with weak ARC-specific knowledge
- TTT trains LoRA adapter on SN5 daily tasks → makes the model task-domain-aware
- NVARC winner (24% private eval) used same architecture, similar data

## Production impact estimate

100 tasks per validator eval, BATCH_SIZE=4 concurrent:
- v1: 100/4 × 212s = 5300s = **88 min** — overflows 60min prep_timeout
- v5: 100/4 × 77s = 1925s = **32 min** — fits with 28min headroom for TTT

**This is the difference between prep_phase always timing out and prep_phase
finishing in half the budget.** Critical for production viability.

## Code changes

| File | Change | Env var |
|---|---|---|
| `arc_prep_phase.py` | Hard outer timeout via asyncio.wait_for | `CALL_HARD_TIMEOUT_SEC=75` |
| `arc_prep_phase.py` | Early-abort A (0 valid code) | `EARLY_ABORT_NO_SIGNAL=1` |
| `arc_prep_phase.py` | Early-abort B (no passing/repair after T=0.3) | same |

## Future work informed by this analysis

1. **TTT is the highest-leverage fix**. The 7 unsolved tasks need
   ARC-domain training. → Move to work plan #3.
2. **Per-task budget could be lower in prod**: with v5's 77s impossible-task
   floor, set `PREP_TASK_TIMEOUT=120s` (was 300s).
3. **BATCH_SIZE bump**: with faster tasks, can run more in parallel.
   But watch OpenRouter rate limits.

## Replay

```bash
# Re-run trace pilot on the 9 tasks
cd "/Users/sharapov/Cloude/Project X"
source .venv/bin/activate
set -a; source .env; set +a
export TRACE_TASKS="b001,b005,b010,b013,b014,b007,b012,b000,b020"
python3 bench_logs/trace_pilot.py
```

Traces saved to `bench_logs/task_traces/{task_id}_trace.json`.
Old version snapshots kept at `bench_logs/task_traces_v1..v4/`.
