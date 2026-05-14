# ARC-AGI-2 public dataset integration

Adds the official `arcprize/ARC-AGI-2` repository as a supplementary source
of training pairs for TTT (Test-Time Training).

## Source

- **Repository**: https://github.com/arcprize/ARC-AGI-2
- **License**: same as ARC Prize (see /data/arc_agi2_public/)
- **Size**: 1000 training + 120 evaluation tasks (~8 MB JSON)
- **Format per task**: `{train: [{input, output}, ...], test: [{input, output}, ...]}`
- **Baked into Docker image** at `/app/data/arc_agi2_public/` during build

## Distribution

Heuristic difficulty (combines grid size + color count + train-pair count, 0-10 scale):

```
diff=0:    6 #
diff=1:   79 ###############
diff=2:  126 #########################
diff=3:  165 #################################
diff=4:  118 #######################
diff=5:  117 #######################
diff=6:  137 ###########################
diff=7:  175 ###################################
diff=8:   73 ##############
diff=9:    4
```

Bimodal — clusters at diff=3 (easy puzzles, small grids) and diff=7 (hard puzzles).
Train-pair counts: 2-10, avg 3.2 per task.

## Integration

`arc_public_loader.py`:
```python
from arc_public_loader import load_public_arc
tasks = load_public_arc(n=30, filter_by="hard")  # 30 hardest tasks
# Returns SN5-compatible: {task_id, train_examples, test_input, test_output, _difficulty}
```

`arc_ttt.py` mixes public tasks with SN5 daily tasks before LoRA training:
```python
if PUBLIC_ARC_N > 0:
    public_tasks = load_public_arc(n=PUBLIC_ARC_N, filter_by=PUBLIC_ARC_FILTER)
    sn5_tasks.extend(public_tasks)  # mixed pool → augmented → trained
```

## Env vars

| Var | Default | Effect |
|---|---|---|
| `PUBLIC_ARC_N` | 30 | Number of public tasks to mix in. 0 = disabled. |
| `PUBLIC_ARC_FILTER` | `hard` | `hard` / `easy` / `""` (random shuffle) |
| `PUBLIC_ARC_DIR` | `/app/data/arc_agi2_public` | Override path (for tests) |

## Time budget math

TTT compute is approximately linear in `total_tasks × TTT_REPEAT`.

Baseline: 100 SN5 tasks × TTT_REPEAT=24 = 2400 sequences → ~30-40 min on H200.

Mixing public:

| `PUBLIC_ARC_N` | Total tasks | Sequences | Est. time | Fits 60min? |
|---|---|---|---|---|
| 0 | 100 | 2400 | 32-40 min | ✅ |
| 30 (default) | 130 | 3120 | 42-52 min | ✅ |
| 50 | 150 | 3600 | 48-60 min | ⚠️ tight |
| 100 | 200 | 4800 | 64-80 min | ❌ |
| 200 | 300 | 7200 | 96-120 min | ❌ |

To push higher PUBLIC_ARC_N, also reduce TTT_REPEAT (e.g. 24 → 12) or
training steps. NVARC reference: ~1440 steps for 103K seq, so we can
afford fewer steps with more data.

## Distribution shift risk

The public ARC-AGI-2 dataset has different distribution than SN5's
synthetic puzzle generator (which chains 3-7 base transformations).
Public tasks are hand-crafted by ARC Prize team — wider variety but
not aligned with the chained pattern SN5 validators score on.

**Risk**: too much public mixing could *hurt* SN5-specific accuracy.

**Mitigation**:
- Default `filter_by="hard"` keeps challenging tasks (matches chain≥5)
- Mix ratio 30:100 (public:SN5) keeps SN5 signal dominant
- Watch incentive in TG after first validator run with new image
- If accuracy drops → lower PUBLIC_ARC_N or disable (PUBLIC_ARC_N=0)

## Verification

```bash
# Smoke test loader
python3 arc_public_loader.py  # loads 10 tasks (PUBLIC_ARC_N env), prints stats

# Full test suite (covers loader edge cases)
pytest tests/test_arc_public_loader.py -v
```

## Future work

1. **Distribution alignment** — sample public tasks weighted toward
   SN5-style chain transformations. Need a classifier (manual labeling
   on 100 tasks first).
2. **Multi-source dataset** — add Kaggle 2024 ARC submissions, NVARC's
   synthetic data, etc. Diversify training signal further.
3. **Per-task TTT** — instead of one global LoRA, train per-task adapter
   at inference time (matches NVARC's setup). Requires inference budget
   reshape (~30s/task for TTT).
