# SN5 Validator Investigation

State as of 2026-05-14 (block 8180580).

## Hyperparameters (Bittensor SDK + taoswap)

| Param | Value | Means |
|---|---|---|
| `tempo` | 360 blocks | ~72 min between weight updates |
| `immunity_period` | 5000 blocks | ~16.6 hours of dereg protection |
| `min_burn` | 500000 rao = 0.0005 TAO | ≈ $0.16 register cost |
| `serving_rate_limit` | 50 | Max RPS from validator |
| `max_validators` | 64 | Validator slot cap |

## Validator roster (15 active on SN5)

Filter: `validator_permit=True`.

| UID | Stake (TAO) | Axon | Notes |
|---|---|---|---|
| 49 | 799,893 | 0.0.0.0:0 (hidden) | Largest validator |
| 6 | 622,110 | 195.189.97.59:8091 | |
| 241 | 269,408 | 0.0.0.0:0 (hidden) | |
| 10 | 198,004 | 34.130.191.26:8091 | |
| 1 | 158,690 | 0.0.0.0:0 (hidden) | |
| 81 | 141,476 | 31.12.82.146:8091 | |
| 210 | 133,677 | 0.0.0.0:0 (hidden) | |
| 176 | 98,874 | 0.0.0.0:0 (hidden) | |
| 54 | 97,082 | 138.2.228.26:8091 | |
| 28 | 42,233 | **192.150.253.122:16485** | **Actively polling us** |

Hidden-axon validators (0.0.0.0:0) — pull from public miners but don't expose
their own axon. Legitimate Bittensor pattern.

## Our miner state (UID 65)

| Field | Value | Status |
|---|---|---|
| Hotkey | `5EFAUW...CGc4x` | ✓ |
| Axon | **188.137.244.97:8091** | ✓ Serving |
| Stake | 0.0 TAO | OK (miner not validator) |
| **Incentive** | **0.0** | ❌ |
| **Rank** | None | ❌ |
| Last update | block 17506 ago | ~58h |
| Process | sn5_miner_server.py PID 148356 | ✓ running |
| Port 8091 | LISTENING | ✓ |

**Important**: taoswap API previously suggested `axon: None` — this was misleading.
The Bittensor SDK confirms axon IS published on-chain. The misleading API was a
red herring; our `/info` endpoint is fully reachable.

## Who's actually polling us

From `/opt/bittensor/logs/sn5_miner.log`, last ~3 days:

| Source IP | Hits | Matched UID |
|---|---|---|
| **167.150.153.5** | 437 | Not in current metagraph (possible validator without published axon, or stale registration) |
| **192.150.253.122** | 299 | **UID 28 (stake 42K, validator_permit=True)** |
| 127.0.0.1 | 14 | local health checks |
| 5.252.179.55 | 3 | possibly probe |
| 188.137.244.97 | 1 | self |

So at least **2 distinct validators** poll us — UID 28 confirmed, and another
probably-validator at 167.150.153.5 (~150 hits/day).

## Weight matrix analysis

Queried `mg.W` (validator × miner matrix). **0 validators set nonzero weight on our UID 65.**

That means: validators are running their full eval pipeline against us
(prep_phase → inference_phase → score), but **the score is 0 every time**.

Three hypotheses for why scoring = 0:

### H1 (most likely): prep_phase fails silently in validator sandbox
- Validator clones our repo, builds Docker, runs prep
- Something fails (missing module? import error? OpenRouter key not passed?)
- `cache.json` empty or only-None values
- inference_phase falls through to identity for all 100 tasks
- 0/100 correct → score=0

This is exactly the blackbox problem that the new logging (#2, commit
2b9a538) is designed to expose. After validator clones the updated repo,
we should see either:
- `prep_events.jsonl` in `/output` (if filesystem persists), OR
- TG messages (if creds passthrough works), OR
- silent failure (if neither — need different mechanism)

### H2: prep succeeds but inference gives wrong answers
- prep solves N/100 tasks via OpenRouter
- vLLM serves merged TTT model
- TTT-trained model + cache produce predictions
- predictions don't match ARC-AGI-2 ground truth → score=0

Possible if our OpenRouter ensemble accuracy is also ~0% on ARC-AGI-2
(consistent with bench v5 = 10%). 10% means 90 wrong + 10 right; threshold
for any incentive is 20% per CLAUDE.md. We're below threshold even on best path.

### H3: Threshold issue
- We DO get scored, just below the 20% threshold
- Result: zero incentive but nonzero raw score
- Weight matrix should still show ~0 weight (validators filter out below threshold)

Cannot distinguish H1 vs H2 without telemetry. **#2 logging deployment is the unlock.**

## Tempo and expected cadence

`tempo=360` means weights are set every ~72 minutes. So a validator typically:
- Runs prep for ~60 min (with internet)
- Runs inference for ~60 min (no internet)
- Reports weights at end of each tempo

For 24h period: up to **~20 evals per validator**. With 15 validators:
**up to 300 evals/day** in theory. Real number depends on how many run the
full eval (vs skip) and validator schedule.

Our log shows 437+299 = ~750 hits in 3 days = **~250/day**. Consistent
with theoretical max — validators ARE running the full cycle on us.

## Actions

| # | Action | Status |
|---|---|---|
| A | Verify our axon published | ✅ Done (SDK shows serving) |
| B | Identify which validators poll us | ✅ Done (UID 28 + 167.150.153.5) |
| C | Determine why score=0 | ⏳ Awaits #2 logging in next validator run |
| D | Resolve H1 vs H2 | ⏳ Depends on C — if prep fails (H1) we'd see TG/JSONL signals; if prep succeeds (H2) we'd see "openrouter_done OK" but still no incentive |
| E | Cron-poll w/sn5_monitor.py | ✅ Deployed, 30min ticks + daily 09:00 |

## Next investigative steps (post-this-document)

### ✅ BREAKTHROUGH: TG creds passthrough via `custom_env_vars`

The /info endpoint already exposes `custom_env_vars` which validators honor
(they use it to receive OPENROUTER_API_KEY). We just extended this to also
pass `TELEGRAM_BOT_TOKEN` and `TELEGRAM_CHAT_ID`.

- sn5_miner_server.py v1.2.0 deployed to VPS (PID 158624)
- Verified `curl http://188.137.244.97:8091/info` returns TG creds
- **Next validator clone will pick up the new repo (commit afe2409+) AND
  the new /info contract → TG events fire from inside sandbox**

This unlocks H1 vs H2 resolution: we'll see exactly where prep/inference
fails (or succeeds).

### Remaining steps

1. **Wait for next validator run on da061f8 + sn5_miner_server v1.2.0**
   (frequency ~5/hour from logs):
   - Expect TG messages: `prep_start`, `cache_loaded`, `openrouter_start`,
     `openrouter_done`, `ttt_start/done`, `prep_done` (or errors)
   - If TG still silent → check for any sandbox HTTPS restrictions

2. **Read `/opt/bittensor/logs/sn5_miner.log`** for any 4xx/5xx HTTP responses
   to validators. Currently all 200 OK — good, no infrastructure bug.

3. **Watch incentive monitor** — `monitor/sn5_monitor.py` runs every 30 min,
   first signal of "validator scored us > 0" comes here.
