# SN5 Monitoring

Two layers of observability.

## Layer 1 — In-container (validator-side) telemetry

`tg_logger.py` emits structured events during prep_phase:

| Event | Level | When |
|---|---|---|
| `prep_start` | START | Beginning of prep, with env snapshot |
| `input_loaded` / `input_missing` | INFO / WARN | After reading `miner_current_dataset.json` |
| `cache_loaded` | INFO | Historical cache loaded |
| `openrouter_start` / `openrouter_batch` / `openrouter_done` | START / INFO / OK | OpenRouter ensemble progress |
| `model_download_start` / `_done` / `_failed` | START / OK / ERROR | HF model fetch |
| `ttt_start` / `ttt_done` / `ttt_failed` / `ttt_disabled` | START / OK / WARN | TTT training |
| `cache_saved` | OK | Final cache written |
| `prep_done` | OK | Successful completion |
| `prep_unhandled_error` | ERROR | Catch-all exception in prep |

Plus a `heartbeat` event every 5 min (silent — no TG ding) for liveness.

**Sinks:**
- Telegram (real-time) — if `TELEGRAM_BOT_TOKEN` + `TELEGRAM_CHAT_ID` available in env
- JSONL buffer (`EVENT_LOG_PATH`, default `/output/prep_events.jsonl`) — always

**Limitation:** validator sandbox typically does NOT pass our private TG creds.
Without them, the buffer file is our only record. If validator clears `/output`
between evals, we get nothing. Open question — see #2 in work plan.

## Layer 2 — Outside-validator monitoring (VPS-side)

`monitor/sn5_monitor.py` runs on our VPS via cron, polls taoswap API and alerts.

### Setup on VPS

```bash
# 1. Copy script
rsync -avz monitor/ vps:/opt/bittensor/monitor/

# 2. Ensure venv has httpx
ssh vps "/opt/bittensor/venv/bin/pip install httpx"

# 3. Cron entries (already added on our VPS — 188.137.244.97)
*/30 * * * * cd /opt/bittensor && set -a && . ./.env && set +a && \
    /opt/bittensor/venv/bin/python3 monitor/sn5_monitor.py >> logs/sn5_monitor.log 2>&1

0 9 * * * cd /opt/bittensor && set -a && . ./.env && set +a && \
    /opt/bittensor/venv/bin/python3 monitor/sn5_monitor.py --daily >> logs/sn5_monitor.log 2>&1
```

### Alerts

- 📈 / 📉 Incentive change > 0.001 (real-time)
- ⏰ No incentive change in 24h+ (warning — validators might not be running us)
- ⚠️ Taoswap API down (silent — to avoid spam)
- 📊 Daily 09:00 UTC report — current state, axon liveness, 24h ticks

### State

Persistent state in `/opt/bittensor/monitor/state.json`:
- `last_incentive` / `last_rank` — for delta detection
- `ticks` — last 100 (~50h) for daily aggregation
- `no_change_since` — for 24h+ silence alert

## TODO

- [ ] Investigate validator env passthrough mechanism (#2 open question)
- [ ] Wire validator hotkey identification into events (#4.5)
- [ ] Read `/output/prep_events.jsonl` on next prep start (if persists)
