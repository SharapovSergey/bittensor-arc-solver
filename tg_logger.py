"""
Telegram + JSONL event logger for SN5 prep phase.

Two channels:
  1. Telegram (real-time alerts during prep — prep has internet)
  2. JSONL local file (`/output/prep_events.jsonl` — survives in /output)

TG send is fire-and-forget: never blocks pipeline, never crashes on failure.

DEPLOYMENT NOTE on Docker env:
  Validator sandbox does NOT pass TELEGRAM_BOT_TOKEN to our container
  (we don't control validator env). Without creds, tg_send() returns False
  silently → all telemetry lands ONLY in /output/prep_events.jsonl buffer.
  If /output persists between evals, we read it back on next prep (TODO).
  Long-term: use a dedicated low-privilege bot whose token is safe to commit.

Env vars:
  TELEGRAM_BOT_TOKEN  — bot API token (required for TG send; if missing → noop)
  TELEGRAM_CHAT_ID    — destination chat ID (required for TG send)
  TG_PREFIX           — optional prefix for messages (default: "SN5")
  TG_HEARTBEAT_SEC    — heartbeat interval (default 300s = 5min)
  TG_RATE_LIMIT_SEC   — minimum gap between non-heartbeat sends (default 1s)
  EVENT_LOG_PATH      — path for JSONL events (default /output/prep_events.jsonl)
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import time
import traceback
from pathlib import Path
from typing import Any, Optional

import httpx

_logger = logging.getLogger("sn5.tg_logger")

_TG_API = "https://api.telegram.org/bot{token}/sendMessage"
_PREFIX = os.getenv("TG_PREFIX", "SN5")
_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")
_RATE_LIMIT_SEC = float(os.getenv("TG_RATE_LIMIT_SEC", "1.0"))
_HEARTBEAT_SEC = float(os.getenv("TG_HEARTBEAT_SEC", "300"))

_last_send_ts: float = 0.0
_event_log_initialized_paths: set = set()


def _get_event_log_path() -> Path:
    """Read EVENT_LOG_PATH lazily — allows different sinks for prep vs inference."""
    return Path(os.getenv("EVENT_LOG_PATH", "/output/prep_events.jsonl"))


def _emoji_for_level(level: str) -> str:
    return {"INFO": "ℹ️", "WARN": "⚠️", "ERROR": "❌", "OK": "✅", "START": "🟢"}.get(
        level, "•"
    )


def _ensure_event_log_dir(path: Path) -> None:
    """Make sure parent dir exists; idempotent per-path."""
    if str(path) in _event_log_initialized_paths:
        return
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        _event_log_initialized_paths.add(str(path))
    except Exception as e:
        _logger.warning(f"event log dir init failed: {e}")


def log_event(event: str, level: str = "INFO", data: Optional[dict] = None) -> dict:
    """
    Append a structured event to the JSONL log AND format for downstream TG send.

    Returns the event dict (caller can pass to tg_send if desired).
    """
    rec = {
        "ts": time.time(),
        "ts_iso": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "event": event,
        "level": level,
        "data": data or {},
    }
    path = _get_event_log_path()
    _ensure_event_log_dir(path)
    try:
        with path.open("a") as f:
            f.write(json.dumps(rec) + "\n")
    except Exception as e:
        _logger.warning(f"event log write failed: {e}")
    return rec


async def tg_send(
    msg: str,
    level: str = "INFO",
    silent: bool = False,
    timeout: float = 5.0,
) -> bool:
    """
    Fire-and-forget Telegram send.

    Returns True on confirmed success (HTTP 200 + ok=True), False otherwise.
    Never raises — always swallows exceptions and logs warning.

    `silent=True` skips Telegram "ding"; useful for heartbeats.
    """
    global _last_send_ts

    if not _BOT_TOKEN or not _CHAT_ID:
        return False

    # Rate limit (heartbeats/etc not flooded)
    now = time.time()
    if now - _last_send_ts < _RATE_LIMIT_SEC:
        # too soon — skip but don't fail
        return False
    _last_send_ts = now

    text = f"[{_PREFIX}] {_emoji_for_level(level)} {msg}"
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            r = await client.post(
                _TG_API.format(token=_BOT_TOKEN),
                json={
                    "chat_id": _CHAT_ID,
                    "text": text[:4000],  # TG limit 4096
                    "disable_notification": silent,
                },
            )
            data = r.json()
            return bool(data.get("ok"))
    except Exception as e:
        _logger.warning(f"tg_send failed: {e}")
        return False


async def emit(
    event: str,
    msg: str,
    level: str = "INFO",
    data: Optional[dict] = None,
    silent: bool = False,
) -> None:
    """
    Combined: log JSONL event + send TG message.

    Use this as the primary milestone API throughout prep_phase.
    """
    log_event(event, level, data)
    await tg_send(msg, level=level, silent=silent)


# ── Heartbeat task ────────────────────────────────────────────────────────────


class Heartbeat:
    """
    Background heartbeat task. Use as context manager:

        async with Heartbeat("prep_phase"):
            await do_prep()
    """

    def __init__(self, phase: str, interval: float = _HEARTBEAT_SEC):
        self.phase = phase
        self.interval = interval
        self.start_ts = 0.0
        self._task: Optional[asyncio.Task] = None

    async def __aenter__(self) -> "Heartbeat":
        self.start_ts = time.time()
        self._task = asyncio.create_task(self._loop())
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        if self._task and not self._task.done():
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass

    async def _loop(self) -> None:
        try:
            while True:
                await asyncio.sleep(self.interval)
                elapsed = int(time.time() - self.start_ts)
                await tg_send(
                    f"heartbeat: {self.phase} running {elapsed}s",
                    level="INFO",
                    silent=True,
                )
        except asyncio.CancelledError:
            return
        except Exception as e:
            _logger.warning(f"heartbeat loop crashed: {e}\n{traceback.format_exc()}")


# ── Error helpers ─────────────────────────────────────────────────────────────


async def report_exception(event: str, exc: BaseException, context: Optional[dict] = None) -> None:
    """Format exception traceback and send to TG + log."""
    tb = "".join(traceback.format_exception_only(type(exc), exc)).strip()
    data = {"exception": tb, "context": context or {}}
    log_event(event, "ERROR", data)
    msg = f"{event}: {tb[:300]}"
    if context:
        msg += f"\nctx: {json.dumps(context)[:200]}"
    await tg_send(msg, level="ERROR")


# ── Smoke test (manual) ───────────────────────────────────────────────────────

if __name__ == "__main__":
    async def _smoke() -> None:
        await emit("smoke_test", "TG logger module smoke-test from main", "START")
        await emit("smoke_test_done", "done", "OK")

    asyncio.run(_smoke())
