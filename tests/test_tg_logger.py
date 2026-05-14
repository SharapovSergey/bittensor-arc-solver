"""Unit tests for tg_logger event log."""
import json
import os
from pathlib import Path

import pytest

import tg_logger


@pytest.fixture(autouse=True)
def _clear_init(tmp_path: Path, monkeypatch):
    """Reset path-init cache and use a clean per-test temp dir for log files."""
    tg_logger._event_log_initialized_paths.clear()
    yield


def test_log_event_writes_to_env_path(tmp_path: Path, monkeypatch):
    log_path = tmp_path / "events.jsonl"
    monkeypatch.setenv("EVENT_LOG_PATH", str(log_path))
    rec = tg_logger.log_event("test_evt", "INFO", {"k": "v"})
    assert rec["event"] == "test_evt"
    assert log_path.exists()
    content = log_path.read_text().strip()
    parsed = json.loads(content)
    assert parsed["event"] == "test_evt"
    assert parsed["data"] == {"k": "v"}


def test_lazy_path_switching(tmp_path: Path, monkeypatch):
    """Different EVENT_LOG_PATH values route to different files within one run."""
    path_a = tmp_path / "a.jsonl"
    path_b = tmp_path / "b.jsonl"

    monkeypatch.setenv("EVENT_LOG_PATH", str(path_a))
    tg_logger.log_event("evt_a", "INFO", {})

    monkeypatch.setenv("EVENT_LOG_PATH", str(path_b))
    tg_logger.log_event("evt_b", "INFO", {})

    assert path_a.exists() and path_b.exists()
    assert "evt_a" in path_a.read_text()
    assert "evt_b" in path_b.read_text()
    assert "evt_b" not in path_a.read_text()
    assert "evt_a" not in path_b.read_text()


def test_log_event_appends_not_overwrites(tmp_path: Path, monkeypatch):
    log_path = tmp_path / "events.jsonl"
    monkeypatch.setenv("EVENT_LOG_PATH", str(log_path))
    tg_logger.log_event("evt1", "INFO", {})
    tg_logger.log_event("evt2", "INFO", {})
    lines = log_path.read_text().strip().split("\n")
    assert len(lines) == 2
    assert json.loads(lines[0])["event"] == "evt1"
    assert json.loads(lines[1])["event"] == "evt2"


def test_log_event_creates_parent_dir(tmp_path: Path, monkeypatch):
    log_path = tmp_path / "deep" / "nested" / "dir" / "events.jsonl"
    monkeypatch.setenv("EVENT_LOG_PATH", str(log_path))
    tg_logger.log_event("evt", "INFO", {})
    assert log_path.exists()


def test_log_event_handles_unwritable_path(tmp_path: Path, monkeypatch, capsys):
    # Path that can't be created (parent is a file, not a directory)
    blocker = tmp_path / "blocker"
    blocker.write_text("x")
    log_path = blocker / "events.jsonl"
    monkeypatch.setenv("EVENT_LOG_PATH", str(log_path))
    # Should NOT raise — just log a warning internally
    rec = tg_logger.log_event("evt", "INFO", {})
    assert rec["event"] == "evt"  # caller still gets back the record


def test_no_tg_creds_returns_false(monkeypatch):
    """tg_send must return False (not raise) when no TG creds set."""
    import asyncio
    monkeypatch.setattr(tg_logger, "_BOT_TOKEN", "")
    monkeypatch.setattr(tg_logger, "_CHAT_ID", "")
    result = asyncio.run(tg_logger.tg_send("hello"))
    assert result is False
