"""
prompt_traces.py — Lightweight prompt trace storage.

Stores PromptTrace records for debugging and prompt optimization.
Keyed by message_id, max 500 entries (FIFO eviction).
Lives alongside other JSON data files in data/prompt_traces.json.
"""

from __future__ import annotations

import json
import logging
import os
import time
from typing import Optional

from prompt_tree import PromptTrace

logger = logging.getLogger(__name__)

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
TRACES_FILE = os.path.join(DATA_DIR, "prompt_traces.json")
MAX_TRACES = 500


def _ensure_dir() -> None:
    os.makedirs(DATA_DIR, exist_ok=True)


def _read_all() -> dict[str, dict]:
    """Read all traces from disk. Returns dict keyed by message_id.

    Legacy/corrupt data is tolerated: a non-dict top level degrades to an
    empty store, and non-dict trace entries are dropped, so callers never
    crash on .keys()/.values()/.get() downstream.
    """
    _ensure_dir()
    if not os.path.exists(TRACES_FILE):
        return {}
    try:
        with open(TRACES_FILE, "r") as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        logger.warning("Failed to read prompt traces: %s", e)
        return {}
    if not isinstance(data, dict):
        logger.warning("prompt_traces.json top level is not a dict, ignoring.")
        return {}
    return {k: v for k, v in data.items() if isinstance(v, dict)}


def _write_all(traces: dict[str, dict]) -> None:
    """Write all traces to disk atomically."""
    _ensure_dir()
    tmp_path = TRACES_FILE + ".tmp"
    try:
        with open(tmp_path, "w") as f:
            json.dump(traces, f, ensure_ascii=False, indent=2)
        os.replace(tmp_path, TRACES_FILE)
    except OSError as e:
        logger.error("Failed to write prompt traces: %s", e)
        try:
            os.remove(tmp_path)
        except OSError:
            pass


def save_trace(trace: PromptTrace) -> None:
    """Save a single prompt trace. FIFO eviction when over MAX_TRACES."""
    traces = _read_all()

    # FIFO eviction: remove oldest entries when over limit
    if len(traces) >= MAX_TRACES:
        # Sort by timestamp, keep newest (MAX_TRACES - 1)
        sorted_ids = sorted(
            traces.keys(),
            key=lambda mid: traces[mid].get("timestamp", ""),
        )
        to_remove = sorted_ids[: len(sorted_ids) - MAX_TRACES + 1]
        for mid in to_remove:
            del traces[mid]

    traces[trace.message_id] = trace.to_dict()
    _write_all(traces)
    logger.debug("Saved prompt trace for message %s (route: %s)", trace.message_id, " → ".join(trace.route))


def get_trace(message_id: str) -> Optional[dict]:
    """Get a prompt trace by message ID."""
    traces = _read_all()
    trace = traces.get(message_id)
    return trace if isinstance(trace, dict) else None


def get_recent_traces(limit: int = 20) -> list[dict]:
    """Get the most recent N prompt traces."""
    traces = _read_all()
    sorted_traces = sorted(
        traces.values(),
        key=lambda t: t.get("timestamp", ""),
        reverse=True,
    )
    return sorted_traces[:limit]


def get_trace_count() -> int:
    """Get total number of stored traces."""
    return len(_read_all())


def clear_traces() -> None:
    """Clear all stored traces."""
    _ensure_dir()
    try:
        with open(TRACES_FILE, "w") as f:
            json.dump({}, f)
        logger.info("Prompt traces cleared")
    except OSError as e:
        logger.error("Failed to clear prompt traces: %s", e)


def reconstruct_prompt(message_id: str) -> Optional[str]:
    """Reconstruct the full prompt for a given message from its trace."""
    trace_data = get_trace(message_id)
    if not trace_data:
        return None

    from prompt_tree import render_chain
    return render_chain(
        trace_data.get("route", []),
        **trace_data.get("variables", {}),
    )
