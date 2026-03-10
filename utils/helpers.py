

"""
utils/helpers.py — Miscellaneous helper functions.
"""

from __future__ import annotations

from datetime import datetime, timezone


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def next_pending_todo(todos: list[dict]) -> dict | None:
    """Return the first TODO with status 'pending', or None if all done."""
    return next((t for t in todos if t["status"] == "pending"), None)


def all_todos_done(todos: list[dict]) -> bool:
    return all(t["status"] in ("completed", "failed") for t in todos) and bool(todos)


def mark_todo(todos: list[dict], task_id: str, status: str, result: str = "") -> list[dict]:
    """Return a new todos list with the given task updated."""
    updated = []
    for t in todos:
        if t["id"] == task_id:
            t = {**t, "status": status, "result": result}
        updated.append(t)
    return updated
