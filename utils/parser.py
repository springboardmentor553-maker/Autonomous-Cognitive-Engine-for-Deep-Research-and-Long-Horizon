

"""
utils/parser.py — Output parsing helpers.
"""

from __future__ import annotations

import json
import re
from typing import Any


def safe_json_loads(text: str) -> dict[str, Any] | list | None:
    """Try to parse JSON; strip markdown fences if present."""
    text = text.strip()
    # Remove ```json ... ``` fences
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.MULTILINE)
    text = re.sub(r"\s*```$", "", text, flags=re.MULTILINE)
    try:
        return json.loads(text)
    except (json.JSONDecodeError, ValueError):
        return None


def extract_tool_calls(message) -> list[dict[str, Any]]:
    """Return list of tool-call dicts from an AIMessage."""
    if not hasattr(message, "tool_calls"):
        return []
    return message.tool_calls or []


def todos_to_markdown(todos: list[dict]) -> str:
    """Pretty-print the TODO list as a markdown checklist."""
    lines = ["## Task Plan\n"]
    for t in todos:
        icon = {"pending": "⬜", "in_progress": "🔄", "completed": "✅", "failed": "❌"}.get(
            t.get("status", "pending"), "⬜"
        )
        lines.append(f"{icon} **{t['id']}**: {t['description']}")
        if t.get("result"):
            lines.append(f"   > {t['result'][:120]}…" if len(t["result"]) > 120 else f"   > {t['result']}")
    return "\n".join(lines)
