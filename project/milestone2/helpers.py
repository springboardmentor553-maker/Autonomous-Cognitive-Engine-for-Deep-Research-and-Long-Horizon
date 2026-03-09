"""
Shared helper utilities for Milestone 2.
"""

import json
import ast
from typing import Any, Dict, List


def safe_parse_todos(content: Any) -> List[Dict]:
    """
    Safely parse write_todos tool output into a list of todo dicts.

    Handles: string representation of list, JSON string, or raw list.
    """
    if isinstance(content, list):
        return content

    if isinstance(content, str):
        content = content.strip()
        # Try JSON first
        try:
            parsed = json.loads(content)
            if isinstance(parsed, list):
                return parsed
        except (json.JSONDecodeError, ValueError):
            pass

        # Try ast.literal_eval for Python list syntax
        if content.startswith("["):
            try:
                parsed = ast.literal_eval(content)
                if isinstance(parsed, list):
                    return parsed
            except (ValueError, SyntaxError):
                pass

    return []


def extract_tool_names(messages: list) -> List[str]:
    """Return ordered list of tool names called across all messages."""
    seen = []
    for msg in messages:
        if hasattr(msg, "tool_calls") and msg.tool_calls:
            for tc in msg.tool_calls:
                seen.append(tc["name"])
        if hasattr(msg, "name") and msg.name and msg.name not in seen:
            seen.append(msg.name)
    return seen


def format_vfs_snapshot(files: Dict[str, str]) -> str:
    """Return a formatted summary of VFS contents."""
    if not files:
        return "(empty)"
    lines = []
    for fname, content in sorted(files.items()):
        preview = content[:100].replace("\n", " ")
        ellipsis = "…" if len(content) > 100 else ""
        lines.append(f"  • {fname} ({len(content)} chars): {preview}{ellipsis}")
    return "\n".join(lines)
