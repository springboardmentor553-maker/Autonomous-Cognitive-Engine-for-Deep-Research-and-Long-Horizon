"""
tools/filesystem/edit_file.py — Virtual file system edit (patch) tool.

Allows the agent to append to or replace sections of an existing virtual file
without rewriting the entire content.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone

from langchain_core.tools import tool

from utils.logger import get_logger

logger = get_logger(__name__)


@tool
def edit_file(path: str, mode: str, content: str, old_text: str = "") -> str:
    """
    Edit an existing virtual file.

    Modes:
        "append"  — Add content to the end of the file.
        "replace" — Replace the first occurrence of old_text with content.
        "overwrite" — Replace the entire file content (same as write_file).

    Args:
        path:     Virtual path of the file to edit.
        mode:     One of "append", "replace", "overwrite".
        content:  New content to add or substitute.
        old_text: (Only for "replace" mode) The exact text to find and replace.

    Returns:
        JSON payload that the graph node will use to update state.
    """
    if not path.startswith("/"):
        path = "/" + path

    valid_modes = {"append", "replace", "overwrite"}
    if mode not in valid_modes:
        return json.dumps({"error": f"Invalid mode '{mode}'. Use one of: {valid_modes}"})

    logger.info(f"edit_file [{mode}] → {path}")

    now = datetime.now(timezone.utc).isoformat()

    return json.dumps(
        {
            "action": "edit_file",
            "path": path,
            "mode": mode,
            "content": content,
            "old_text": old_text,
            "updated_at": now,
        }
    )
