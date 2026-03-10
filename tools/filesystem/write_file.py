

"""
tools/filesystem/write_file.py — Virtual file system write tool.

Saves content to an in-memory path inside the agent's state.
The actual state update happens in the graph node that processes tool results.
This tool returns a structured JSON payload that the node uses to mutate state.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone

from langchain_core.tools import tool

from utils.logger import get_logger

logger = get_logger(__name__)


@tool
def write_file(path: str, content: str) -> str:
    """
    Write content to a virtual file in the agent's memory.

    Use this to persist research findings, summaries, code drafts, or any
    intermediate output so it can be retrieved later without consuming context.

    Naming conventions (recommended):
        /research/<topic>.md   — research notes
        /summaries/<name>.md   — article/document summaries
        /drafts/<name>.txt     — work-in-progress text
        /data/<name>.json      — structured data

    Args:
        path:    Virtual path starting with "/".  Example: "/research/ai_trends.md"
        content: Full text content to store.

    Returns:
        JSON confirmation that the graph node will use to update state.
    """
    if not path.startswith("/"):
        path = "/" + path

    now = datetime.now(timezone.utc).isoformat()
    logger.info(f"write_file → {path} ({len(content)} chars)")

    return json.dumps(
        {
            "action": "write_file",
            "path": path,
            "content": content,
            "created_at": now,
            "updated_at": now,
        }
    )
