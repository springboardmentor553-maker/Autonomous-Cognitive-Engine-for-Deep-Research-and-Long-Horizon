"""
tools/filesystem/read_file.py — Virtual file system read tool.

Retrieves content previously saved with write_file.
"""

from __future__ import annotations

import json

from langchain_core.tools import tool

from utils.logger import get_logger

logger = get_logger(__name__)

# The actual VFS dict is injected at runtime by the graph node via a closure.
# We store it in a module-level variable that build_graph.py populates.
_vfs_ref: dict = {}


def set_vfs_reference(vfs: dict) -> None:
    """Called by the graph to bind the live VFS dict to this tool."""
    global _vfs_ref
    _vfs_ref = vfs


@tool
def read_file(path: str) -> str:
    """
    Read content from a virtual file previously saved with write_file.

    Use this to retrieve research notes, summaries, or any intermediate output
    before synthesising a final response.

    Args:
        path: Virtual path of the file. Example: "/research/ai_trends.md"

    Returns:
        The file content as a string, or an error message if not found.
    """
    if not path.startswith("/"):
        path = "/" + path

    logger.info(f"read_file ← {path}")

    if path not in _vfs_ref:
        available = list(_vfs_ref.keys())
        return json.dumps(
            {
                "error": f"File not found: {path}",
                "available_files": available,
            }
        )

    entry = _vfs_ref[path]
    content = entry["content"] if isinstance(entry, dict) else entry
    return json.dumps({"path": path, "content": content})
