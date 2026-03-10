"""
tools/filesystem/ls.py — List virtual file system contents.
"""

from __future__ import annotations

import json

from langchain_core.tools import tool

from utils.logger import get_logger

logger = get_logger(__name__)

_vfs_ref: dict = {}


def set_vfs_reference(vfs: dict) -> None:
    global _vfs_ref
    _vfs_ref = vfs


@tool
def ls(directory: str = "/") -> str:
    """
    List files in the virtual file system.

    Args:
        directory: Directory prefix to filter by. Defaults to "/" (list all).

    Returns:
        JSON list of file paths and their sizes.
    """
    prefix = directory if directory.startswith("/") else "/" + directory
    if not prefix.endswith("/") and prefix != "/":
        prefix = prefix + "/"

    matches = []
    for path, entry in _vfs_ref.items():
        if prefix == "/" or path.startswith(prefix):
            content = entry["content"] if isinstance(entry, dict) else entry
            matches.append(
                {
                    "path": path,
                    "size_chars": len(content),
                    "updated_at": entry.get("updated_at", "unknown") if isinstance(entry, dict) else "unknown",
                }
            )

    logger.info(f"ls '{prefix}' → {len(matches)} files")
    return json.dumps({"directory": prefix, "files": matches, "count": len(matches)})
