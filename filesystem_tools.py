"""
filesystem_tools.py - Virtual File System Tools for Deep Cognitive Task Framework
Milestone 2: Context Offloading via Virtual File System

These tools let the agent interact with a dict inside AgentState["virtual_files"].
This acts as a persistent scratchpad / short-term memory across all reasoning steps,
solving the LLM context window limitation for long-horizon tasks.

Tools:
    ls          — list all files currently in the virtual file system
    read_file   — read the full content of a file
    write_file  — create or overwrite a file
    edit_file   — find-and-replace text inside an existing file
    delete_file — remove a file from the virtual file system
"""

import json
from langchain_core.tools import tool

# ─────────────────────────────────────────────────────────────────────────────
# Shared in-memory store
# The agent's tool_node_wrapper keeps this in sync with AgentState.virtual_files
# ─────────────────────────────────────────────────────────────────────────────

# This module-level dict acts as the runtime backing store.
# It is synced TO and FROM AgentState.virtual_files by the orchestrator in main.py.
_VIRTUAL_FS: dict[str, str] = {}


def get_virtual_fs() -> dict[str, str]:
    """Return the current virtual file system state (called by orchestrator)."""
    return dict(_VIRTUAL_FS)


def set_virtual_fs(files: dict[str, str]) -> None:
    """Overwrite the virtual file system with new state (called by orchestrator)."""
    global _VIRTUAL_FS
    _VIRTUAL_FS.clear()
    _VIRTUAL_FS.update(files)


# ─────────────────────────────────────────────────────────────────────────────
# Tool Implementations
# ─────────────────────────────────────────────────────────────────────────────

@tool
def ls(directory: str = "/") -> str:
    """
    List all files currently stored in the virtual file system.

    Use this to see what notes, summaries, or drafts you have already saved.

    Args:
        directory: Ignored (all files are in one flat namespace). Pass "/" or "".

    Returns:
        JSON with a list of filenames and their sizes (character count).

    Example:
        ls("/")  →  {"files": [{"name": "research_notes.txt", "size": 342}, ...]}
    """
    if not _VIRTUAL_FS:
        return json.dumps({"files": [], "message": "Virtual file system is empty."})

    file_list = [
        {"name": fname, "size": len(content), "lines": content.count("\n") + 1}
        for fname, content in _VIRTUAL_FS.items()
    ]
    return json.dumps({
        "files": file_list,
        "total_files": len(file_list),
        "message": f"{len(file_list)} file(s) in virtual file system."
    }, indent=2)


@tool
def read_file(filename: str) -> str:
    """
    Read the full content of a file from the virtual file system.

    Use this to retrieve previously saved research notes, summaries, or drafts
    before synthesizing or writing the final output.

    Args:
        filename: The name of the file to read (e.g., "research_notes.txt").

    Returns:
        JSON with the file content, or an error if the file does not exist.

    Example:
        read_file("research_notes.txt")
    """
    if filename not in _VIRTUAL_FS:
        available = list(_VIRTUAL_FS.keys())
        return json.dumps({
            "success": False,
            "error": f"File '{filename}' not found.",
            "available_files": available
        })

    content = _VIRTUAL_FS[filename]
    return json.dumps({
        "success": True,
        "filename": filename,
        "content": content,
        "size": len(content),
        "lines": content.count("\n") + 1
    })


@tool
def write_file(filename: str, content: str) -> str:
    """
    Write (create or overwrite) a file in the virtual file system.

    Use this to save intermediate results such as:
    - Research notes after gathering information
    - Summaries of individual articles or sources
    - Draft sections before combining them
    - Logs or observations from tool calls

    This is the primary tool for offloading context — saving information here
    keeps it out of the LLM context window while making it retrievable later.

    Args:
        filename: The name of the file (e.g., "research_notes.txt", "draft_v1.md").
        content:  The full text content to write to the file.

    Returns:
        JSON confirming the write operation with file size.

    Example:
        write_file("notes.txt", "Key finding 1: ...\nKey finding 2: ...")
    """
    _VIRTUAL_FS[filename] = content
    return json.dumps({
        "success": True,
        "message": f"File '{filename}' written successfully.",
        "filename": filename,
        "size": len(content),
        "lines": content.count("\n") + 1
    })


@tool
def edit_file(filename: str, old_text: str, new_text: str) -> str:
    """
    Edit an existing file by replacing a specific piece of text with new text.

    Use this when you want to update or append to an existing file without
    rewriting the entire content. Ideal for incremental note-taking or
    correcting/expanding a draft.

    Args:
        filename: The name of the file to edit.
        old_text: The exact text to find and replace (must exist in the file).
        new_text: The replacement text to insert in place of old_text.

    Returns:
        JSON confirming the edit, or an error if the file/text was not found.

    Example:
        edit_file("notes.txt", "Key finding 1: TBD", "Key finding 1: LLMs improve productivity by 30%")
    """
    if filename not in _VIRTUAL_FS:
        return json.dumps({
            "success": False,
            "error": f"File '{filename}' not found. Use write_file to create it first.",
            "available_files": list(_VIRTUAL_FS.keys())
        })

    current = _VIRTUAL_FS[filename]
    if old_text not in current:
        return json.dumps({
            "success": False,
            "error": f"Text to replace was not found in '{filename}'.",
            "hint": "Use read_file first to see the exact current content."
        })

    updated = current.replace(old_text, new_text, 1)
    _VIRTUAL_FS[filename] = updated
    return json.dumps({
        "success": True,
        "message": f"File '{filename}' edited successfully.",
        "filename": filename,
        "old_size": len(current),
        "new_size": len(updated)
    })


@tool
def delete_file(filename: str) -> str:
    """
    Delete a file from the virtual file system.

    Use this to clean up temporary notes or drafts that are no longer needed.

    Args:
        filename: The name of the file to delete.

    Returns:
        JSON confirming deletion or an error if the file was not found.
    """
    if filename not in _VIRTUAL_FS:
        return json.dumps({
            "success": False,
            "error": f"File '{filename}' not found.",
            "available_files": list(_VIRTUAL_FS.keys())
        })

    del _VIRTUAL_FS[filename]
    return json.dumps({
        "success": True,
        "message": f"File '{filename}' deleted successfully."
    })


# ─────────────────────────────────────────────────────────────────────────────
# Export
# ─────────────────────────────────────────────────────────────────────────────

FILESYSTEM_TOOLS = [ls, read_file, write_file, edit_file, delete_file]