"""
Virtual File System Tools - Milestone 2

Implements a simple in-memory virtual file system via a shared state dict.
Tools:
  - write_file  : store content under a filename
  - read_file   : retrieve content by filename
  - ls          : list all stored filenames
  - edit_file   : overwrite existing file content

The VFS state is a plain Python dict passed by reference so all tools share
the same storage within a single agent run.
"""

from typing import Dict

# ─────────────────────────────────────────────
# Shared Virtual File System (in-memory store)
# ─────────────────────────────────────────────
_vfs: Dict[str, str] = {}


def reset_vfs():
    """Clear the VFS — call at the start of each task run."""
    _vfs.clear()


# ─────────────────────────────────────────────
# Core VFS operations
# ─────────────────────────────────────────────

def write_file(input_str: str) -> str:
    """
    Write content to a virtual file.

    Expected input format:  "filename.txt|content goes here"
    The pipe '|' separates the filename from the content.
    """
    if "|" not in input_str:
        return "Error: Input must be in format 'filename|content'."

    filename, content = input_str.split("|", 1)
    filename = filename.strip()
    content = content.strip()

    if not filename:
        return "Error: Filename cannot be empty."

    _vfs[filename] = content
    return f"File '{filename}' written successfully."


def read_file(filename: str) -> str:
    """
    Read content from a virtual file.

    Args:
        filename: The name of the file to read.

    Returns:
        File content string, or an error message if not found.
    """
    filename = filename.strip()
    if filename in _vfs:
        return _vfs[filename]
    return f"Error: File '{filename}' not found. Available files: {list(_vfs.keys())}"


def ls(_: str = "") -> str:
    """
    List all files currently stored in the virtual file system.

    Returns:
        Newline-separated list of filenames, or a message if empty.
    """
    if not _vfs:
        return "Virtual file system is empty."
    return "\n".join(sorted(_vfs.keys()))


def edit_file(input_str: str) -> str:
    """
    Edit (overwrite) an existing virtual file.

    Expected input format:  "filename.txt|new content goes here"
    Returns an error if the file does not already exist (use write_file for new files).
    """
    if "|" not in input_str:
        return "Error: Input must be in format 'filename|new_content'."

    filename, new_content = input_str.split("|", 1)
    filename = filename.strip()
    new_content = new_content.strip()

    if filename not in _vfs:
        return (f"Error: File '{filename}' not found. "
                f"Use write_file to create it first. "
                f"Available files: {list(_vfs.keys())}")

    _vfs[filename] = new_content
    return f"File '{filename}' updated successfully."


# ─────────────────────────────────────────────
# Helper for reading multiple files selectively
# ─────────────────────────────────────────────

def read_selected_files(filenames: list) -> str:
    """
    Read only the specified files and concatenate their contents.
    This enforces selective retrieval — do NOT read all files blindly.

    Args:
        filenames: List of filenames to read.

    Returns:
        Combined content string with headers per file.
    """
    result = []
    for fname in filenames:
        content = read_file(fname)
        result.append(f"=== {fname} ===\n{content}")
    return "\n\n".join(result)


def get_vfs_snapshot() -> Dict[str, str]:
    """Return a copy of the current VFS state (for logging/debugging)."""
    return dict(_vfs)
