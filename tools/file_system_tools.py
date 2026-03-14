"""
Virtual File System tools – Milestone 2.

Provides four tools that let the agent manage an in-memory file system
stored inside ``state["files"]``.

Tools
-----
ls          – list all files
read_file   – return the content of a file
write_file  – create a new file (fails if it already exists)
edit_file   – overwrite / update an existing file

All state mutations are communicated through a structured JSON payload
that the graph node ``process_tool_calls`` intercepts and applies.
"""

from __future__ import annotations

import json

from langchain_core.tools import tool


# ---------------------------------------------------------------------------
# ls
# ---------------------------------------------------------------------------

@tool
def ls(directory: str = "/") -> str:
    """
    List all files currently stored in the virtual file system.

    Parameters
    ----------
    directory : str
        Ignored (included for UX familiarity); always lists all files.

    Returns
    -------
    str
        A JSON payload containing the list of filenames, or an empty list
        message if no files exist yet.
    """
    # The actual file listing is injected by the graph node because we don't
    # have direct state access inside a @tool function.  We return a sentinel
    # that the node replaces with real data.
    payload = json.dumps({"action": "ls"})
    return payload


# ---------------------------------------------------------------------------
# read_file
# ---------------------------------------------------------------------------

@tool
def read_file(filename: str) -> str:
    """
    Read the content of a file from the virtual file system.

    Parameters
    ----------
    filename : str
        The name of the file to read (e.g. "research_notes.txt").

    Returns
    -------
    str
        A JSON payload that the graph node resolves to the file's content,
        or an error if the file does not exist.
    """
    if not filename or not filename.strip():
        return "ERROR: filename must not be empty."

    payload = json.dumps({"action": "read_file", "filename": filename.strip()})
    return payload


# ---------------------------------------------------------------------------
# write_file
# ---------------------------------------------------------------------------

@tool
def write_file(filename: str, content: str) -> str:
    """
    Create a new file in the virtual file system.

    Parameters
    ----------
    filename : str
        Name for the new file (e.g. "summary.md").
    content : str
        Text content to store inside the file.

    Returns
    -------
    str
        A JSON payload instructing the graph node to create the file.
        Returns an error string if inputs are invalid.
    """
    if not filename or not filename.strip():
        return "ERROR: filename must not be empty."
    if content is None:
        return "ERROR: content must not be None."

    payload = json.dumps(
        {"action": "write_file", "filename": filename.strip(), "content": content}
    )
    return payload


# ---------------------------------------------------------------------------
# edit_file
# ---------------------------------------------------------------------------

@tool
def edit_file(filename: str, new_content: str) -> str:
    """
    Overwrite the content of an existing file in the virtual file system.

    Parameters
    ----------
    filename : str
        Name of the file to update.
    new_content : str
        The replacement content for the file.

    Returns
    -------
    str
        A JSON payload instructing the graph node to update the file.
    """
    if not filename or not filename.strip():
        return "ERROR: filename must not be empty."
    if new_content is None:
        return "ERROR: new_content must not be None."

    payload = json.dumps(
        {"action": "edit_file", "filename": filename.strip(), "new_content": new_content}
    )
    return payload


# ---------------------------------------------------------------------------
# Helper: apply VFS mutations to the state files dict
# ---------------------------------------------------------------------------

def apply_vfs_action(raw_output: str, files: dict[str, str]) -> tuple[str, dict[str, str]]:
    """
    Interpret a VFS tool payload and apply the mutation to *files*.

    Parameters
    ----------
    raw_output : str
        The raw string returned by one of the VFS tools.
    files : dict[str, str]
        Current virtual file system mapping (copied from state).

    Returns
    -------
    tuple[str, dict[str, str]]
        (human_readable_result, updated_files_dict)
        The first element is what should be sent back to the agent as the
        tool result.  The second is the updated files mapping.
    """
    try:
        data: dict = json.loads(raw_output)
    except json.JSONDecodeError:
        # Not a VFS payload – pass through as-is.
        return raw_output, files

    action = data.get("action")
    updated_files = dict(files)  # shallow copy so we don't mutate state directly

    if action == "ls":
        if not updated_files:
            return "No files found. The virtual file system is empty.", updated_files
        file_list = "\n".join(f"  - {name}" for name in sorted(updated_files))
        return f"Files in virtual file system:\n{file_list}", updated_files

    elif action == "read_file":
        fname = data["filename"]
        if fname not in updated_files:
            return f"ERROR: File '{fname}' does not exist.", updated_files
        return updated_files[fname], updated_files

    elif action == "write_file":
        fname = data["filename"]
        if fname in updated_files:
            return (
                f"ERROR: File '{fname}' already exists. Use edit_file to update it.",
                updated_files,
            )
        updated_files[fname] = data["content"]
        return f"File '{fname}' created successfully.", updated_files

    elif action == "edit_file":
        fname = data["filename"]
        if fname not in updated_files:
            return (
                f"ERROR: File '{fname}' does not exist. Use write_file to create it.",
                updated_files,
            )
        updated_files[fname] = data["new_content"]
        return f"File '{fname}' updated successfully.", updated_files

    # Unknown action – return raw
    return raw_output, updated_files
