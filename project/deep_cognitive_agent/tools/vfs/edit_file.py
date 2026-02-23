"""
VFS edit_file tool - Edits content of an existing virtual file in state.

Usage:
    edit_file(state, "filename.txt", "new content here")
"""


def edit_file(state: dict, filename: str, new_content: str) -> str:
    """Edit content of a virtual file stored in state['files'].

    Args:
        state: The agent state dict containing a 'files' key.
        filename: Name of the virtual file to edit.
        new_content: New content to replace existing content.

    Returns:
        Success message or error if file not found.
    """
    if filename in state["files"]:
        state["files"][filename] = new_content
        return f"File '{filename}' updated."
    else:
        return f"File '{filename}' not found."
