"""
VFS write_file tool - Writes content to a virtual file in state.

Usage:
    write_file(state, "filename.txt", "content here")
    The file is stored in state["files"][filename].
"""


def write_file(state: dict, filename: str, content: str) -> str:
    """Write content to a virtual file stored in state['files'].

    Args:
        state: The agent state dict containing a 'files' key.
        filename: Name of the virtual file to write.
        content: Content to write to the file.

    Returns:
        Success message string.
    """
    state["files"][filename] = content
    return f"File '{filename}' written successfully."
