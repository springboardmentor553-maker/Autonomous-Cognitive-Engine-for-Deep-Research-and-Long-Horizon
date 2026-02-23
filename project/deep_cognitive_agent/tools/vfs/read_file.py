"""
VFS read_file tool - Reads content from a virtual file in state.

Usage:
    content = read_file(state, "filename.txt")
"""


def read_file(state: dict, filename: str) -> str:
    """Read content from a virtual file stored in state['files'].

    Args:
        state: The agent state dict containing a 'files' key.
        filename: Name of the virtual file to read.

    Returns:
        File content string, or error message if not found.
    """
    if filename in state["files"]:
        return state["files"][filename]
    else:
        return f"File '{filename}' not found."
