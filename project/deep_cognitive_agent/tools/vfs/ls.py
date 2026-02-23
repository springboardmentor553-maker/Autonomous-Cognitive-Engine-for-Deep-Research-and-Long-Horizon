"""
VFS ls tool - Lists all virtual files in state.

Usage:
    file_list = ls(state)
"""


def ls(state: dict) -> list:
    """List all files in the virtual file system.

    Args:
        state: The agent state dict containing a 'files' key.

    Returns:
        List of filenames currently stored in the VFS.
    """
    return list(state["files"].keys())
