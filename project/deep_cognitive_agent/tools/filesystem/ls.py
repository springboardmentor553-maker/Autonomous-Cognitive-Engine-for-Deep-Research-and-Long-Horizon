from langchain_core.tools import tool


@tool
def ls(state: dict):
    """
    List files in the virtual file system.
    """

    files = state.get("files", {})

    if not files:
        return "No files stored."

    return list(files.keys())