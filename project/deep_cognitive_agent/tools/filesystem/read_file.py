from langchain_core.tools import tool


@tool
def read_file(state: dict, filename: str):
    """
    Read file from virtual file system.
    """

    files = state.get("files", {})

    if filename not in files:
        return f"File '{filename}' not found."

    return files[filename]