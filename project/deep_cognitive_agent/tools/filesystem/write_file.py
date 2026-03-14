from langchain_core.tools import tool


@tool
def write_file(state: dict, filename: str, content: str):
    """
    Save content into the virtual file system.
    """

    if "files" not in state:
        state["files"] = {}

    state["files"][filename] = content

    return f"File '{filename}' written successfully."