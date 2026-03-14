from langchain_core.tools import tool


@tool
def edit_file(state: dict, filename: str, new_content: str):
    """
    Edit existing file content.
    """

    files = state.get("files", {})

    if filename not in files:
        return f"File '{filename}' does not exist."

    files[filename] = new_content

    return f"File '{filename}' updated successfully."