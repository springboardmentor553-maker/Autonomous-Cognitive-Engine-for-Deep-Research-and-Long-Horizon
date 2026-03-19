def write_file(state, filename, content):
    state["files"][filename] = content
    state["trace"].append(f"write_file({filename})")
    return f"{filename} written successfully"


def read_file(state, filename):
    if filename not in state["files"]:
        return "File not found"

    state["trace"].append(f"read_file({filename})")
    return state["files"][filename]


def edit_file(state, filename, new_content):
    if filename not in state["files"]:
        return "File not found"

    state["files"][filename] = new_content
    state["trace"].append(f"edit_file({filename})")
    return f"{filename} updated successfully"


def ls(state):
    files = list(state["files"].keys())
    state["trace"].append("ls()")
    return files