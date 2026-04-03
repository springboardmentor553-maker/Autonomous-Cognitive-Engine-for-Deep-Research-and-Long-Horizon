def write_file(state, filename, content):
    state["files"][filename] = content
    state["trace"].append(f"WRITE: {filename}")

def read_file(state, filename):
    state["trace"].append(f"READ: {filename}")
    return state["files"].get(filename, "")

def edit_file(state, filename, content):
    state["files"][filename] = content
    state["trace"].append(f"EDIT: {filename}")

def ls(state):
    return list(state["files"].keys())