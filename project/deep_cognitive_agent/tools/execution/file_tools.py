from langchain_core.tools import tool
from tools.filesystem.storage import vfs

@tool
def write_file(filename: str, content: str):
    """Write summarized content to a virtual file. Use this to offload context."""
    return vfs.write(filename, content)

@tool
def read_file(filename: str):
    """Read a specific file. ONLY load the files needed for the current step."""
    return vfs.read(filename)

@tool
def ls(_: str = ""):
    """List all available files in the virtual system."""
    files = vfs.ls()
    return f"Available files: {files}" if files else "No files found."

@tool
def edit_file(filename: str, new_content: str):
    """Update/Edit an existing virtual file with refined information."""
    if filename in vfs.files:
        return vfs.write(filename, new_content)
    return f"File '{filename}' not found."