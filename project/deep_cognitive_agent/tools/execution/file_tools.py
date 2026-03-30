from langchain_core.tools import tool
from langgraph.types import Command
from langgraph.prebuilt import InjectedState
from typing import Annotated

@tool
def write_file(filename: str, content: str):
    """Saves a file to the Virtual File System (VFS). Use this to store intermediate research."""
    
    print(f"💾 Saving to VFS State: {filename}")
    
    # Push the new file into the state dictionary
    return Command(
        update={
            "files": {filename: content}
        }
    )

@tool
def read_file(filename: str, state: Annotated[dict, InjectedState]):
    """
    Reads the content of a file from the Virtual File System (VFS).
    Use this to retrieve intermediate results for final synthesis.
    """
    files = state.get("files", {})
    
    if filename in files:
        print(f"📖 Reading from VFS: {filename}")
        return files[filename]
    else:
        return f"Error: File '{filename}' not found in the VFS. Available files: {list(files.keys())}"

@tool
def ls():
    """Lists all files currently saved in the Virtual File System."""
    return "Check your state['files'] keys for current directory contents."