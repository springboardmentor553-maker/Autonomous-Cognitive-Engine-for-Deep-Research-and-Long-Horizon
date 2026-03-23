from langchain_core.tools import tool

@tool
def write_file(filename: str, content: str) -> str:
    """Write content to a virtual file."""
    return f"FILE_WRITE::{filename}::{content}"

@tool
def read_file(filename: str) -> str:
    """Read content from a virtual file."""
    return f"FILE_READ::{filename}"

@tool
def edit_file(filename: str, new_content: str) -> str:
    """Overwrite a virtual file with new content."""
    return f"FILE_EDIT::{filename}::{new_content}"

@tool
def ls() -> str:
    """List all virtual files currently stored."""
    return "FILE_LS"