from __future__ import annotations

from langchain_core.tools import tool

from app.state import GraphState
from storage.file_store import edit_file, list_files, read_file, write_file


def build_storage_tools(shared_state: GraphState) -> list:
    @tool
    def write_file_tool(filename: str, content: str) -> str:
        """Write content to a file in the shared virtual file system."""
        path = write_file(filename, content, shared_state)
        return f"Wrote content to {path}"

    @tool
    def read_file_tool(filename: str) -> str:
        """Read a file from the shared virtual file system."""
        return read_file(filename, shared_state)

    @tool
    def list_files_tool() -> str:
        """List files stored in the shared virtual file system."""
        files = list_files(shared_state)
        return "\n".join(files) if files else "No files stored yet."

    @tool
    def edit_file_tool(filename: str, target_text: str, replacement_text: str) -> str:
        """Edit an existing file by replacing target text with replacement text."""
        return edit_file(filename, target_text, replacement_text, shared_state)

    return [write_file_tool, read_file_tool, list_files_tool, edit_file_tool]
