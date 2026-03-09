from langchain_core.tools import tool
from langgraph.prebuilt import InjectedState
from langgraph.types import Command
from typing import Annotated

@tool
def write_file(filename: str, content: str, state: Annotated[dict, InjectedState]):
    """Write a file to the virtual file system."""
    return Command(update={"vfs": {filename: content}})

print(write_file.invoke({"filename": "test.txt", "content": "hello world", "state": {"vfs": {}}}))
