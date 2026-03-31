from langchain_core.tools import tool
from langgraph.prebuilt import InjectedState
from typing import Annotated

@tool
def ls(state: Annotated[dict, InjectedState]):
    """List files in the virtual file system."""
    vfs = state.get("vfs", {})
    return list(vfs.keys())

print(ls.invoke({"state": {"vfs": {"file.txt": "hello"}}}))
