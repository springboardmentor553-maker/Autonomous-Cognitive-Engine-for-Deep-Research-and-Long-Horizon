from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode
from typing import TypedDict, Annotated
import operator
from langgraph.types import Command
from langchain_core.tools import tool

class State(TypedDict):
    vfs: Annotated[dict, operator.ior]
    messages: list

@tool
def write_file(filename: str, content: str):
    """Write a file."""
    return Command(update={"vfs": {filename: content}})

def agent(state):
    return {"messages": [{"tool_calls": [{"name": "write_file", "args": {"filename": "a.txt", "content": "hello"}, "id": "1"}]}]}

workflow = StateGraph(State)
workflow.add_node("agent", agent)
workflow.add_node("tools", ToolNode([write_file]))

workflow.add_edge(START, "agent")
workflow.add_edge("agent", "tools")
workflow.add_edge("tools", END)
app = workflow.compile()

res = app.invoke({"vfs": {}, "messages": []})
print("Result state:", res["vfs"])
