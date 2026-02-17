from typing import Literal
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, ToolMessage
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver
from workflow.memory_state import AgentState
from brains.mainagent import write_todos, create_todo_items, TodoListInput
from brains.researcher import web_search
import json
import uuid


def create_system_prompt() -> str:
    """Load system prompt from instructions file."""
    from pathlib import Path
    instructions_path = Path(__file__).parent.parent / "instructions" / "mainagent.txt"

    if instructions_path.exists():
        with open(instructions_path, 'r') as f:
            return f.read()

    return """You are an AI agent. For ANY complex task, you MUST use write_todos tool FIRST to create 4-6 structured sub-tasks. Each task must start with an action verb and be specific."""


def create_agent_executor():
    """Create the main agent workflow with task planning capability."""

    llm = ChatGoogleGenerativeAI(
model="gemini-2.5-flash-lite",
        temperature=0.3,
        max_tokens=2000
    )

    tools = [write_todos, web_search]
    llm_with_tools = llm.bind_tools(tools)

    def agent_node(state: AgentState):
        """Main reasoning node."""
        messages = state["messages"]
        todos = state.get("todos", [])

        if todos:
            pending_count = sum(1 for t in todos if t["status"] == "pending")
            completed_count = sum(1 for t in todos if t["status"] == "completed")
            status_msg = f"\n\n[CURRENT STATUS: {completed_count}/{len(todos)} tasks completed, {pending_count} pending]"

            if messages and isinstance(messages[-1], HumanMessage):
                messages = messages.copy()
                messages[-1].content = messages[-1].content + status_msg

        response = llm_with_tools.invoke(messages)
        return {"messages": [response]}

    def tool_node_wrapper(state: AgentState):
        """Execute tools and handle write_todos specially."""
        messages = state["messages"]
        last_message = messages[-1]

        if hasattr(last_message, "tool_calls") and last_message.tool_calls:
            for tool_call in last_message.tool_calls:
                if tool_call["name"] == "write_todos":
                    try:
                        todo_inputs = TodoListInput(**tool_call["args"]).todos
                        new_todos = create_todo_items(todo_inputs)

                        assert 4 <= len(new_todos) <= 6, f"Expected 4-6 TODOs, got {len(new_todos)}"

                        tool_response = ToolMessage(
                            content=json.dumps({
                                "status": "success",
                                "todo_count": len(new_todos),
                                "todos": [
                                    {
                                        "id": t["id"],
                                        "index": t["index"],
                                        "description": t["description"]
                                    }
                                    for t in new_todos
                                ]
                            }),
                            tool_call_id=tool_call["id"]
                        )

                        return {
                            "messages": [tool_response],
                            "todos": new_todos,
                            "current_todo_id": new_todos[0]["id"] if new_todos else None
                        }

                    except Exception as e:
                        print(f"\n⚠️  WARNING: write_todos failed: {e}")
                        print("Using fallback structured TODO creation...")

                        fallback_todos = [
                            {
                                "id": str(uuid.uuid4()),
                                "index": i,
                                "description": f"Step {i}: Complete sub-task {i} for the given objective",
                                "status": "pending",
                                "result": None,
                                "created_by": "fallback"
                            }
                            for i in range(1, 5)
                        ]

                        tool_response = ToolMessage(
                            content=json.dumps({
                                "status": "fallback",
                                "todo_count": 4,
                                "error": str(e),
                                "message": "Fallback structured TODO creation used"
                            }),
                            tool_call_id=tool_call["id"]
                        )

                        return {
                            "messages": [tool_response],
                            "todos": fallback_todos,
                            "current_todo_id": fallback_todos[0]["id"]
                        }

        tool_node = ToolNode(tools)
        return tool_node.invoke(state)

    def should_continue(state: AgentState) -> Literal["tools", "end"]:
        """Determine if agent should continue to tools or end."""
        messages = state["messages"]
        last_message = messages[-1]

        if hasattr(last_message, "tool_calls") and last_message.tool_calls:
            return "tools"

        return "end"

    workflow = StateGraph(AgentState)
    workflow.add_node("agent", agent_node)
    workflow.add_node("tools", tool_node_wrapper)
    workflow.set_entry_point("agent")
    workflow.add_conditional_edges(
        "agent",
        should_continue,
        {"tools": "tools", "end": END}
    )
    workflow.add_edge("tools", "agent")

    memory = MemorySaver()
    return workflow.compile(checkpointer=memory)