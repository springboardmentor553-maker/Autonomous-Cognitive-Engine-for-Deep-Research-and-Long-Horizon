from typing import Literal
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, ToolMessage
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver
from workflow.memory_state import AgentState
from brains.mainagent import write_todos, create_todo_items, TodoListInput
from brains.researcher import web_search
from brains.filetools import write_file, read_file, list_files, edit_file
import json
import uuid


def create_system_prompt() -> str:
    """Enhanced prompt for Milestone 2 context offloading."""
    from pathlib import Path
    instructions_path = Path(__file__).parent.parent / "instructions" / "mainagent.txt"

    if instructions_path.exists():
        with open(instructions_path, 'r') as f:
            return f.read()

    return """MILESTONE 2: CONTEXT OFFLOADING - CRITICAL RULES

FILE SYSTEM USAGE:
1. Store SUMMARIES, not raw data (condense first, then save)
2. Use MEANINGFUL filenames (climate_causes_summary.txt, NOT file1.txt)
3. SELECTIVE retrieval - only read files you need NOW
4. Use edit_file() to update, NOT create duplicates
5. Avoid context window explosion - keep memory clean

WORKFLOW PATTERN:
Step 1: Plan with write_todos (5 steps)
Step 2: Process input → SUMMARIZE → write_file("meaningful_name.txt")
Step 3: When needed, read_file("specific_file.txt") - NOT all files at once
Step 4: If updating: edit_file() - NO duplicates
Step 5: Synthesize from SELECTED files only

NAMING CONVENTION:
✓ GOOD: "section1_analysis.txt", "financial_q1_summary.txt"
✗ BAD: "file1.txt", "data.txt", "output.txt"

DEPENDENCY CHAIN:
- Each step builds on previous
- Files store intermediate state
- Retrieval is targeted and purposeful
- No unnecessary file creation"""


def create_agent_executor():
    """Create agent with enhanced file tools."""

    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=0.0,
        max_tokens=2000
    )

    # Full tool suite including edit_file
    tools = [write_todos, web_search, write_file, read_file, list_files, edit_file]
    llm_with_tools = llm.bind_tools(tools)

    def agent_node(state: AgentState):
        """Main reasoning node."""
        messages = state["messages"]
        todos = state.get("todos", [])

        if todos:
            pending_count = sum(1 for t in todos if t["status"] == "pending")
            completed_count = sum(1 for t in todos if t["status"] == "completed")
            status_msg = f"\n\n[STATUS: {completed_count}/5 tasks completed, {pending_count} pending]"

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
                        assert len(new_todos) == 5, f"Must be EXACTLY 5 TODOs, got {len(new_todos)}"

                        tool_response = ToolMessage(
                            content=json.dumps({
                                "status": "success",
                                "todo_count": 5,
                                "todos": [{"id": t["id"], "index": t["index"], "description": t["description"]} for t in new_todos]
                            }),
                            tool_call_id=tool_call["id"]
                        )

                        return {
                            "messages": [tool_response],
                            "todos": new_todos,
                            "current_todo_id": new_todos[0]["id"]
                        }

                    except Exception as e:
                        # Fallback without warning print
                        fallback_todos = [
                            {
                                "id": str(uuid.uuid4()),
                                "index": i,
                                "description": [
                                    "Research current best practices and frameworks",
                                    "Analyze specific requirements and constraints",
                                    "Design detailed solution architecture",
                                    "Develop implementation roadmap and steps",
                                    "Validate through testing and review process"
                                ][i-1],
                                "status": "pending",
                                "result": None,
                                "created_by": "fallback-strong"
                            }
                            for i in range(1, 6)
                        ]

                        tool_response = ToolMessage(
                            content=json.dumps({"status": "fallback", "todo_count": 5}),
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
        """Continue until all tools executed."""
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
