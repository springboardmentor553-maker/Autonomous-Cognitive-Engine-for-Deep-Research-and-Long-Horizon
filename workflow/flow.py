# File: workflow/flow.py

"""
═══════════════════════════════════════════════════════════════════
MAIN AGENT WORKFLOW: LangGraph State Machine with Delegation
═══════════════════════════════════════════════════════════════════
Milestone 3: Integrated Planning + Execution + Delegation
═══════════════════════════════════════════════════════════════════
"""

from typing import Literal, Annotated, List, Dict, Any
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, ToolMessage, SystemMessage
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver
from workflow.memory_state import AgentState
from brains.mainagent import write_todos, create_todo_items, TodoListInput
from brains.researcher import web_search
from brains.filetools import write_file, read_file, list_files, edit_file
from brains.delegation_tools import delegate_task, list_available_agents
import json
import uuid


def create_system_prompt() -> str:
    """Enhanced prompt for Milestone 3 with delegation capabilities."""
    from pathlib import Path
    instructions_path = Path(__file__).parent.parent / "instructions" / "mainagent.txt"

    if instructions_path.exists():
        with open(instructions_path, 'r') as f:
            return f.read()

    return """MILESTONE 3: AUTONOMOUS COGNITIVE ENGINE - DEEP RESEARCH AGENT

ROLE:
You are an advanced AI agent capable of executing complex, long-horizon tasks
through planning, context management, and sub-agent delegation.

CAPABILITIES:
1. PLANNING (write_todos): Break down complex goals into structured TODO steps
2. CONTEXT OFFLOADING (write_file, read_file, edit_file, ls): Store/retrieve information
3. DELEGATION (delegate_task, list_available_agents): Invoke specialized sub-agents

AVAILABLE SUB-AGENTS:
- search: Web search and information gathering specialist
- summarizer: Text summarization and condensation specialist  
- analysis: Comparative analysis and reasoning specialist

WORKFLOW:
1. PLAN: Always start by creating a TODO list (5-7 steps for complex tasks)
2. EXECUTE: Process each TODO item sequentially
   - For simple tasks: Use file tools directly
   - For specialized tasks: Use delegate_task to invoke sub-agents
3. STORE: Save important results to files using write_file
4. SYNTHESIZE: Read stored files and create final output
5. COMPLETE: Mark all TODOs as done and provide final result

CRITICAL RULES:
- ALWAYS create a plan first using write_todos
- Use file storage for intermediate results (context offloading)
- Delegate specialized tasks to appropriate sub-agents
- Read files before using their content
- Keep summaries concise (100-200 words per file)
- Track TODO status (pending/done)

RESPONSE FORMAT:
- Think step-by-step before acting
- Use tools explicitly (do not describe, actually call them)
- After each tool call, wait for the result before proceeding
"""


def create_agent_executor():
    """Create agent with full Milestone 3 capabilities."""

    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=0.0,
        max_tokens=2000
    )

    # Full tool suite including delegation
    tools = [
        write_todos, 
        web_search, 
        write_file, 
        read_file, 
        list_files, 
        edit_file,
        delegate_task,
        list_available_agents
    ]
    llm_with_tools = llm.bind_tools(tools)

    def agent_node(state: AgentState):
        """Main reasoning node with delegation awareness."""
        messages = state["messages"]
        todos = state.get("todos", [])
        delegation_history = state.get("delegation_history", [])

        if todos:
            pending_count = sum(1 for t in todos if t["status"] == "pending")
            completed_count = sum(1 for t in todos if t["status"] == "completed")
            status_msg = f"\n\n[STATUS: {completed_count}/{len(todos)} tasks completed, {pending_count} pending]"
            
            if delegation_history:
                status_msg += f"\n[DELEGATIONS: {len(delegation_history)} sub-agents invoked]"

            if messages and isinstance(messages[-1], HumanMessage):
                messages = messages.copy()
                messages[-1].content = messages[-1].content + status_msg

        response = llm_with_tools.invoke(messages)
        return {"messages": [response]}

    def tool_node_wrapper(state: AgentState):
        """Execute tools and handle write_todos and delegate_task specially."""
        messages = state["messages"]
        last_message = messages[-1]

        if hasattr(last_message, "tool_calls") and last_message.tool_calls:
            for tool_call in last_message.tool_calls:
                # Handle write_todos
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
                
                # Handle delegate_task
                elif tool_call["name"] == "delegate_task":
                    try:
                        args = tool_call["args"]
                        agent_type = args.get("agent_type", "search")
                        task_description = args.get("task_description", "")
                        delegate_id = args.get("delegate_id", str(uuid.uuid4())[:8])
                        
                        # Invoke sub-agent
                        from sub_agents.registry import sub_agent_registry
                        result = sub_agent_registry.invoke(agent_type, task_description, delegate_id)
                        
                        # Track delegation
                        delegation_history = state.get("delegation_history", [])
                        delegation_history.append({
                            "id": delegate_id,
                            "agent_type": agent_type,
                            "task": task_description,
                            "result_preview": result[:100] + "..." if len(result) > 100 else result
                        })
                        
                        tool_response = ToolMessage(
                            content=f"[Delegation Complete]\nAgent: {agent_type}\nID: {delegate_id}\nResult: {result}",
                            tool_call_id=tool_call["id"]
                        )

                        return {
                            "messages": [tool_response],
                            "delegation_history": delegation_history
                        }
                        
                    except Exception as e:
                        tool_response = ToolMessage(
                            content=f"Error delegating task: {str(e)}",
                            tool_call_id=tool_call["id"]
                        )
                        return {"messages": [tool_response]}

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
