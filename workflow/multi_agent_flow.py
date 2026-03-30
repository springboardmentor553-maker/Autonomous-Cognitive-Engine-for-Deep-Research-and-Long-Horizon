"""
Multi-Agent Workflow (Milestone 3)
WITH ACTIVE AGENT TRACKING
"""
import os
from typing import Literal
from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from workflow.multi_agent_state import MultiAgentState
from brains.supervisor import create_supervisor_agent, delegate_to_researcher, delegate_to_writer, delegate_to_reviewer
from brains.researcher import create_researcher_agent, web_search
from brains.writer import create_writer_agent
from brains.reviewer import create_reviewer_agent
from brains.filetools import get_fs_stats, read_file, write_file, edit_file


# Tool call tracking
TOOL_CALL_TRACKER = {
    "web_search": 0,
    "write_file": 0,
    "read_file": 0,
    "write_todos": 0,
    "edit_file": 0,
    "delegate_to_researcher": 0,
    "delegate_to_writer": 0,
    "delegate_to_reviewer": 0,
    "simple_text_operation": 0
}


def track_tool_call(tool_name: str, args: dict = None):
    """Track tool usage with details."""
    if tool_name in TOOL_CALL_TRACKER:
        TOOL_CALL_TRACKER[tool_name] += 1
        
        if tool_name == "web_search":
            query = args.get("query", "") if args else ""
            print(f"      🔍 WEB SEARCH CALL #{TOOL_CALL_TRACKER['web_search']}: {query[:60]}")
        elif tool_name == "write_file":
            filename = args.get("filename", "") if args else ""
            print(f"      💾 WRITE FILE CALL #{TOOL_CALL_TRACKER['write_file']}: {filename}")
        elif tool_name == "read_file":
            filename = args.get("filename", "") if args else ""
            print(f"      📖 READ FILE CALL #{TOOL_CALL_TRACKER['read_file']}: {filename}")
        elif tool_name in ["delegate_to_researcher", "delegate_to_writer", "delegate_to_reviewer"]:
            print(f"      🤝 DELEGATION: {tool_name}")


def execute_tools(tool_calls, available_tools):
    """Execute tool calls manually and return results."""
    results = []
    
    for tool_call in tool_calls:
        tool_name = tool_call.get("name", "")
        tool_args = tool_call.get("args", {})
        tool_id = tool_call.get("id", "")
        
        tool_func = None
        for t in available_tools:
            if t.name == tool_name:
                tool_func = t
                break
        
        if tool_func:
            try:
                result = tool_func.invoke(tool_args)
                results.append(ToolMessage(
                    content=str(result),
                    tool_call_id=tool_id
                ))
            except Exception as e:
                results.append(ToolMessage(
                    content=f"Error executing {tool_name}: {str(e)}",
                    tool_call_id=tool_id
                ))
        else:
            results.append(ToolMessage(
                content=f"Tool {tool_name} not found",
                tool_call_id=tool_id
            ))
    
    return results


def supervisor_node(state: MultiAgentState) -> MultiAgentState:
    """Supervisor with ACTIVE AGENT TRACKING"""
    print(f"\n{'='*80}")
    print("SUPERVISOR - Analyzing workflow state")
    print(f"{'='*80}")
    
    fs_stats = get_fs_stats()
    state["created_files"] = fs_stats.get("files", [])
    state["active_agent"] = "supervisor"  # TRACK ACTIVE AGENT
    
    current_step = state.get('current_step', 1)
    completed_steps = state.get('completed_steps', [])
    todos = state.get("todos", [])
    
    print(f"Current step: {current_step}")
    print(f"Completed steps: {completed_steps}")
    
    supervisor_llm, system_prompt = create_supervisor_agent()
    messages = state["messages"] if state.get("messages") else []
    
    if not any(isinstance(m, SystemMessage) for m in messages):
        messages = [SystemMessage(content=system_prompt)] + messages
    
    if todos and current_step <= len(todos):
        current_todo = todos[current_step - 1]
        task_description = current_todo.get('description', '')
        
        print(f"\n   📋 Delegating step {current_step}: {task_description}")
        
        delegation_prompt = f"""Coordinate workflow step {current_step}/{len(todos)}.

CURRENT TASK: {task_description}

DELEGATION DECISION:
- Research/data gathering → delegate_to_researcher
- Writing/documents → delegate_to_writer
- Review/quality check → delegate_to_reviewer

Call the appropriate delegation tool NOW.
"""
        
        messages = messages + [HumanMessage(content=delegation_prompt)]
        response = supervisor_llm.invoke(messages)
        messages = messages + [response]
        
        if hasattr(response, "tool_calls") and response.tool_calls:
            print(f"\n   🤝 Supervisor Delegation Calls:")
            for tc in response.tool_calls:
                tool_name = tc.get("name", "")
                args = tc.get("args", {})
                track_tool_call(tool_name, args)
            
            delegation_tools = [delegate_to_researcher, delegate_to_writer, delegate_to_reviewer]
            tool_results = execute_tools(response.tool_calls, delegation_tools)
            messages = messages + tool_results
        
        state["messages"] = messages
        
    elif todos and len(todos) > 0 and len(completed_steps) >= len(todos):
        print(f"\n   ✅ All {len(todos)} steps completed!")
        completion_msg = HumanMessage(content="All tasks completed successfully")
        state["messages"] = messages + [completion_msg]
    else:
        state["messages"] = messages
    
    return state


def researcher_node(state: MultiAgentState) -> MultiAgentState:
    """Researcher with ACTIVE AGENT TRACKING"""
    print(f"\n{'='*80}")
    print("RESEARCHER - Executing research task")
    print(f"{'='*80}")
    
    current_step = state.get("current_step", 1)
    todos = state.get("todos", [])
    
    if current_step <= len(todos):
        task = todos[current_step - 1]
        print(f"Task: {task.get('description', 'Unknown')}")
    
    state["researcher_status"] = "working"
    state["active_agent"] = "researcher"  # TRACK ACTIVE AGENT
    
    researcher_llm, system_prompt = create_researcher_agent()
    messages = state["messages"]
    
    if not any(isinstance(m, SystemMessage) for m in messages):
        messages = [SystemMessage(content=system_prompt)] + messages
    
    if current_step <= len(todos):
        task_msg = HumanMessage(content=f"Execute: {todos[current_step - 1].get('description', '')}. Use web_search (2x minimum) and write_file.")
        messages = messages + [task_msg]
    
    response = researcher_llm.invoke(messages)
    messages = messages + [response]
    
    if hasattr(response, "tool_calls") and response.tool_calls:
        print(f"\n   🔧 Researcher Tool Calls:")
        for tc in response.tool_calls:
            tool_name = tc.get("name", "")
            args = tc.get("args", {})
            track_tool_call(tool_name, args)
        
        tool_results = execute_tools(response.tool_calls, [web_search, write_file, read_file])
        messages = messages + tool_results
    
    state["messages"] = messages
    state["researcher_status"] = "complete"
    state["completed_steps"] = state.get("completed_steps", []) + [current_step]
    state["current_step"] = current_step + 1
    
    fs_stats = get_fs_stats()
    state["created_files"] = fs_stats.get("files", [])
    
    print(f"\n   ✅ Research complete - Step {current_step} done")
    
    return state


def writer_node(state: MultiAgentState) -> MultiAgentState:
    """Writer with ACTIVE AGENT TRACKING"""
    print(f"\n{'='*80}")
    print("WRITER - Creating content")
    print(f"{'='*80}")
    
    current_step = state.get("current_step", 1)
    todos = state.get("todos", [])
    
    if current_step <= len(todos):
        task = todos[current_step - 1]
        print(f"Task: {task.get('description', 'Unknown')}")
    
    state["writer_status"] = "working"
    state["active_agent"] = "writer"  # TRACK ACTIVE AGENT
    
    writer_llm, system_prompt = create_writer_agent()
    messages = state["messages"]
    
    if not any(isinstance(m, SystemMessage) for m in messages):
        messages = [SystemMessage(content=system_prompt)] + messages
    
    if current_step <= len(todos):
        task_msg = HumanMessage(content=f"Execute: {todos[current_step - 1].get('description', '')}. Read research files and write comprehensive report.")
        messages = messages + [task_msg]
    
    response = writer_llm.invoke(messages)
    messages = messages + [response]
    
    if hasattr(response, "tool_calls") and response.tool_calls:
        print(f"\n   🔧 Writer Tool Calls:")
        for tc in response.tool_calls:
            tool_name = tc.get("name", "")
            args = tc.get("args", {})
            track_tool_call(tool_name, args)
        
        tool_results = execute_tools(response.tool_calls, [read_file, write_file, edit_file])
        messages = messages + tool_results
    
    state["messages"] = messages
    state["writer_status"] = "complete"
    state["completed_steps"] = state.get("completed_steps", []) + [current_step]
    state["current_step"] = current_step + 1
    
    fs_stats = get_fs_stats()
    state["created_files"] = fs_stats.get("files", [])
    
    print(f"\n   ✅ Writing complete - Step {current_step} done")
    
    return state


def reviewer_node(state: MultiAgentState) -> MultiAgentState:
    """Reviewer with ACTIVE AGENT TRACKING"""
    print(f"\n{'='*80}")
    print("REVIEWER - Quality assurance")
    print(f"{'='*80}")
    
    current_step = state.get("current_step", 1)
    todos = state.get("todos", [])
    
    if current_step <= len(todos):
        task = todos[current_step - 1]
        print(f"Task: {task.get('description', 'Unknown')}")
    
    state["reviewer_status"] = "working"
    state["active_agent"] = "reviewer"  # TRACK ACTIVE AGENT
    
    reviewer_llm, system_prompt = create_reviewer_agent()
    messages = state["messages"]
    
    if not any(isinstance(m, SystemMessage) for m in messages):
        messages = [SystemMessage(content=system_prompt)] + messages
    
    if current_step <= len(todos):
        task_msg = HumanMessage(content=f"Execute: {todos[current_step - 1].get('description', '')}. Read report and write final reviewed version.")
        messages = messages + [task_msg]
    
    response = reviewer_llm.invoke(messages)
    messages = messages + [response]
    
    if hasattr(response, "tool_calls") and response.tool_calls:
        print(f"\n   🔧 Reviewer Tool Calls:")
        for tc in response.tool_calls:
            tool_name = tc.get("name", "")
            args = tc.get("args", {})
            track_tool_call(tool_name, args)
        
        tool_results = execute_tools(response.tool_calls, [read_file, write_file, edit_file])
        messages = messages + tool_results
    
    state["messages"] = messages
    state["reviewer_status"] = "complete"
    state["completed_steps"] = state.get("completed_steps", []) + [current_step]
    state["current_step"] = current_step + 1
    
    fs_stats = get_fs_stats()
    state["created_files"] = fs_stats.get("files", [])
    
    print(f"\n   ✅ Review complete - Step {current_step} done")
    
    return state


def route_next_agent(state: MultiAgentState) -> Literal["researcher", "writer", "reviewer", "end"]:
    """Routing logic"""
    current_step = state.get("current_step", 1)
    completed = state.get("completed_steps", [])
    todos = state.get("todos", [])
    
    if todos and len(completed) >= len(todos):
        print("\n✅ All steps complete - ending workflow")
        return "end"
    
    if current_step <= 3:
        print(f"\n→ Routing to RESEARCHER for step {current_step}")
        return "researcher"
    elif current_step == 4:
        print(f"\n→ Routing to WRITER for step {current_step}")
        return "writer"
    elif current_step == 5:
        print(f"\n→ Routing to REVIEWER for step {current_step}")
        return "reviewer"
    else:
        print("\n✅ Workflow complete")
        return "end"


def create_multi_agent_workflow():
    """Create the multi-agent workflow graph."""
    workflow = StateGraph(MultiAgentState)
    
    workflow.add_node("supervisor", supervisor_node)
    workflow.add_node("researcher", researcher_node)
    workflow.add_node("writer", writer_node)
    workflow.add_node("reviewer", reviewer_node)
    
    workflow.add_conditional_edges(
        "supervisor",
        route_next_agent,
        {
            "researcher": "researcher",
            "writer": "writer",
            "reviewer": "reviewer",
            "end": END
        }
    )
    
    workflow.add_edge("researcher", "supervisor")
    workflow.add_edge("writer", "supervisor")
    workflow.add_edge("reviewer", "supervisor")
    
    workflow.set_entry_point("supervisor")
    
    return workflow.compile()


def get_tool_call_stats():
    """Get tool call statistics."""
    return TOOL_CALL_TRACKER.copy()


def reset_tool_call_stats():
    """Reset tool call statistics."""
    global TOOL_CALL_TRACKER
    TOOL_CALL_TRACKER = {
        "web_search": 0,
        "write_file": 0,
        "read_file": 0,
        "write_todos": 0,
        "edit_file": 0,
        "delegate_to_researcher": 0,
        "delegate_to_writer": 0,
        "delegate_to_reviewer": 0,
        "simple_text_operation": 0
    }