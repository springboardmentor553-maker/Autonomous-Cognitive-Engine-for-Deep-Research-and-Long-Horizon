"""
delegation_tool.py - Sub-Agent Delegation Tool for Milestone 3
Deep Cognitive Task Framework

Provides the `task` tool that the supervisor agent calls to delegate a TODO
item to a specialized sub-agent. The tool:
  1. Looks up the named agent in SUB_AGENT_REGISTRY
  2. Dispatches the task + context to the sub-agent's runner
  3. Returns the result as JSON for the supervisor to integrate

Also provides `list_agents` — lets the supervisor discover available sub-agents.
"""

import json
import time
from langchain_core.tools import tool
from sub_agents.registry import SUB_AGENT_REGISTRY, run_sub_agent, list_available_agents


# ─────────────────────────────────────────────
# Delegation Tools
# ─────────────────────────────────────────────

@tool
def task(agent_name: str, sub_task: str, context: str = "") -> str:
    """
    Delegate a sub-task to a specialized sub-agent.

    Use this tool when a TODO item is best handled by a specialist:
      - Complex research  → delegate to "web_search_agent"
      - Summarizing text  → delegate to "summarization_agent"
      - Code review/draft → delegate to "code_analysis_agent"

    WHEN TO USE:
      Call this instead of doing the work yourself when:
      - The sub-task requires deep, structured research
      - You need to condense a large piece of text
      - The sub-task involves code analysis or generation

    HOW TO USE:
      1. Call list_agents() first if you are unsure which agent to use.
      2. Call task(agent_name, sub_task, context) to delegate.
      3. Save the returned result using write_file to the virtual file system.
      4. Mark the relevant TODO as complete with mark_todo_complete.

    Args:
        agent_name: Name of the sub-agent to delegate to. One of:
                    "summarization_agent", "web_search_agent", "code_analysis_agent"
        sub_task  : Clear, concise description of what the sub-agent should do.
        context   : Optional content, code, or background to pass to the sub-agent.

    Returns:
        JSON with:
          - success     : bool
          - agent_name  : the agent that handled the task
          - sub_task    : the task description
          - result      : the sub-agent's output (str)
          - duration_s  : time taken in seconds
    """
    if agent_name not in SUB_AGENT_REGISTRY:
        available = list(SUB_AGENT_REGISTRY.keys())
        return json.dumps({
            "success": False,
            "error": f"Unknown agent '{agent_name}'.",
            "available_agents": available,
            "hint": "Call list_agents() to see all available sub-agents and their descriptions.",
        })

    start = time.time()
    try:
        result = run_sub_agent(
            agent_name=agent_name,
            task=sub_task,
            context=context,
        )
        duration = round(time.time() - start, 2)
        return json.dumps({
            "success": True,
            "agent_name": agent_name,
            "sub_task": sub_task,
            "result": result,
            "duration_s": duration,
            "result_length": len(result),
        }, indent=2)
    except Exception as e:
        duration = round(time.time() - start, 2)
        return json.dumps({
            "success": False,
            "agent_name": agent_name,
            "sub_task": sub_task,
            "error": str(e),
            "duration_s": duration,
        })


@tool
def list_agents() -> str:
    """
    List all available specialized sub-agents and their descriptions.

    Call this to discover which agent to use for a given type of task before
    calling the task() delegation tool.

    Returns:
        JSON with available agent names, descriptions, and example tasks.
    """
    agents = list_available_agents()
    return json.dumps({
        "available_agents": agents,
        "total": len(agents),
        "usage_hint": (
            "Pass one of the 'name' values to the task() tool as agent_name. "
            "Choose based on the task type and the agent's description."
        ),
    }, indent=2)


# ─────────────────────────────────────────────
# Export
# ─────────────────────────────────────────────

DELEGATION_TOOLS = [task, list_agents]
