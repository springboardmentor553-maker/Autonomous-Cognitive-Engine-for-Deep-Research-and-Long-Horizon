"""
tools/delegation/delegate_task.py
====================================
Task Delegation Tool

Mentor exact spec:

    def task(agent_name: str, input_data: str):
        if agent_name not in sub_agents:
            return "Agent not found."
        agent  = sub_agents[agent_name]
        result = agent.invoke(input_data)
        return result

    task("summarizer", content)

This is exposed as a LangChain @tool so the supervisor LLM can call it
during its ReAct reasoning loop.

After calling this tool the supervisor must:
    1. Take the returned result
    2. Call write_file to store it in the VFS
    3. Continue to the next TODO

Mentor integration check:
    "check whether the result returned by the sub-agent
     is integrated into the workflow"
"""

import json
from datetime import datetime, timezone

from langchain_core.tools import tool
from utils.logger import get_logger

logger = get_logger(__name__)


@tool
def delegate_task(agent_name: str, input_data: str) -> str:
    """
    Delegate a task to a specialized sub-agent.

    The supervisor calls this when a TODO needs a specialist.
    After receiving the result, the supervisor MUST call write_file
    to store it, then continue to the next TODO.

    Available agents:
      "summarizer"   -> summarizes text, extracts key points
      "web_searcher" -> searches the web, returns structured findings

    When to delegate:
      "summarizer"   : task says summarize / condense / extract key points
      "web_searcher" : task says search / find / look up / research

    When NOT to delegate (supervisor handles directly):
      compare / analyze / synthesize / write report / edit files

    Args:
        agent_name : "summarizer" or "web_searcher"
        input_data : the text to summarize OR the search query to run

    Returns:
        JSON string containing the sub-agent result + metadata
    """
    from sub_agents.registry import sub_agents

    logger.info(f"delegate_task -> '{agent_name}': {input_data[:80]}")

    # ── Mentor exact pattern ──────────────────────────────────────────────────
    if agent_name not in sub_agents:
        available = list(sub_agents.keys())
        logger.error(f"Agent '{agent_name}' not found. Available: {available}")
        return f"Agent not found. Available: {available}"

    agent  = sub_agents[agent_name]
    result = agent.invoke(input_data)
    # ─────────────────────────────────────────────────────────────────────────

    logger.info(f"delegate_task: '{agent_name}' returned {len(str(result))} chars")

    # Return JSON envelope so tools_node can update delegation_log in state
    return json.dumps({
        "action":       "delegate_task",
        "agent_name":   agent_name,
        "input_data":   input_data[:100],
        "result":       result,
        "status":       "completed",
        "delegated_at": datetime.now(timezone.utc).isoformat(),
    })