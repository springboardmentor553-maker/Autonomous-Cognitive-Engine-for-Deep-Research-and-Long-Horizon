"""
Task Delegation Tool - Milestone 3: Multi-Agent Collaboration

Delegates subtasks to specialized sub-agents via the SubAgentRegistry.
The supervisor calls this tool to assign work to the appropriate agent.

Workflow:
  1. Supervisor receives a request
  2. Supervisor decides which agent should handle it
  3. Supervisor calls delegate_task(agent_name, input_data)
  4. This tool looks up the agent in the registry, instantiates it, invokes it
  5. Result returns to the supervisor
"""

import time
from utils.helpers import is_rate_limit_error, is_server_overload_error, parse_retry_after


# ── Cached agent instances (avoid re-creating per call) ──────────────
_agent_cache = {}


def delegate_task(registry, llm, agent_name: str, input_data: str,
                  max_retries: int = 3) -> str:
    """Delegate a task to a named sub-agent via the registry.

    Args:
        registry: The SubAgentRegistry containing registered agents.
        llm: The LLM instance to pass to the agent.
        agent_name: Name of the sub-agent to delegate to.
        input_data: Input string to pass to the agent's invoke() method.
        max_retries: Number of retries on transient errors.

    Returns:
        Result string from the sub-agent.
    """
    # Check if agent exists in registry
    if agent_name not in registry.list_agents():
        available = registry.list_agents()
        return (f"Agent '{agent_name}' not found in registry. "
                f"Available agents: {available}")

    # Get or create cached agent instance
    if agent_name not in _agent_cache:
        _agent_cache[agent_name] = registry.create(agent_name, llm)

    agent = _agent_cache[agent_name]

    # Invoke with retry logic for transient errors
    for attempt in range(max_retries):
        try:
            result = agent.invoke(input_data)
            return result
        except Exception as e:
            err_str = str(e)
            if attempt < max_retries - 1:
                if is_rate_limit_error(err_str):
                    wait = parse_retry_after(err_str)
                    print(f"    ⏳ [{agent_name}] Rate limited. Waiting {wait}s...")
                    time.sleep(wait)
                    continue
                if is_server_overload_error(err_str):
                    wait = min(2 ** attempt * 10, 60)
                    print(f"    ⏳ [{agent_name}] Server overloaded. "
                          f"Waiting {wait}s...")
                    time.sleep(wait)
                    continue
            raise


def clear_agent_cache():
    """Clear the cached agent instances (useful for testing)."""
    _agent_cache.clear()
