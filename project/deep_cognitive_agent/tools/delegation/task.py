"""
Task Delegation Tool - Delegates subtasks to sub-agents.

Placeholder for future milestones where the supervisor can
delegate specific tasks to registered sub-agents.
"""


def delegate_task(agent_name: str, task: str, context: dict = None) -> str:
    """Delegate a task to a named sub-agent.

    Args:
        agent_name: Name of the sub-agent to delegate to.
        task: Task description to execute.
        context: Optional context dict to pass along.

    Returns:
        Result string from the sub-agent.
    """
    # Placeholder — will be wired to SubAgentRegistry in future milestones
    return f"Task '{task}' delegated to {agent_name}. (not yet implemented)"
