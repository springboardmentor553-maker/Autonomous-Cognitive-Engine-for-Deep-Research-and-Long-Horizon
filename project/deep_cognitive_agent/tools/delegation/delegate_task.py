from langchain_core.tools import tool
from project.deep_cognitive_agent.sub_agents.registry import sub_agents


@tool
def delegate_task(agent_name: str, input_data: str) -> str:
    """
    Delegates a task to a specialized sub-agent.
    """

    print(f"\n[DELEGATION] Sending task to sub-agent: {agent_name}")

    if agent_name not in sub_agents:
        raise ValueError(f"Sub-agent '{agent_name}' not found")

    agent = sub_agents[agent_name]

    result = agent.invoke(input_data)

    print(f"[DELEGATION COMPLETE] Result received from {agent_name}")

    return result