from langchain_core.tools import tool
from sub_agents.registry import sub_agents

@tool
def task(agent_name: str, input_data: str) -> str:
    """
    Delegate a sub-task to a specialized sub-agent.
    Available agents:
    - 'summarizer': summarizes long text
    - 'web_search': searches the web for a query
    """
    agent = sub_agents.get(agent_name)
    if not agent:
        return f"Agent '{agent_name}' not found. Available: {list(sub_agents.keys())}"
    result = agent.invoke(input_data)
    return result