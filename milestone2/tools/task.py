from agents.registry import sub_agents
from state import state

def task(agent_name: str, input_data: str):

    state["trace"].append(f"task({agent_name})")

    agent = sub_agents.get(agent_name)

    if not agent:
        return "Agent not found."

    result = agent.invoke(input_data)

    return result