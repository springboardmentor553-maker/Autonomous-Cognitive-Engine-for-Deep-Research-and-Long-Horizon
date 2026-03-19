from agents.registry import sub_agents
from state import state   # 👈 import state

def task(agent_name: str, input_data: str):
    
    # 👇 ADD THIS (trace delegation)
    state["trace"].append(f"task({agent_name})")

    print(f"[Supervisor] Delegating to {agent_name}...")

    agent = sub_agents.get(agent_name)

    if not agent:
        return "Agent not found."

    result = agent.invoke(input_data)

    print(f"[Sub-Agent] {agent_name} completed task.")

    return result