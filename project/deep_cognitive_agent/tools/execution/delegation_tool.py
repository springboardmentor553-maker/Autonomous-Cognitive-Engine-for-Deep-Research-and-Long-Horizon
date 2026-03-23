from langchain_core.tools import tool
from brains.sub_agents import SUB_AGENTS

@tool
def task_delegate(agent_name: str, input_data: str):
    """
    Delegate a specific task to a specialized sub-agent.
    
    Available Specialists:
    - 'researcher': For deep-dive data gathering and technical extraction.
    - 'summarizer': For condensing long research into clean summaries.
    - 'comparator': For analyzing differences/similarities between datasets.
    - 'refiner': For the final professional polish and report merging.
    
    Use this when a task requires specialized focus beyond simple coordination.
    """
    if agent_name not in SUB_AGENTS:
        return f"Error: Agent '{agent_name}' not found in the specialist registry."
    
    # Hand-off execution to the specialist
    worker = SUB_AGENTS[agent_name]
    result = worker.invoke(input_data)
    
    return result