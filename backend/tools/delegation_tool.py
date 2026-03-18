from backend.sub_agents.research_agent import research_agent
from backend.sub_agents.summarizer_agent import summarizer_agent
from backend.sub_agents.code_agent import code_agent


AGENT_REGISTRY = {
    "research": research_agent,
    "analysis": summarizer_agent,   # you can improve later
    "summary": summarizer_agent,
    "code": code_agent
}


def task_tool(task_type, input_data):

    if task_type not in AGENT_REGISTRY:
        raise ValueError(f"Agent {task_type} not found")

    print(f"DELEGATION → {task_type}_agent")

    agent = AGENT_REGISTRY[task_type]

    result = agent(input_data)

    print(f"SUBAGENT → {task_type} completed")

    return result