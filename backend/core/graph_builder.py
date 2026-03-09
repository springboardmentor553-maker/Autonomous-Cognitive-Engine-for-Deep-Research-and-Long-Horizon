from backend.sub_agents.research_agent import research_agent
from backend.sub_agents.summarizer_agent import summarizer_agent
from backend.sub_agents.code_agent import code_agent


def build_execution_graph():

    graph = {
        "research": research_agent,
        "summarize": summarizer_agent,
        "code": code_agent
    }

    return graph
