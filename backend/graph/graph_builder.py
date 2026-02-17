from langgraph.graph import StateGraph, END
from backend.graph.state import AgentState
from backend.graph.nodes import planning_node, validation_node, synthesis_node
from backend.graph.conditional_edges import planning_router


def build_graph():

    workflow = StateGraph(AgentState)

    workflow.add_node("planning", planning_node)
    workflow.add_node("validation", validation_node)
    workflow.add_node("synthesis", synthesis_node)

    workflow.set_entry_point("planning")

    workflow.add_edge("planning", "validation")

    workflow.add_conditional_edges("validation", planning_router)

    workflow.add_edge("synthesis", END)

    return workflow.compile()