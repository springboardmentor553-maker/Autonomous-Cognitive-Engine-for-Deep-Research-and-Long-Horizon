from langgraph.graph import StateGraph, START, END
from graphs.state import AgentState
from graphs.supervisor_node import supervisor_router
from graphs.execution_node import (
    execute_researcher, 
    execute_summarizer, 
    execute_comparator, 
    execute_refiner
)

def build_cognitive_engine():
    # Initialize the graph with your AgentState definition
    workflow = StateGraph(AgentState)

    # Add the execution nodes (The Workers)
    workflow.add_node("researcher", execute_researcher)
    workflow.add_node("summarizer", execute_summarizer)
    workflow.add_node("comparator", execute_comparator)
    workflow.add_node("refiner", execute_refiner)

    # Define the routing map once to keep things clean
    routing_map = {
        "researcher": "researcher",
        "summarizer": "summarizer",
        "comparator": "comparator",
        "refiner": "refiner",
        "__end__": END
    }

    # Set the Supervisor as the entry point
    workflow.set_conditional_entry_point(
        supervisor_router,
        routing_map
    )

    # THE FIX: Use conditional edges to loop back to the router logic 
    # instead of pointing back to START.
    workflow.add_conditional_edges("researcher", supervisor_router, routing_map)
    workflow.add_conditional_edges("summarizer", supervisor_router, routing_map)
    workflow.add_conditional_edges("comparator", supervisor_router, routing_map)
    workflow.add_conditional_edges("refiner", supervisor_router, routing_map)

    # Compile and return the engine
    return workflow.compile()