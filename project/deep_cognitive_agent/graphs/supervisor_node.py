from graphs.state import AgentState

def supervisor_router(state: AgentState) -> str:
    """
    Examines the todos in the state and routes to the appropriate sub-agent.
    """
    todos = state.get("todos", [])
    
    for todo in todos:
        if todo.get("status") == "pending":
            task_text = todo.get("task", "").lower()
            
            # Routing logic mapped to your SUB_AGENTS keys in brains/sub_agents.py
            if "research" in task_text or "search" in task_text:
                return "researcher"
            elif "summarize" in task_text:
                return "summarizer"
            elif "compare" in task_text:
                return "comparator"
            elif "refine" in task_text:
                return "refiner"
            else:
                # Default fallback
                return "researcher" 
                
    # If no tasks are pending, the engine's job is done
    return "__end__"