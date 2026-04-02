"""
Multi-Agent Workflow - Ollama (llama3)
Optimised for speed — hardcoded delegation, no unnecessary LLM calls.
"""
from langgraph.graph import StateGraph, END
from typing import TypedDict, List, Annotated
import operator
from langchain_core.messages import BaseMessage

from brains.supervisor import create_supervisor
from brains.researcher import create_researcher
from brains.writer import create_writer
from brains.reviewer import create_reviewer

# Tool call tracking
tool_call_stats = {"web_search": 0, "write_file": 0}

def get_tool_call_stats():
    return tool_call_stats.copy()

def reset_tool_call_stats():
    global tool_call_stats
    tool_call_stats = {"web_search": 0, "write_file": 0}


class AgentState(TypedDict):
    messages:          Annotated[List[BaseMessage], operator.add]
    todos:             List[dict]
    current_step:      int
    completed_steps:   List[int]
    active_agent:      str
    next:              str
    created_files:     List[str]
    pending_files:     List[str]
    researcher_status: str
    writer_status:     str
    reviewer_status:   str
    user_task:         str
    final_output:      str


def create_multi_agent_workflow():
    """Create optimised multi-agent workflow."""

    supervisor = create_supervisor()
    researcher = create_researcher()
    writer     = create_writer()
    reviewer   = create_reviewer()

    def supervisor_node(state):
        print(f"\n{'='*50}\nSUPERVISOR - Step {state.get('current_step', 1)}\n{'='*50}")
        return supervisor(state)

    def researcher_node(state):
        print(f"\n[RESEARCHER] Working on step {state.get('current_step', 1)}")
        state["researcher_status"] = "working"
        state["active_agent"]      = "researcher"
        result = researcher(state)
        current = state.get("current_step", 1)
        if current < 3:
            result["current_step"] = current + 1
        result["researcher_status"] = "idle"
        result["next"]              = "supervisor"
        return result

    def writer_node(state):
        print("\n[WRITER] Creating report")
        state["writer_status"] = "working"
        state["active_agent"]  = "writer"
        result = writer(state)
        result["current_step"]  = 5
        result["writer_status"] = "idle"
        result["next"]          = "supervisor"
        return result

    def reviewer_node(state):
        print("\n[REVIEWER] Final review")
        state["reviewer_status"] = "working"
        state["active_agent"]    = "reviewer"
        result = reviewer(state)
        result["completed_steps"]  = [1, 2, 3, 4, 5]
        result["reviewer_status"]  = "idle"
        result["next"]             = "FINISH"
        return result

    def route_supervisor(state):
        next_agent = state.get("next", "FINISH")
        print(f"[ROUTER] Next: {next_agent}")
        return END if next_agent == "FINISH" else next_agent

    workflow = StateGraph(AgentState)
    workflow.add_node("supervisor", supervisor_node)
    workflow.add_node("researcher", researcher_node)
    workflow.add_node("writer",     writer_node)
    workflow.add_node("reviewer",   reviewer_node)

    workflow.set_entry_point("supervisor")
    workflow.add_conditional_edges(
        "supervisor", route_supervisor,
        {"researcher": "researcher", "writer": "writer", "reviewer": "reviewer", END: END}
    )
    workflow.add_edge("researcher", "supervisor")
    workflow.add_edge("writer",     "supervisor")
    workflow.add_edge("reviewer",   END)

    return workflow.compile()
