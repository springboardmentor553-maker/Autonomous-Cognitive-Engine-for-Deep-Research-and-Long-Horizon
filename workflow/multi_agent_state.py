"""
Multi-Agent State Management
Shared state across all agents in the workflow
"""
from typing import TypedDict, Annotated, List, Dict, Any
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage


class MultiAgentState(TypedDict):
    """
    Shared state across all agents in multi-agent workflow.
    
    This state is accessible by:
    - Supervisor (orchestrator)
    - Researcher (data gathering)
    - Writer (content creation)
    - Reviewer (quality assurance)
    """
    
    # Messages (conversation history)
    messages: Annotated[list[BaseMessage], add_messages]
    
    # Planning (from Milestone 1)
    todos: List[Dict[str, Any]]
    
    # Workflow tracking
    current_step: int
    completed_steps: List[int]
    active_agent: str
    
    # File tracking
    created_files: List[str]
    pending_files: List[str]
    
    # Agent status
    researcher_status: str   # "idle" | "working" | "complete"
    writer_status: str       # "idle" | "working" | "complete"
    reviewer_status: str     # "idle" | "working" | "complete"
    
    # Task context
    user_task: str
    final_output: str