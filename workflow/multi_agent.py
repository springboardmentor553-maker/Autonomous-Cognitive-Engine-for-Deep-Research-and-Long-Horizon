
"""
Multi-Agent State Management
Shared state across all agents in the workflow
"""
from typing import TypedDict, Annotated, List, Optional
from langgraph.graph.message import add_messages

class MultiAgentState(TypedDict):
    """
    Shared state for multi-agent workflow.
    All agents can read and update this state.
    """
    # Messages (conversation history)
    messages: Annotated[list, add_messages]
    
    # Planning (from Milestone 1)
    todos: List[dict]
    
    # Workflow tracking
    current_step: int
    completed_steps: List[int]
    active_agent: Optional[str]
    
    # File tracking
    created_files: List[str]
    pending_files: List[str]
    
    # Agent status
    researcher_status: str  # "idle" | "working" | "complete"
    writer_status: str
    reviewer_status: str
    
    # Final output
    final_output: Optional[str]