"""
LangGraph State Definition - Milestone 1 & 2.

Defines the AgentState TypedDict that holds messages, todos,
virtual file system, trace log, and final output.
"""

from typing import TypedDict, List, Dict, Annotated
from langgraph.graph.message import add_messages


class AgentState(TypedDict):
    """
    State schema for the Deep Cognitive Agent.

    Attributes:
        messages: List of conversation messages (managed by LangGraph)
        todos: Enriched todo items with step_type, output_file, depends_on
        files: Virtual file system - dict mapping filename → content
        current_step: Index of the step currently being considered
        final_output: The final synthesized output string
        quality_score: LLM-evaluated quality score (0-100) for final_output
        quality_target: Configured minimum target score for quality gate
        quality_passed: Whether final_output met/exceeded quality_target
        quality_report: Detailed evaluator feedback used for refinement
        trace_log: Ordered list of tool invocations for evaluation tracing
    """
    messages: Annotated[list, add_messages]
    todos: List[Dict]
    files: Dict[str, str]
    current_step: int | None
    final_output: str
    quality_score: int | None
    quality_target: int | None
    quality_passed: bool
    quality_report: Dict
    trace_log: List[Dict]
