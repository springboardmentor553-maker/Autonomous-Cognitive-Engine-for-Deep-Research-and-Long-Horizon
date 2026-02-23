"""
Supervisor Agent - Orchestrates the planning and execution pipeline.

This module wraps the LangGraph StateGraph into a reusable
SupervisorAgent class for Milestone 2+.
"""

from typing import Dict, Optional
from langchain_core.messages import HumanMessage

from graphs.main_graph import build_graph


class SupervisorAgent:
    """High-level agent that manages plan → execute → synthesize."""

    def __init__(self, llm):
        self.llm = llm
        self.graph = build_graph(llm)

    def run(self, task: str) -> Dict:
        """Run the full pipeline on a task.

        Args:
            task: Natural-language task description.

        Returns:
            Final AgentState dict with todos, files, final_output.
        """
        initial_state = {
            "messages": [HumanMessage(content=task)],
            "todos": [],
            "files": {},
            "final_output": "",
            "current_step": None,
        }
        return self.graph.invoke(initial_state)
