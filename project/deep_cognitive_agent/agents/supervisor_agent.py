"""
Supervisor Agent - Orchestrates multi-agent collaboration.

Milestone 2: Wraps plan → execute → synthesize pipeline.
Milestone 3: Delegates execution to specialized sub-agents via registry.
"""

from typing import Dict
from langchain_core.messages import HumanMessage

from graphs.main_graph import build_graph
from graphs.main_graph_m3 import build_graph_m3
from registry.subagent_registry import registry, build_registry


class SupervisorAgent:
    """High-level agent that manages plan → delegate → synthesize."""

    def __init__(self, llm, milestone: int = 3):
        self.llm = llm
        self.milestone = milestone
        if milestone >= 3:
            self.registry = build_registry()
            self.graph = build_graph_m3(llm, self.registry)
        else:
            self.registry = None
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
            "trace_log": [],
        }
        return self.graph.invoke(initial_state)
