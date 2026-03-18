"""
agent/planner_agent.py — High-level planner interface.

Provides a convenience wrapper around the planner prompt + LLM for
standalone use (e.g., testing task decomposition in isolation).
"""

from __future__ import annotations

from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage
from pathlib import Path

import config
from tools.planning.write_todos import write_todos
from utils.logger import get_logger
from utils.parser import safe_json_loads

logger = get_logger(__name__)

_PROMPT_PATH = Path(__file__).parent.parent / "prompts" / "planner_prompt.txt"


class PlannerAgent:
    """
    Standalone planner: decomposes a user request into a TODO list.
    Useful for unit-testing milestone 1 without running the full graph.
    """

    def __init__(self):
        config.validate_config()
        self._llm = ChatGroq(
            model=config.MODEL_NAME,
            groq_api_key=config.GROQ_API_KEY,
            temperature=0,
        ).bind_tools([write_todos])
        self._system = _PROMPT_PATH.read_text() if _PROMPT_PATH.exists() else ""

    def plan(self, request: str) -> list[dict]:
        """
        Decompose request into ordered TODO items.

        Returns:
            List of TodoItem dicts (id, description, status, result).
        """
        logger.info(f"PlannerAgent.plan: {request[:80]}…")

        messages = [
            SystemMessage(content=self._system),
            HumanMessage(content=f"User request:\n\n{request}"),
        ]

        response = self._llm.invoke(messages)

        # Execute the write_todos tool call
        for tc in getattr(response, "tool_calls", []):
            if tc["name"] == "write_todos":
                result = write_todos.invoke(tc["args"])
                parsed = safe_json_loads(result)
                if parsed:
                    return parsed.get("todos", [])

        logger.warning("PlannerAgent: write_todos was not called")
        return []