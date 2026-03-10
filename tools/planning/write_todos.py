"""
tools/planning/write_todos.py — Task decomposition tool.

The agent calls write_todos to break a high-level goal into concrete sub-tasks
stored in the LangGraph state.  Each call REPLACES the entire TODO list so the
agent can re-plan at any point.
"""

from __future__ import annotations

import json
import uuid
from typing import Any

from langchain_core.tools import tool

from utils.logger import get_logger

logger = get_logger(__name__)


@tool
def write_todos(tasks: list[str]) -> str:
    """
    Decompose the current goal into an ordered list of sub-tasks.

    Call this FIRST before any other tool to create a structured execution plan.
    Each task should be a clear, actionable step.  The agent will work through
    tasks sequentially, marking each completed before moving to the next.

    Args:
        tasks: List of task descriptions in execution order.
               Example: ["Search for X", "Summarise findings", "Write report"]

    Returns:
        JSON string confirming the TODO list was created.
    """
    todo_items = []
    for i, description in enumerate(tasks):
        todo_items.append(
            {
                "id": f"task_{i + 1}",
                "description": description,
                "status": "pending",
                "result": "",
            }
        )

    logger.info(f"Created {len(todo_items)} TODO tasks")

    return json.dumps(
        {
            "status": "todos_created",
            "count": len(todo_items),
            "todos": todo_items,
        },
        indent=2,
    )


# ─── Helper used by graph nodes (not a LangChain tool) ────────────────────────

def parse_todos_from_tool_result(raw: str) -> list[dict[str, Any]]:
    """Extract the todos list from the JSON string returned by write_todos."""
    data = json.loads(raw)
    return data.get("todos", [])
