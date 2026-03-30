from __future__ import annotations

from typing import Any, TypedDict

from app.config import MAX_GRAPH_ITERATIONS


class TodoItem(TypedDict, total=False):
    id: str
    task: str
    status: str
    assigned_agent: str
    result_file: str
    notes: str
    error: str


class EvaluationResult(TypedDict, total=False):
    score: int
    passed: bool
    strengths: list[str]
    weaknesses: list[str]
    improvements: list[str]
    summary: str


class GraphState(TypedDict, total=False):
    user_request: str
    thread_id: str
    messages: list[dict[str, str]]
    todos: list[TodoItem]
    files: dict[str, str]
    delegation_log: list[dict[str, str]]
    final_report: str
    evaluation: EvaluationResult
    queued_followups: list[str]
    last_error: str
    needs_replan: bool
    iteration: int
    max_iterations: int
    benchmark: dict[str, Any]


def create_todo(task: str, index: int) -> TodoItem:
    return {
        "id": f"todo-{index}",
        "task": task,
        "status": "pending",
        "assigned_agent": "unassigned",
        "result_file": "",
        "notes": "",
        "error": "",
    }


def create_initial_state(user_request: str, thread_id: str) -> GraphState:
    return {
        "user_request": user_request,
        "thread_id": thread_id,
        "messages": [{"role": "user", "content": user_request}],
        "todos": [],
        "files": {},
        "delegation_log": [],
        "final_report": "",
        "evaluation": {
            "score": 0,
            "passed": False,
            "strengths": [],
            "weaknesses": [],
            "improvements": [],
            "summary": "",
        },
        "queued_followups": [],
        "last_error": "",
        "needs_replan": False,
        "iteration": 0,
        "max_iterations": MAX_GRAPH_ITERATIONS,
        "benchmark": {},
    }
