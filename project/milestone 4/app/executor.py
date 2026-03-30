from __future__ import annotations

from agents.summarizer import SummarizerAgent
from agents.web_searcher import WebSearcherAgent
from app.models import LLMClient
from app.parsing import extract_json_object
from app.state import GraphState, TodoItem
from storage.file_store import write_file


ROUTER_PROMPT = """
You are a task router.
Choose exactly one handler for the task.
Return JSON with keys:
- agent: one of ["researcher", "summarizer", "general"]
- reason: short reason
""".strip()


FOLLOWUP_PROMPT = """
You are an execution reviewer.
Inspect the task result and decide whether more TODO items are needed.
Return JSON with keys:
- follow_up_tasks: list of strings
- notes: short note
""".strip()


def _get_next_pending_todo(state: GraphState) -> TodoItem | None:
    for todo in state["todos"]:
        if todo["status"] == "pending":
            return todo
    return None


def _route_task(task_text: str) -> dict:
    llm = LLMClient()
    raw = llm.predict(task_text, system_prompt=ROUTER_PROMPT)
    parsed = extract_json_object(raw)
    if parsed.get("agent") not in {"researcher", "summarizer", "general"}:
        return {"agent": "general", "reason": "Fallback routing"}
    return parsed


def _review_outcome(task_text: str, result_text: str) -> dict:
    llm = LLMClient()
    raw = llm.predict(
        f"TASK:\n{task_text}\n\nRESULT:\n{result_text}",
        system_prompt=FOLLOWUP_PROMPT,
    )
    parsed = extract_json_object(raw)
    followups = parsed.get("follow_up_tasks")
    if not isinstance(followups, list):
        followups = []
    return {
        "follow_up_tasks": [str(item).strip() for item in followups if str(item).strip()],
        "notes": str(parsed.get("notes", "")),
    }


def execute_next_todo(state: GraphState) -> GraphState:
    todo = _get_next_pending_todo(state)
    if not todo:
        return state

    todo["status"] = "in_progress"
    route = _route_task(todo["task"])
    agent_name = route["agent"]
    todo["assigned_agent"] = agent_name

    try:
        if agent_name == "researcher":
            result = WebSearcherAgent().run(todo["task"], state)
            filename = f"{todo['id']}_research.txt"
        elif agent_name == "summarizer":
            result = SummarizerAgent().run(todo["task"], state)
            filename = f"{todo['id']}_summary.txt"
        else:
            result = LLMClient().predict(
                todo["task"],
                system_prompt=(
                    "You are the general execution node in an autonomous research workflow. "
                    "Complete the task clearly and write a clean output."
                ),
            )
            filename = f"{todo['id']}_task.txt"

        write_file(filename, result, state)
        todo["result_file"] = filename
        todo["status"] = "done"

        review = _review_outcome(todo["task"], result)
        todo["notes"] = review["notes"]
        if review["follow_up_tasks"]:
            state["queued_followups"].extend(review["follow_up_tasks"])
            state["needs_replan"] = True

        state["delegation_log"].append(
            {"task": todo["task"], "agent": agent_name, "file": filename}
        )
        state["last_error"] = ""
    except Exception as exc:
        todo["status"] = "failed"
        todo["error"] = str(exc)
        state["last_error"] = str(exc)
        state["needs_replan"] = True

    return state
