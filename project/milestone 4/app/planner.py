from __future__ import annotations

from app.models import LLMClient
from app.state import TodoItem, create_todo


PLANNER_SYSTEM_PROMPT = """
You are the planning node in a supervisor-style autonomous research system.
Break a user request into short execution-ready TODOs.
Prefer 4 to 7 steps.
Return one task per line with no numbering.
Make the steps concrete enough to execute.
""".strip()


REPLANNER_SYSTEM_PROMPT = """
You are a replanning node.
Given a user request, current TODOs, and newly discovered follow-up work, add only the missing next tasks.
Return only new tasks, one per line, with no numbering.
Do not repeat existing completed tasks.
""".strip()


def _lines_to_todos(lines: str, start_index: int) -> list[TodoItem]:
    todos: list[TodoItem] = []
    next_index = start_index
    for raw_step in lines.splitlines():
        step = raw_step.strip("-* 1234567890. ").strip()
        if step:
            todos.append(create_todo(step, next_index))
            next_index += 1
    return todos


def write_todos(task_description: str) -> list[TodoItem]:
    llm = LLMClient()
    steps = llm.predict(task_description, system_prompt=PLANNER_SYSTEM_PROMPT)
    todos = _lines_to_todos(steps, 1)

    if not todos:
        todos = [
            create_todo(f"Research the request: {task_description}", 1),
            create_todo("Summarize the key findings", 2),
            create_todo("Synthesize a final report", 3),
            create_todo("Evaluate the quality of the report", 4),
        ]

    return todos


def replan_todos(
    user_request: str,
    existing_todos: list[TodoItem],
    follow_up_tasks: list[str],
) -> list[TodoItem]:
    llm = LLMClient()
    existing_text = "\n".join(
        f"- {todo['task']} [{todo['status']}]"
        for todo in existing_todos
    )
    followup_text = "\n".join(f"- {task}" for task in follow_up_tasks)

    prompt = (
        f"User request:\n{user_request}\n\n"
        f"Current TODOs:\n{existing_text}\n\n"
        f"Discovered follow-up tasks:\n{followup_text}"
    )
    steps = llm.predict(prompt, system_prompt=REPLANNER_SYSTEM_PROMPT)
    start_index = len(existing_todos) + 1
    new_todos = _lines_to_todos(steps, start_index)

    if not new_todos:
        new_todos = [create_todo(task, start_index + idx) for idx, task in enumerate(follow_up_tasks)]

    existing_labels = {todo["task"].strip().lower() for todo in existing_todos}
    filtered: list[TodoItem] = []
    for todo in new_todos:
        if todo["task"].strip().lower() not in existing_labels:
            filtered.append(todo)
            existing_labels.add(todo["task"].strip().lower())

    return filtered
