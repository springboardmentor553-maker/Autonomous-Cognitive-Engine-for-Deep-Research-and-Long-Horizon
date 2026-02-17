from typing import Dict
from backend.tools.planning_tool import write_todos


def planning_node(state: Dict) -> Dict:

    todos = write_todos(state["user_request"])

    return {
        "todos": todos,
        "planning_meta": {
            "total_tasks": len(todos),
            "retry_count": state.get("planning_meta", {}).get("retry_count", 0) + 1,
            "validated": False,
            "validation_errors": []
        }
    }


def validation_node(state: Dict) -> Dict:

    errors = []
    todos = state["todos"]

    if not (5 <= len(todos) <= 8):
        errors.append("Invalid number of tasks.")

    for idx, task in enumerate(todos, start=1):
        if task["id"] != idx:
            errors.append("Task IDs not sequential.")
        if len(task["description"].strip()) < 20:
            errors.append("Description too short.")

    state["planning_meta"]["validation_errors"] = errors
    state["planning_meta"]["validated"] = len(errors) == 0

    return {
        "planning_meta": state["planning_meta"]
    }


def synthesis_node(state: Dict) -> Dict:

    output = "\nSTRUCTURED RESEARCH PLAN\n"
    output += "=" * 60 + "\n\n"

    for task in state["todos"]:
        output += f"{task['id']}. {task['title']}\n"
        output += f"{task['description']}\n\n"

    output += "=" * 60 + "\n"
    output += f"Validated: {state['planning_meta']['validated']}\n"
    output += f"Retries: {state['planning_meta']['retry_count']}\n"

    return {
        "final_output": output
    }