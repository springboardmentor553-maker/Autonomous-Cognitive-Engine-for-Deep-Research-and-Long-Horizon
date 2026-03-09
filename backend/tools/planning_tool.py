import json
from backend.tools.file_system_tools import write_file


def write_todos(state, objective):

    tasks = [
        "Research AI Applications",
        "Review Medical Imaging AI",
        "Examine Chatbots in Healthcare",
        "Study Predictive Analytics",
        "Analyze Clinical Decision Support",
        "Document Findings and Insights"
    ]

    state["todos"] = tasks

    plan = {
        "objective": objective,
        "tasks": tasks
    }

    write_file("planning/task_plan.json", json.dumps(plan, indent=2))

    return state
