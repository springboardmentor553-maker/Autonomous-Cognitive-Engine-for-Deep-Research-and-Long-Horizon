import json

TODO_FILE = "generated_todos.json"

def write_todos(task: str):
    """
    Takes a complex task description and converts it into structured TODOs
    """
    todos = [
        {"id": 1, "task": "Understand the problem statement", "status": "pending"},
        {"id": 2, "task": "Collect required information", "status": "pending"},
        {"id": 3, "task": "Analyze gathered information", "status": "pending"},
        {"id": 4, "task": "Prepare final output", "status": "pending"}
    ]

    with open(TODO_FILE, "w") as f:
        json.dump(
            {
                "original_task": task,
                "todos": todos
            },
            f,
            indent=4
        )

    return todos