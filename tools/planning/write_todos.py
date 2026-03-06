import os
import json

def write_todos_tool(todos):
    """Writes the generated TODOs to a file."""
    os.makedirs("test_results", exist_ok=True)
    file_path = os.path.join("test_results", "todos.txt")
    with open(file_path, "a", encoding="utf-8") as f:
        f.write(str(todos) + "\n---\n")
    return f"Successfully wrote to {file_path}"