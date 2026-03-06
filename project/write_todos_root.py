import os

def write_todos_tool(todos: str):
    """Writes the generated TODOs to a file."""
    # Ensure the directory exists
    os.makedirs("test_results", exist_ok=True)
    file_path = os.path.join("test_results", "todos.txt")
    
    # Append the todos to the file
    with open(file_path, "a", encoding="utf-8") as f:
        f.write(todos + "\n---\n")
    return f"Successfully wrote to {file_path}"
