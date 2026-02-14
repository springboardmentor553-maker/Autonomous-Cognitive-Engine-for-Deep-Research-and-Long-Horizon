from tools.planning import write_todos

if __name__ == "__main__":
    goal = "Build a basic autonomous cognitive agent"
    todos = write_todos(goal)

    print("\nGenerated TODOs:\n")
    print(todos)
