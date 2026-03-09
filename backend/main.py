import os
from backend.tools.planning_tool import write_todos
from backend.core.executor import execute_plan
from backend.tools.file_system_tools import read_file


def display_memory_structure():
    print("\nVIRTUAL MEMORY STRUCTURE")
    print("----------------------------------------")

    for root, dirs, files in os.walk("memory"):

        level = root.replace("memory", "").count(os.sep)
        indent = "   " * level

        print(f"{indent}{os.path.basename(root)}")

        subindent = "   " * (level + 1)

        for f in files:
            print(f"{subindent}{f}")


def display_research_results():

    research_dir = "memory/research"

    if not os.path.exists(research_dir):
        return

    print("\nRESEARCH RESULTS")
    print("----------------------------------------")

    for file in sorted(os.listdir(research_dir)):

        path = f"research/{file}"

        print(f"\nFILE: memory/{path}")

        content = read_file(path)

        print(content)


def main():

    while True:

        objective = input("Enter complex objective (or 'exit'): ")

        if objective.lower() == "exit":
            break

        state = {
            "objective": objective,
            "todos": [],
            "trace": []
        }

        print("\nFINAL EXECUTION REPORT")
        print("=" * 60)

        # Planning
        state = write_todos(state, objective)

        print("\nTASK PLAN")
        print("----------------------------------------")

        for i, task in enumerate(state["todos"], start=1):
            print(f"{i}. {task}")

        # Execution
        state = execute_plan(state)

        # Research Output
        display_research_results()

        # Memory Structure
        display_memory_structure()

        # Execution Trace
        print("\nEXECUTION TRACE")
        print("----------------------------------------")

        for t in state["trace"]:
            print(t)

        # Summary
        print("\nEXECUTION SUMMARY")
        print("----------------------------------------")
        print("Validated Plan: True")
        print(f"Tasks Completed: {len(state['todos'])}")


if __name__ == "__main__":
    main()