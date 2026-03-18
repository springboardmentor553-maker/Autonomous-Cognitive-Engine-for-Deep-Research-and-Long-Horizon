from backend.tools.delegation_tool import task_tool
from backend.tools.file_system_tools import write_file, read_file
from backend.tools.planning_tool import write_todos


def run_agent(objective):

    # -------------------------------
    # STEP 1: PLANNING
    # -------------------------------
    todos = write_todos(objective)

    # -------------------------------
    # STEP 2: EXECUTION
    # -------------------------------
    for i, task in enumerate(todos, 1):

        task_name = task["task"]

        print(f"\n--- Task {i}: {task_name} ---\n")

        # -------------------------------
        # FINAL SUMMARY HANDLING
        # -------------------------------
        if "Final Summary" in task_name:

            print("\nGenerating Final Insights Across All Tasks...\n")

            print("TASK TOOL → Evaluating task: final summary")

            all_summaries = ""

            for j in range(1, i):
                all_summaries += read_file(f"summary_{j}.txt") + "\n"

            final_summary = task_tool("summary", "FINAL " + all_summaries)

            write_file(f"summary_{i}.txt", final_summary)

            continue

        # -------------------------------
        # STEP 2.1 → RESEARCH
        # -------------------------------
        print("TASK TOOL → Evaluating task: research")
        research = task_tool("research", task_name)

        write_file(f"research_{i}.txt", research)

        # -------------------------------
        # STEP 2.2 → ANALYSIS
        # -------------------------------
        print("TASK TOOL → Evaluating task: analysis")
        analysis = task_tool("analysis", research)

        write_file(f"analysis_{i}.txt", analysis)

        # -------------------------------
        # STEP 2.3 → SUMMARY
        # -------------------------------
        print("TASK TOOL → Evaluating task: summary")
        summary = task_tool("summary", analysis)

        write_file(f"summary_{i}.txt", summary)

    # -------------------------------
    # STEP 3: FINAL REPORT
    # -------------------------------
    print("\n\nFINAL EXECUTION REPORT")
    print("=" * 60)

    # -------------------------------
    # TASK PLAN
    # -------------------------------
    print("\nTASK PLAN:\n")

    for i, task in enumerate(todos, 1):
        print(f"{i}. {task['task']}")
        print("   → This task explores an important AI application in healthcare.\n")

    # -------------------------------
    # DETAILED RESULTS
    # -------------------------------
    print("\nDETAILED RESULTS\n")

    for i in range(1, len(todos) + 1):

        print(f"\n========== TASK {i} ==========\n")

        # FINAL SUMMARY → show only summary
        if i == len(todos):

            print(f"--- summary_{i}.txt ---\n")
            print(read_file(f"summary_{i}.txt"))

            break

        # NORMAL TASKS
        print(f"--- research_{i}.txt ---\n")
        print(read_file(f"research_{i}.txt"))

        print(f"\n--- analysis_{i}.txt ---\n")
        print(read_file(f"analysis_{i}.txt"))

        print(f"\n--- summary_{i}.txt ---\n")
        print(read_file(f"summary_{i}.txt"))

    print("\n--- Execution Completed Successfully ---\n")