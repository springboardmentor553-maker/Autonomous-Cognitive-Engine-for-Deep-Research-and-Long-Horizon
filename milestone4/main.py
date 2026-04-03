from state import state
from tools.write_todos import write_todos
from tools.file_tools import write_file
from engine.execution import execution_loop
from engine.synthesis import synthesize_results


def run():

    user_input = input("Enter tasks (comma separated): ").strip()

    if not user_input:
        print("⚠️ No input provided.")
        return

    tasks = [t.strip() for t in user_input.split(",") if t.strip()]

    # STEP 1 — Planning
    state["todos"] = []
    state["trace"] = []

    for task_input in tasks:
        todos = write_todos(task_input)
        state["todos"].extend(todos)

    print("\n📋 TODO LIST:")
    for t in state["todos"]:
        print(f"- {t['task']} ({t['type']})")

    # STEP 2 — Execution
    execution_loop(state)

    # STEP 3 — Synthesis
    final_output = synthesize_results(state)

    print("\n📊 FINAL OUTPUT:\n")
    print(final_output)

    # ✅ Save final report (IMPORTANT for evaluation)
    write_file(state, "final_report.txt", final_output)

    # ✅ Optional: save trace log
    write_file(state, "execution_trace.txt", "\n".join(state["trace"]))


if __name__ == "__main__":
    run()