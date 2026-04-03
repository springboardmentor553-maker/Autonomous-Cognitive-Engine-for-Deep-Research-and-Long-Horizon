from tools.task import task
from tools.file_tools import write_file
from utils.llm import llm
import time

def execution_loop(state):

    max_tasks = len(state["todos"])   # ✅ run all tasks
    count = 0

    while any(todo["status"] == "pending" for todo in state["todos"]):

        for todo in state["todos"]:

            if todo["status"] != "pending":
                continue

            # 🔴 Stop if all tasks done
            if count >= max_tasks:
                print("\n✅ All tasks executed")
                state["trace"].append("All tasks executed")
                return

            task_text = todo.get("task", "").strip()
            task_type = todo.get("type", "general").lower()

            # 🔴 Skip empty tasks
            if not task_text:
                todo["status"] = "done"
                continue

            # 🔴 Skip overly long tasks
            if len(task_text) > 120:
                state["trace"].append(f"Skipped long task: {task_text}")
                todo["status"] = "done"
                continue

            print(f"\nExecuting: {task_text} ({task_type})")
            state["trace"].append(f"Executing: {task_text} [{task_type}]")

            try:
                # ✅ TYPE-BASED EXECUTION

                if task_type == "summarize":
                    result = task("summarizer", task_text)
                    state["trace"].append("Used summarizer agent")

                elif task_type == "research":
                    response = llm.invoke(
                        f"Short factual info (3-4 lines): {task_text}"
                    )
                    result = response.content

                else:  # general
                    response = llm.invoke(
                        f"Give a short and clear answer: {task_text}"
                    )
                    result = response.content

                # 🔒 Safe filename
                safe_name = "".join(c for c in task_text if c.isalnum() or c in (" ", "_"))
                filename = safe_name.replace(" ", "_")[:40] + f"_{count}.txt"

                write_file(state, filename, result)

                todo["status"] = "done"
                count += 1

                # ⏳ Slight delay to avoid rate limit
                time.sleep(2)

            except Exception as e:
                print("Error:", e)
                state["trace"].append(f"Error: {str(e)}")

                # ❗ Mark as done to avoid infinite loop
                todo["status"] = "done"

                # ⏳ Backoff delay
                time.sleep(2)