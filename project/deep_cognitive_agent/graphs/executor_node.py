from langchain_google_genai import ChatGoogleGenerativeAI
from .execution_state import ExecutionState

# Delegation tool
from project.deep_cognitive_agent.tools.delegation.delegate_task import delegate_task

# File system tool
from project.deep_cognitive_agent.tools.filesystem.write_file import write_file

import os
import time


llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0,
    google_api_key=os.getenv("GOOGLE_API_KEY"),
)

MAX_RETRIES = 2


def execute_step(state: ExecutionState) -> ExecutionState:
    """
    Milestone 4 Executor (Supervisor Brain)

    Responsibilities:
    - Decide execution vs delegation
    - Store outputs in memory + file system
    - Track delegation logs
    """

    if state["current_step"] >= len(state["todos"]):
        return state

    step_index = state["current_step"]
    step_task = state["todos"][step_index]["task"]

    # Ensure status exists
    if "status" not in state["todos"][step_index]:
        state["todos"][step_index]["status"] = "pending"

    print("\n" + "-" * 60)
    print(f"[EXECUTOR NODE] Executing Step {step_index + 1}")
    print(f"Task: {step_task}")
    print("-" * 60)

    retries = 0
    start_time = time.time()

    while retries <= MAX_RETRIES:
        try:

            # ==================================================
            # SMART SUPERVISOR DECISION LOGIC (FINAL FIX)
            # ==================================================

            task_lower = step_task.lower()

            # Delegate if task is complex OR requires explanation/summarization
            if any(word in task_lower for word in [
                "summarize", "summary", "analyze", "overview",
                "conclude", "explain", "outline", "describe"
            ]) or len(step_task.split()) > 8:

                print("[SUPERVISOR] Delegating to Summarization Agent")

                output = delegate_task.invoke({
                    "agent_name": "summarizer",
                    "input_data": step_task
                })

                # Track delegation
                state["delegation_log"].append({
                    "step": step_index + 1,
                    "task": step_task,
                    "agent": "summarizer"
                })

            else:
                print("[SUPERVISOR] Executing task directly")

                response = llm.invoke(step_task)
                output = response.content

            # ==================================================
            # QUALITY CHECK
            # ==================================================

            if len(str(output).split()) < 20:
                output += "\n\n[Notice: Output may be too brief]"

            # ==================================================
            # STORE OUTPUT
            # ==================================================

            state["step_outputs"].append(str(output))

            if "files" not in state:
                state["files"] = {}

            filename = f"step_{step_index + 1}.txt"

            # Save in virtual memory
            state["files"][filename] = str(output)

            # Save via tool (for LangSmith trace)
            write_file.invoke({
                "filename": filename,
                "content": output,
                "state": state
            })

            print(f"[FILE SAVED] {filename}")

            # ==================================================
            # UPDATE STATE
            # ==================================================

            state["execution_count"] += 1
            state["current_step"] += 1
            state["todos"][step_index]["status"] = "done"

            end_time = time.time()
            print(f"[STEP COMPLETED] Time Taken: {round(end_time - start_time, 2)} sec")

            return state

        except Exception as e:
            retries += 1
            print(f"[RETRY {retries}] Error: {e}")

    # ==================================================
    # FAILURE FALLBACK
    # ==================================================

    state["step_outputs"].append("Execution failed after retries.")
    state["todos"][step_index]["status"] = "done"
    state["current_step"] += 1
    state["execution_count"] += 1

    return state