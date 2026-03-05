from langchain_google_genai import ChatGoogleGenerativeAI
from .execution_state import ExecutionState
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
    Executes one step from the todo list.
    Includes retry logic and execution tracking.
    """

    if state["current_step"] >= len(state["todos"]):
        return state

    step_index = state["current_step"]
    step_task = state["todos"][step_index]["task"]

    print("\n" + "-" * 60)
    print(f"[EXECUTOR NODE] Executing Step {step_index + 1}")
    print(f"Task: {step_task}")
    print("-" * 60)

    retries = 0
    start_time = time.time()

    while retries <= MAX_RETRIES:
        try:
            response = llm.invoke(step_task)
            output = response.content

            # Quality gate
            if len(output.split()) < 20:
                output += "\n\n[Notice: Output may be too brief]"

            state["step_outputs"].append(output)
            state["execution_count"] += 1
            state["current_step"] += 1

            end_time = time.time()
            print(f"[STEP COMPLETED] Time Taken: {round(end_time - start_time, 2)} sec")

            return state

        except Exception as e:
            retries += 1
            print(f"[RETRY {retries}] Error: {e}")

    # If all retries fail
    state["step_outputs"].append("Execution failed after retries.")
    state["current_step"] += 1
    state["execution_count"] += 1

    return state