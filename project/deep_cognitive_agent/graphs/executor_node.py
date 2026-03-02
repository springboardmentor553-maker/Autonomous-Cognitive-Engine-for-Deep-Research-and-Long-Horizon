from langchain_google_genai import ChatGoogleGenerativeAI
from graphs.execution_state import ExecutionState
import os

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0,
    google_api_key=os.getenv("GOOGLE_API_KEY"),
)


def execute_step(state: ExecutionState) -> ExecutionState:
    """
    Executes one step from the TODO list.
    """

    todos = state["todos"]
    current_index = state["current_step"]

    # If all steps are completed, return state
    if current_index >= len(todos):
        return state

    step_text = todos[current_index]["task"]

    print(f"\nExecuting Step {current_index + 1}: {step_text}")

    response = llm.invoke(
        f"""
You are executing a step from a structured plan.

Step:
{step_text}

Provide a detailed and precise execution output.
"""
    )

    # Store output
    state["step_outputs"].append(response.content)

    # Move to next step
    state["current_step"] += 1

    return state