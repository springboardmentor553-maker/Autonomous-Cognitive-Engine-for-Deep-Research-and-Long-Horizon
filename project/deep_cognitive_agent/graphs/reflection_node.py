from langchain_google_genai import ChatGoogleGenerativeAI
from .execution_state import ExecutionState
import os

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0,
    google_api_key=os.getenv("GOOGLE_API_KEY"),
)


def reflect_on_step(state: ExecutionState) -> ExecutionState:
    """
    Reflect on the most recently executed step.
    """

    if not state["step_outputs"]:
        return state

    latest_output = state["step_outputs"][-1]
    latest_step_index = state["current_step"] - 1
    latest_step = state["todos"][latest_step_index]["task"]

    response = llm.invoke(
        f"""
You are reviewing the execution quality of a step.

Step:
{latest_step}

Execution Output:
{latest_output}

Evaluate:
- Is the output complete?
- Is it logically consistent?
- Is anything missing?

Respond briefly with evaluation and suggestions.
"""
    )

    state["reflection_notes"].append(response.content)

    return state