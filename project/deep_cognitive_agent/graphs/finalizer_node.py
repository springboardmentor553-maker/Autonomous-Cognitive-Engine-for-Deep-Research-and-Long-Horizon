from langchain_google_genai import ChatGoogleGenerativeAI
from graphs.execution_state import ExecutionState
import os

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0,
    google_api_key=os.getenv("GOOGLE_API_KEY"),
)


def finalize_answer(state: ExecutionState) -> ExecutionState:
    """
    Combines all executed step outputs into
    one structured final answer.
    """

    combined_text = "\n\n".join(state["step_outputs"])

    response = llm.invoke(
        f"""
You are synthesizing the final result of a multi-step execution.

Original Task:
{state['task']}

Executed Step Outputs:
{combined_text}

Generate a structured, professional final response.
"""
    )

    state["final_answer"] = response.content

    return state