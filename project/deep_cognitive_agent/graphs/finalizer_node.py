from langchain_google_genai import ChatGoogleGenerativeAI
from .execution_state import ExecutionState
import os

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0,
    google_api_key=os.getenv("GOOGLE_API_KEY"),
)


def finalize_answer(state: ExecutionState) -> ExecutionState:
    """
    Combines all executed step outputs into
    one structured final answer with confidence score.
    """

    print("\n" + "=" * 60)
    print("[SYNTHESIS NODE] Generating Final Consolidated Answer")
    print(f"Total Steps Executed: {state['execution_count']}")
    print("=" * 60)

    combined_text = "\n\n".join(state["step_outputs"])

    response = llm.invoke(
        f"""
You are synthesizing the final result of a multi-step execution.

Original Task:
{state['task']}

Executed Step Outputs:
{combined_text}

Generate a structured, professional final response.
Use headings and clean formatting.
"""
    )

    final_answer = response.content

    # Confidence scoring
    word_count = len(final_answer.split())
    confidence = "High" if word_count > 300 else "Moderate"

    final_answer += f"\n\n---\nConfidence Level: {confidence}"

    state["final_answer"] = final_answer

    print("\n[SYNTHESIS COMPLETE]\n")

    return state