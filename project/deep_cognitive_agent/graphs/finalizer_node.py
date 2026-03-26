from langchain_google_genai import ChatGoogleGenerativeAI
from .execution_state import ExecutionState

from project.deep_cognitive_agent.tools.filesystem.read_file import read_file

import os


llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0,
    google_api_key=os.getenv("GOOGLE_API_KEY"),
)


def finalize_answer(state: ExecutionState) -> ExecutionState:
    """
    Milestone 4 Finalizer

    Responsibilities:
    - Retrieve stored results (memory)
    - Combine all data
    - Generate structured final report
    """

    print("\n" + "=" * 60)
    print("[SYNTHESIS NODE] Generating Final Consolidated Answer")
    print(f"Total Steps Executed: {state['execution_count']}")
    print("=" * 60)

    # ==================================================
    # READ FROM VIRTUAL FILE SYSTEM
    # ==================================================

    files = state.get("files", {})
    combined_data = ""

    if files:
        print("\n[VIRTUAL FILE SYSTEM] Reading stored step outputs")

        for filename in files:

            # Tool call (LangSmith trace)
            read_file.invoke({
                "filename": filename,
                "state": state
            })

            print(f"[READ FILE] {filename}")

            combined_data += f"\n\n--- {filename} ---\n"
            combined_data += files[filename]

    else:
        print("[WARNING] No files found in virtual file system")

    # ==================================================
    # FINAL SYNTHESIS
    # ==================================================

    response = llm.invoke(
        f"""
You are an expert AI system generating a final report.

Original Task:
{state['task']}

Collected Data from Execution:
{combined_data}

Generate a clean, structured report with:

1. Title
2. Overview
3. Key Steps Performed
4. Detailed Explanation
5. Final Conclusion

Rules:
- Keep it clear and readable
- Avoid unnecessary symbols (*, ###)
- Use simple structured formatting
- Make it presentation-ready
"""
    )

    final_answer = response.content

    # ==================================================
    # CONFIDENCE SCORE
    # ==================================================

    word_count = len(final_answer.split())
    confidence = "High" if word_count > 300 else "Moderate"

    final_answer += f"\n\n--------------------------------\nConfidence Level: {confidence}\n--------------------------------"

    # ==================================================
    # SAVE FINAL ANSWER
    # ==================================================

    state["final_answer"] = final_answer

    print("\n[SYNTHESIS COMPLETE]\n")

    return state