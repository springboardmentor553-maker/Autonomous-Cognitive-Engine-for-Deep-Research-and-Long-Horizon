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
    Combines all executed step outputs into
    one structured final answer with confidence score.
    """

    print("\n" + "=" * 60)
    print("[SYNTHESIS NODE] Generating Final Consolidated Answer")
    print(f"Total Steps Executed: {state['execution_count']}")
    print("=" * 60)

    # ---- Read from Virtual File System (Milestone 2) ----
    files = state.get("files", {})

    file_contents = []

    if files:
        print("\n[VIRTUAL FILE SYSTEM] Reading stored step outputs")

        for filename, content in files.items():
            # tool call (LangSmith trace)
            read_file.invoke({
                "filename": filename, 
                "state": state
            })  
            print(f"[READ FILE] {filename}")
            file_contents.append(content)

    else:
        print("[WARNING] No files found in virtual file system")

    combined_text = "\n\n".join(file_contents)
    # ----------------------------------------------------

    response = llm.invoke(
    f"""
You are an expert technical writer synthesizing the final result of a multi-step autonomous agent execution.

Original Task:
{state['task']}

Executed Step Outputs:
{combined_text}

Create a **clean, structured report** using the following format:

----------------------------------------------------
TITLE
----------------------------------------------------

1. Overview
- Short explanation of the system/problem

2. Key Components / Steps
Summarize the 5 execution steps clearly.

3. Detailed Explanation
Explain the system in an organized way with headings.

4. Architecture / Workflow
Describe the process flow if relevant.

5. Key Techniques or Algorithms
List important methods used.

6. Evaluation / Improvements
Explain how the system can be improved.

----------------------------------------------------
End the report with a short **Conclusion**.

Formatting Rules:
• Use clear section headers  
• Use bullet points when possible  
• Avoid large text blocks  
• Keep the explanation concise but informative
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