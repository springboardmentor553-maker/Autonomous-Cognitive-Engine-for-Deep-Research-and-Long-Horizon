import os
import ast
import json
import time
from typing import TypedDict, List, Dict, Any
from dotenv import load_dotenv

load_dotenv()

from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver

from project.deep_cognitive_agent.tools.planning.write_todos import write_todos
from project.deep_cognitive_agent.graphs.execution_graph import build_execution_graph
from project.deep_cognitive_agent.graphs.execution_state import ExecutionState

from project.deep_cognitive_agent.tools.filesystem.write_file import write_file
from project.deep_cognitive_agent.tools.filesystem.read_file import read_file
from project.deep_cognitive_agent.tools.filesystem.edit_file import edit_file
from project.deep_cognitive_agent.tools.filesystem.ls import ls


# -------------------------
# LLM (Gemini)
# -------------------------
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0,
    google_api_key=os.getenv("GOOGLE_API_KEY"),
    max_output_tokens=800,
)

# Bind tools
llm_with_tools = llm.bind_tools([
    write_todos,
    write_file,
    read_file,
    edit_file,
    ls
])


# -------------------------
# System Prompt
# -------------------------
SYSTEM_PROMPT = """
You are a supervisor agent.

Rules:
1. Always create TODO plans first.
2. Delegate specialized tasks when required.
3. Store large outputs using write_file.
4. Retrieve results using read_file.
5. Integrate everything before final answer.
"""


# -------------------------
# Agent Factory
# -------------------------
def create_planning_agent():
    memory = MemorySaver()

    agent = create_react_agent(
        model=llm_with_tools,
        tools=[
            write_todos,
            write_file,
            read_file,
            edit_file,
            ls
        ],
        checkpointer=memory,
        prompt=SYSTEM_PROMPT,
    )

    return agent


# -------------------------
# Run Agent
# -------------------------
def run_agent(agent, task: str, thread_id: str = "default") -> Dict:
    config = {"configurable": {"thread_id": thread_id}}
    input_message = {"messages": [("user", task)]}

    final_state = agent.invoke(input_message, config)

    todos = []

    for msg in final_state.get("messages", []):
        if getattr(msg, "type", None) == "tool" and getattr(msg, "name", None) == "write_todos":
            
            content = msg.content

            if isinstance(content, list):
                todos = content
            elif isinstance(content, str):
                try:
                    todos = ast.literal_eval(content)
                except Exception:
                    todos = []

    return {
        "task": task,
        "messages": final_state.get("messages", []),
        "todos": todos
    }


# -------------------------
# Full Cognitive Flow 
# -------------------------
class ExecutionState(TypedDict):
    task: str
    todos: List[Dict[str, Any]]
    current_step: int
    execution_count: int
    step_outputs: List[Any]
    reflection_notes: List[str]
    files: Dict[str, str]
    delegation_log: List[Dict[str, Any]]
    final_answer: str

def run_full_agent(task: str):

    start_time = time.time()

    planner_agent = create_planning_agent()
    planning_result = run_agent(planner_agent, task)

    todos = planning_result["todos"]

    if not todos:
        print("⚠️ No todos generated. Execution aborted.")
        return None

    execution_graph = build_execution_graph()

    initial_state: ExecutionState = {
        "task": task,
        "todos": todos,
        "current_step": 0,
        "execution_count": 0,
        "step_outputs": [],
        "reflection_notes": [],
        "files": {},
        "delegation_log": [],
        "final_answer": "",
    }

    final_state = execution_graph.invoke(initial_state)

    end_time = time.time()
    print(f"\n[PIPELINE COMPLETED] Total Time: {round(end_time - start_time, 2)} seconds")

    return final_state


# -------------------------
# Save Output
# -------------------------
def save_result_to_json(result: Dict, filename: str, output_dir: str = "outputs"):
    os.makedirs(output_dir, exist_ok=True)

    serializable_result = {
        "task": result["task"],
        "todos": result["todos"],
        "message_count": len(result["messages"])
    }

    filepath = os.path.join(output_dir, filename)
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(serializable_result, f, indent=2)

    print(f"Saved result to {filepath}")
    return filepath


# -------------------------
# Main
# -------------------------
if __name__ == "__main__":
    print("=" * 70)
    print("AUTONOMOUS COGNITIVE AGENT")
    print("Milestone 4: Full Autonomous System")
    print("=" * 70)

    task = "summarize how a recommendation system works and list its key components"

    result = run_full_agent(task)

    if result:
        print("\n" + "=" * 70)
        print("AUTONOMOUS AGENT FINAL REPORT")
        print("=" * 70 + "\n")

        print(result["final_answer"])

        print("\n" + "=" * 70)

        # =========================================
        # Evaluation
        # =========================================

        print("\n[EVALUATION] Scoring final output...\n")

        evaluation = llm.invoke(
            f"""
Rate the quality of the following report from 1 to 10.

Consider:
- completeness
- clarity
- structure
- usefulness

Report:
{result["final_answer"]}

Return only a number.
"""
        )

        print(f"Quality Score: {evaluation.content}")

    print("\nPipeline Execution Completed")