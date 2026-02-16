"""
Main Application - Milestone 1: ReAct Planning Agent
LangChain 1.x + LangGraph 1.x + Gemini
"""

import os
import ast
import json
from typing import Dict
from dotenv import load_dotenv

load_dotenv()

# Optional: disable tracing during development
os.environ["LANGCHAIN_TRACING_V2"] = "false"

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver

from tools.planning.write_todos import write_todos


# -------------------------
# LLM (Gemini)
# -------------------------
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0,
    google_api_key=os.getenv("GOOGLE_API_KEY"),
    max_output_tokens=800,
)

# IMPORTANT: bind tools to model
llm_with_tools = llm.bind_tools([write_todos])


# -------------------------
# Tool Registration (1.x style)
# -------------------------
@tool
def write_todos_tool(task: str):
    """Break a complex task into structured TODO steps."""
    return write_todos(task)


# -------------------------
# System Prompt
# -------------------------
SYSTEM_PROMPT = """You are a strict planning agent.

IMPORTANT RULES:
1. You MUST call the write_todos_tool FIRST for ANY user request.
2. NEVER answer directly without planning.
3. After generating the TODO list, present it clearly.
"""


# -------------------------
# Agent Factory
# -------------------------
def create_planning_agent():
    memory = MemorySaver()

    agent = create_react_agent(
        model=llm_with_tools,
        tools=[write_todos],
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

    # DEBUG: print all messages (optional)
    # print(final_state["messages"])

    for msg in final_state.get("messages", []):
        # In LangGraph 1.x tool messages have type == "tool"
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
    print("=" * 60)
    print("Milestone 1: ReAct Planning Agent (Gemini)")
    print("=" * 60)

    agent = create_planning_agent()

    task = "Build an AI chatbot architecture"

    result = run_agent(agent, task, thread_id="test-1")

    print("\nFinal Messages:")
    for msg in result["messages"]:
        print(msg)

    save_result_to_json(result, "test_output.json")

    print("\nDone.")
