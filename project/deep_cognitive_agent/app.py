"""
Main Application - Milestone 1: ReAct Planning Agent

This implements a strict planning agent that:
- MUST call write_todos tool first for any complex task
- Uses Groq (Llama 3.3 70B free tier) to dynamically generate TODO steps
- Stores todos in LangGraph state
- Never answers directly without planning first
- Has LangSmith tracing enabled
"""

import os
import json
from typing import List, Dict
from dotenv import load_dotenv

# Load environment variables from .env file BEFORE any LangChain imports
load_dotenv()

# Enable LangSmith Tracing
os.environ["LANGCHAIN_TRACING_V2"] = os.getenv("LANGCHAIN_TRACING_V2", "true")
os.environ["LANGCHAIN_PROJECT"] = os.getenv("LANGCHAIN_PROJECT", "milestone_1_planning")

from langchain_groq import ChatGroq
from langchain_core.tools import Tool
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver

# Import the dynamic write_todos function
from tools.planning.write_todos import write_todos, planning_prompt
from graphs.state import AgentState


# Initialize LLM (Groq free tier - Llama 3.3 70B)
llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0,
    groq_api_key=os.getenv("GROQ_API_KEY"),
)


# Create the write_todos tool with strong description
write_todos_tool = Tool(
    name="write_todos",
    func=write_todos,
    description=(
        "Use this tool to decompose complex tasks into structured to-do lists "
        "before any execution. This tool MUST be called for complex tasks. "
        "Input: The complex task description as a string. "
        "Output: A dict with a 'todos' key containing a list of structured "
        "TODO items, each with 'task' and 'status' fields."
    ),
)


# ── System prompt — enforces strict ReAct planning discipline ──
SYSTEM_PROMPT = """You are a strict ReAct planning agent for Milestone 1.

ABSOLUTE RULES — you must follow every one of these without exception:

1. For ANY complex task the user gives you, you MUST call the write_todos tool FIRST.
2. You MUST NOT answer the user directly or generate your own list of steps.
3. You MUST NOT skip planning or attempt to execute any task.
4. Milestone 1 only requires decomposition into structured todos — do NOT execute tasks.
5. After calling write_todos, report the structured TODO list returned by the tool.
   Do NOT add, remove, or reword the steps.

ReAct discipline:
  - THINK: reason briefly about what tool to call.
  - ACT: call write_todos with the user's task.
  - OBSERVE: read the structured todos returned.
  - RESPOND: present the todos to the user exactly as returned.

If the write_todos tool is not called, the response is INVALID."""


def create_planning_agent():
    """
    Create and return the ReAct planning agent with write_todos tool.
    """
    # Create memory saver for checkpointing (optional but useful)
    memory = MemorySaver()

    # Create the ReAct agent. The system behavior is injected later as a
    # system message when we call the agent, since this version of
    # create_react_agent no longer accepts system_prompt/state_modifier.
    agent = create_react_agent(
        model=llm,
        tools=[write_todos_tool],
        checkpointer=memory,
    )
    
    return agent


def run_agent(agent, task: str, thread_id: str = "default") -> Dict:
    """
    Run the planning agent on a task and return the result with todos.
    
    Args:
        agent: The ReAct agent instance
        task: The complex task to plan
        thread_id: Unique thread identifier for conversation
        
    Returns:
        Dict with 'messages' and 'todos' from the final state
    """
    # Configuration for the agent run
    config = {"configurable": {"thread_id": thread_id}}
    
    # Input messages: include a system message so the agent is
    # instructed to ALWAYS call write_todos first and never answer
    # directly.
    input_message = {"messages": [("system", SYSTEM_PROMPT), ("user", task)]}
    
    # Run the agent and collect the final state
    final_state = None
    todos = []
    
    for event in agent.stream(input_message, config, stream_mode="values"):
        final_state = event
        
        # Check for tool messages that contain todos
        if "messages" in event:
            for msg in event["messages"]:
                # Check if this is a tool message from write_todos
                if hasattr(msg, 'name') and msg.name == "write_todos":
                    try:
                        content = msg.content
                        if isinstance(content, str):
                            parsed = json.loads(content)
                        elif isinstance(content, dict):
                            parsed = content
                        else:
                            parsed = {}

                        # Handle {"todos": [...]} format from write_todos
                        if isinstance(parsed, dict) and "todos" in parsed:
                            todos = parsed["todos"]
                        elif isinstance(parsed, list):
                            todos = parsed
                    except (json.JSONDecodeError, TypeError):
                        pass
    
    # If we successfully extracted todos, also attach them to the
    # underlying LangGraph state object so that state["todos"] is
    # populated in addition to our returned result dictionary.
    if final_state is not None and todos:
        final_state["todos"] = todos

    # Build result
    result = {
        "task": task,
        "messages": final_state.get("messages", []) if final_state else [],
        "todos": todos
    }
    
    return result


def save_result_to_json(result: Dict, filename: str, output_dir: str = "outputs"):
    """
    Save the agent result to a JSON file.
    
    Args:
        result: The result dictionary from run_agent
        filename: Name of the output file
        output_dir: Directory to save outputs (created if doesn't exist)
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Prepare serializable result
    serializable_result = {
        "task": result["task"],
        "todos": result["todos"],
        "message_count": len(result["messages"])
    }
    
    # Add final assistant message if available
    for msg in reversed(result["messages"]):
        if hasattr(msg, 'content') and hasattr(msg, 'type') and msg.type == "ai":
            serializable_result["final_response"] = msg.content
            break
    
    # Save to JSON
    filepath = os.path.join(output_dir, filename)
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(serializable_result, f, indent=2, ensure_ascii=False)
    
    print(f"Saved result to {filepath}")
    return filepath


# Main execution
if __name__ == "__main__":
    print("=" * 60)
    print("Milestone 1: ReAct Planning Agent")
    print("=" * 60)
    
    # Create the agent
    agent = create_planning_agent()
    
    # Test task
    test_task = "Build an AI chatbot architecture"
    
    print(f"\nTask: {test_task}")
    print("-" * 40)
    
    # Run the agent
    result = run_agent(agent, test_task, thread_id="test-1")
    
    # Display todos
    print("\nGenerated TODOs:")
    for i, todo in enumerate(result["todos"], 1):
        print(f"  {i}. {todo['task']} [{todo['status']}]")
    
    # Save to JSON
    save_result_to_json(result, "test_output.json")
    
    print("\n" + "=" * 60)
    print("Agent run complete. Check LangSmith for traces.")
    print("=" * 60)
