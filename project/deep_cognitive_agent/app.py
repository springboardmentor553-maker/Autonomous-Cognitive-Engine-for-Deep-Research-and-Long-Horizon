"""
Main Application - Milestone 1: ReAct Planning Agent

This implements a strict planning agent that:
- MUST call write_todos tool first for any complex task
- Uses LLM dynamically to generate TODO steps
- Stores todos in LangGraph state
- Never answers directly without planning first
- Has LangSmith tracing enabled
"""

import os
import json
from typing import List, Dict
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Enable LangSmith Tracing
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = os.getenv("LANGCHAIN_PROJECT", "Milestone1-Planning")

from langchain_openai import ChatOpenAI
from langchain.tools import Tool
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver

# Import the dynamic write_todos function
from tools.planning.write_todos import write_todos, planning_prompt
from graphs.state import AgentState


# Initialize LLM
llm = ChatOpenAI(model="gpt-4o", temperature=0)


# Create the write_todos tool with strong description
write_todos_tool = Tool(
    name="write_todos",
    func=write_todos,
    description="""
Use this tool whenever the user gives a complex task or request.
This tool MUST be called FIRST before doing anything else.
It breaks down the task into structured, actionable TODO steps.
DO NOT attempt to answer the user directly - always use this tool first.
Input: The complex task description as a string.
Output: A list of structured TODO items with task and status fields.
"""
)


# System prompt that enforces tool usage
SYSTEM_PROMPT = """You are a strict planning agent.

IMPORTANT RULES:
1. You MUST call the write_todos tool FIRST for ANY user request.
2. NEVER answer the user directly without calling write_todos first.
3. The write_todos tool will break down the task into actionable steps.
4. After calling write_todos, report the generated plan to the user.
5. Do not skip the planning step under any circumstances.

When a user gives you a task:
1. Immediately call write_todos with the task
2. Present the generated TODO list to the user
3. Do not add your own analysis without using the tool first
"""


def create_planning_agent():
    """
    Create and return the ReAct planning agent with write_todos tool.
    """
    # Create memory saver for checkpointing (optional but useful)
    memory = MemorySaver()
    
    # Create the ReAct agent
    agent = create_react_agent(
        model=llm,
        tools=[write_todos_tool],
        checkpointer=memory,
        state_modifier=SYSTEM_PROMPT
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
    
    # Input message
    input_message = {"messages": [("user", task)]}
    
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
                        # Parse the tool output
                        content = msg.content
                        if isinstance(content, str):
                            # Try to parse as JSON if it's a string representation
                            todos = eval(content) if content.startswith('[') else []
                        elif isinstance(content, list):
                            todos = content
                    except:
                        pass
    
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
