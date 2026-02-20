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
import re
import json
import time
from typing import List, Dict
from dotenv import load_dotenv

# Load environment variables from .env file BEFORE any LangChain imports
load_dotenv()

# Enable LangSmith Tracing only when a key is available
os.environ["LANGCHAIN_TRACING_V2"] = os.getenv("LANGCHAIN_TRACING_V2", "false")
os.environ["LANGCHAIN_PROJECT"] = os.getenv("LANGCHAIN_PROJECT", "milestone_1_planning")

# ── Early validation ──
_groq_key = os.getenv("GROQ_API_KEY", "")
if not _groq_key or _groq_key.startswith("your_"):
    raise SystemExit(
        "\n[ERROR] GROQ_API_KEY is missing or still set to the placeholder.\n"
        "       1. Go to https://console.groq.com/keys and create an API key.\n"
        "       2. Put it in project/deep_cognitive_agent/.env:\n"
        "          GROQ_API_KEY=gsk_xxxxxxxxxxxxxxxxxxxxxxxx\n"
        "       3. Or set it in PowerShell:  $env:GROQ_API_KEY = \"gsk_xxx\"\n"
    )

from langchain_groq import ChatGroq
from langchain_core.tools import StructuredTool
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver
from pydantic import BaseModel, Field

# Import the dynamic write_todos function
from tools.planning.write_todos import write_todos, planning_prompt
from graphs.state import AgentState


# Initialize LLM for the agent.
# Model is configurable via GROQ_MODEL env var; defaults to 8b-instant
# which has much higher free-tier daily token limits than the 70b model.
_model_name = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
print(f"[init] Using Groq model: {_model_name}")
llm = ChatGroq(
    model=_model_name,
    temperature=0,
    groq_api_key=os.getenv("GROQ_API_KEY"),
)


def _parse_retry_after(err_str: str) -> int:
    """Extract recommended wait seconds from a Groq rate-limit error message."""
    match = re.search(r"try again in (?:(\d+)m)?(\d+(?:\.\d+)?)s", err_str)
    if match:
        minutes = int(match.group(1) or 0)
        seconds = float(match.group(2))
        return int(minutes * 60 + seconds) + 2  # small safety margin
    return 30  # sensible default


# Explicit input schema so the LLM knows to pass {"task": "..."} not __arg1
class WriteTodosInput(BaseModel):
    task: str = Field(description="The complex task description to break down into TODO steps")


# Create the write_todos tool with explicit schema
write_todos_tool = StructuredTool.from_function(
    func=write_todos,
    name="write_todos",
    description=(
        "Use this tool to decompose complex tasks into structured to-do lists "
        "before any execution. This tool MUST be called for every task. "
        "Input: a single 'task' string with the task description. "
        "Output: a dict with a 'todos' key containing a list of structured "
        "TODO items, each with 'task' and 'status' fields."
    ),
    args_schema=WriteTodosInput,
)


# ── System prompt — enforces strict planning-first discipline ──
SYSTEM_PROMPT = """You are an autonomous Planning Agent. Your ONLY responsibility is to create a clear, structured plan before any task execution.

CRITICAL RULES:
1. Call write_todos EXACTLY ONCE with the ENTIRE user task as input.
   - Do NOT call write_todos multiple times.
   - Do NOT split the task into separate tool calls.
   - Pass the full original task string to the tool.

2. Plan Before Acting (Mandatory)
   - ALWAYS create a plan before answering any task.
   - NEVER execute the task.
   - NEVER give the final answer.
   - Return ONLY the plan.

3. Stop After Planning
   - After the tool returns the to-do list:
     * DO NOT continue reasoning
     * DO NOT provide explanations
     * DO NOT answer the question
     * DO NOT summarize
   - Present the tool output and stop.

FORBIDDEN ACTIONS:
- Calling write_todos more than once
- Making multiple parallel tool calls
- Answering the task or executing steps
- Explaining reasoning or producing extra text

SUCCESS: One write_todos call → present the result → stop."""


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
    
    # Run the agent with retry on rate-limit (429) errors
    final_state = None
    todos = []
    max_retries = 5
    event_stream = []

    for attempt in range(max_retries):
        try:
            # Use a fresh thread_id for retries to avoid stale message history
            retry_config = {"configurable": {"thread_id": f"{thread_id}-a{attempt}"}}
            event_stream = list(agent.stream(input_message, retry_config, stream_mode="values"))
            break
        except Exception as e:
            err_str = str(e)
            is_rate_limit = "429" in err_str or "rate_limit" in err_str.lower()

            if not is_rate_limit or attempt >= max_retries - 1:
                raise

            # Detect daily token limit (won't reset for hours)
            if "tokens per day" in err_str.lower() or "(tpd)" in err_str.lower():
                print(f"  ⚠ Daily token limit (TPD) reached for the current model.")
                print(f"    Tip: change GROQ_MODEL in .env to a model with higher limits,")
                print(f"         or wait until the daily quota resets (midnight UTC).")
                raise

            wait = _parse_retry_after(err_str)
            print(f"  ⏳ Rate limited. Waiting {wait}s before retry {attempt + 2}/{max_retries}...")
            time.sleep(wait)
            continue

    for event in event_stream:
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
