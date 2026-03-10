"""
main.py - Deep Cognitive Task Framework: Milestone 1
Interactive mode: accepts live user input from the terminal
"""

import json
import os
from dotenv import load_dotenv

load_dotenv()

# LangSmith Tracing — set BEFORE importing langchain modules
LANGCHAIN_TRACING = os.getenv("LANGCHAIN_TRACING_V2", "false").lower() == "true"
if LANGCHAIN_TRACING:
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY", "")
    os.environ["LANGCHAIN_PROJECT"] = os.getenv("LANGCHAIN_PROJECT", "milestone1-deep-agent")
    print(f"LangSmith tracing ENABLED → Project: {os.environ['LANGCHAIN_PROJECT']}")
else:
    print("ℹ  LangSmith tracing DISABLED (set LANGCHAIN_TRACING_V2=true in .env to enable)")

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage, SystemMessage
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode

from state import AgentState, TodoItem
from tools import PLANNING_TOOLS

# LLM Setup

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY not found. Please set it in your .env file.")

llm = ChatGoogleGenerativeAI(
    model="gemini-flash-latest",
    google_api_key=GOOGLE_API_KEY,
    temperature=0,
)

llm_with_tools = llm.bind_tools(PLANNING_TOOLS)

# System Prompt
SYSTEM_PROMPT = """You are a Deep Cognitive Task Agent specialized in structured task planning.

## PRIME DIRECTIVE — PLANNING FIRST (NON-NEGOTIABLE):
Before ANY other action, you MUST call `write_todos` to decompose the user's request.
Do NOT answer, explain, search, or use any other tool before calling `write_todos`.
This rule has zero exceptions. Every request starts with write_todos.

## RULE 1 — EXACTLY 5 STEPS, NO MORE, NO LESS:
When calling `write_todos`, you MUST provide a list of EXACTLY 5 tasks.
If you provide 4 or 6 tasks, that is an error. Always exactly 5.

## RULE 2 — STRONG ACTION VERBS (MANDATORY):
Every task string MUST begin with one of these five capitalized action verbs:
  RESEARCH   — gather information, find sources, investigate the topic
  ANALYZE    — examine, evaluate, compare, and assess what was gathered
  SYNTHESIZE — combine findings, merge insights, consolidate information
  DRAFT      — write, compose, or create the required output
  REVIEW     — verify, validate, refine, and finalize the output

## THE REQUIRED 5-STEP STRUCTURE (always in this order):
  Task 1: RESEARCH   [gather all relevant information on the topic]
  Task 2: ANALYZE    [examine and evaluate the gathered information]
  Task 3: SYNTHESIZE [combine findings into coherent insights]
  Task 4: DRAFT      [write/create the required output artifact]
  Task 5: REVIEW     [verify accuracy, refine, and finalize]

## EXAMPLE — "Write a report on renewable energy":
write_todos(tasks=[
  "RESEARCH recent data, studies, and statistics on renewable energy sources from 2020-2024",
  "ANALYZE the growth trends, cost reductions, and adoption barriers in the gathered research",
  "SYNTHESIZE key findings into coherent insights covering technological advances and market dynamics",
  "DRAFT a comprehensive report with executive summary, main sections, and conclusions",
  "REVIEW the report for factual accuracy, logical flow, and completeness then finalize"
])

## YOUR WORKFLOW:
1. Receive user request
2. IMMEDIATELY call write_todos with EXACTLY 5 tasks
3. After write_todos succeeds, confirm the plan to the user

CRITICAL: EXACTLY 5 tasks. Each MUST start with RESEARCH / ANALYZE / SYNTHESIZE / DRAFT / REVIEW.
"""



# Helpers


def extract_todos_from_messages(state: AgentState) -> list[TodoItem]:
    todos = list(state.get("todos", []))
    for msg in state["messages"]:
        if isinstance(msg, ToolMessage) and msg.name == "write_todos":
            try:
                result = json.loads(msg.content)
                if result.get("success") and "todos" in result:
                    existing_ids = {t["id"] for t in todos}
                    for todo in result["todos"]:
                        if todo["id"] not in existing_ids:
                            todos.append(todo)
            except (json.JSONDecodeError, KeyError):
                pass
        elif isinstance(msg, ToolMessage) and msg.name == "mark_todo_complete":
            try:
                result = json.loads(msg.content)
                if result.get("success"):
                    todo_id = result["todo_id"]
                    for todo in todos:
                        if todo["id"] == todo_id:
                            todo["status"] = "completed"
            except (json.JSONDecodeError, KeyError):
                pass
    return todos


def check_write_todos_invoked(state: AgentState) -> bool:
    for msg in state["messages"]:
        if isinstance(msg, ToolMessage) and msg.name == "write_todos":
            return True
    return False


# Graph Nodes

def agent_node(state: AgentState) -> AgentState:
    messages = state["messages"]
    if not messages or not isinstance(messages[0], SystemMessage):
        messages = [SystemMessage(content=SYSTEM_PROMPT)] + list(messages)
    response = llm_with_tools.invoke(messages)
    return {
        "messages": [response],
        "todos": state.get("todos", []),
        "current_task": state.get("current_task", ""),
        "final_output": state.get("final_output", ""),
        "write_todos_invoked": state.get("write_todos_invoked", False),
    }


def tool_node_wrapper(state: AgentState) -> AgentState:
    tool_node = ToolNode(PLANNING_TOOLS)
    result = tool_node.invoke(state)
    updated_state = {**state, **result}
    return {
        **result,
        "todos": extract_todos_from_messages(updated_state),
        "write_todos_invoked": check_write_todos_invoked(updated_state),
    }


def should_continue(state: AgentState) -> str:
    last_message = state["messages"][-1]
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "tools"
    return END


# Build Graph

def build_agent():
    graph = StateGraph(AgentState)
    graph.add_node("agent", agent_node)
    graph.add_node("tools", tool_node_wrapper)
    graph.set_entry_point("agent")
    graph.add_conditional_edges("agent", should_continue, {"tools": "tools", END: END})
    graph.add_edge("tools", "agent")
    return graph.compile()


# Run Agent (single request)

def run_agent(user_request: str, run_name: str = "agent-run") -> dict:
    agent = build_agent()
    initial_state: AgentState = {
        "messages": [HumanMessage(content=user_request)],
        "todos": [],
        "current_task": "",
        "final_output": "",
        "write_todos_invoked": False,
    }

    invoke_config = {"run_name": run_name} if LANGCHAIN_TRACING else {}
    final_state = agent.invoke(initial_state, config=invoke_config)

    # Update state flags
    final_state["write_todos_invoked"] = check_write_todos_invoked(final_state)
    final_state["todos"] = extract_todos_from_messages(final_state)

    return final_state


# Display Results


def display_results(state: dict, user_request: str):
    todos = state.get("todos", [])
    write_todos_called = state.get("write_todos_invoked", False)

    print("\n" + "─" * 60)
    print("  📋  GENERATED TASK PLAN")
    print("─" * 60)
    print(f"  write_todos invoked : {'✅ YES' if write_todos_called else '❌ NO'}")
    print(f"  Total tasks created : {len(todos)} {'✅' if len(todos) == 5 else '⚠️  (expected 5)'}")
    print()

    if todos:
        for i, todo in enumerate(todos, 1):
            print(f"  {i}. [{todo['id']}] {todo['task']}")
            print(f"     Status: {todo['status']}")
            print()
    else:
        print("  ⚠️  No tasks were generated.")

    # Show agent's final text response
    last_msg = state["messages"][-1]
    if isinstance(last_msg, AIMessage) and last_msg.content:
        print("─" * 60)
        print("  🤖  AGENT RESPONSE")
        print("─" * 60)
        print(f"  {last_msg.content}")

    print("─" * 60)

    # Save to file
    output = {"request": user_request, "todos": todos}
    with open("generated_todos.json", "w") as f:
        json.dump(output, f, indent=2)
    print("  💾  Saved to generated_todos.json")
    print("─" * 60 + "\n")


# Interactive Input Loop

def interactive_mode():
    """Run the agent in a continuous interactive loop."""
    print("\n" + "=" * 60)
    print("  🧠  DEEP COGNITIVE TASK AGENT — Milestone 1")
    print("  Powered by LangGraph + Google Gemini")
    print("=" * 60)
    print("  Enter any complex task or research request.")
    print("  The agent will break it down into a 5-step plan.")
    print()
    print("  Commands:")
    print("    'quit' or 'exit' — stop the agent")
    print("    'clear'          — start fresh")
    print("=" * 60 + "\n")

    run_count = 0

    while True:
        try:
            # Get input from user
            user_input = input("  You: ").strip()

            # Handle special commands
            if not user_input:
                print("  ⚠️  Please enter a request.\n")
                continue

            if user_input.lower() in ("quit", "exit"):
                print("\n  👋  Goodbye!\n")
                break

            if user_input.lower() == "clear":
                os.system("cls" if os.name == "nt" else "clear")
                interactive_mode()
                return

            # Run the agent
            run_count += 1
            print(f"\n  ⏳  Planning your task...\n")

            state = run_agent(
                user_request=user_input,
                run_name=f"interactive-run-{run_count}"
            )

            display_results(state, user_input)

        except KeyboardInterrupt:
            print("\n\n  👋  Interrupted. Goodbye!\n")
            break
        except Exception as e:
            print(f"\n  ❌  Error: {e}\n")
            print("  Please try again.\n")


# Entry Point

if __name__ == "__main__":
    interactive_mode()