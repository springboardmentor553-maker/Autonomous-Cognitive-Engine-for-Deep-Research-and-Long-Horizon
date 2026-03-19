"""
main.py – Entry point for the Autonomous Cognitive Engine.

Usage
-----
    python main.py

The script:
1. Loads environment variables from .env
2. Validates all required API keys
3. Configures LangSmith tracing if requested
4. Builds the LangGraph
5. Runs an interactive REPL where the user can submit research requests

Press Ctrl+C or type "exit" to quit.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Load .env before importing anything that reads env vars
# ---------------------------------------------------------------------------
try:
    from dotenv import load_dotenv

    env_path = Path(__file__).parent / ".env"
    if env_path.exists():
        load_dotenv(dotenv_path=env_path)
        print(f"[info] Loaded environment from {env_path}")
    else:
        print(
            "[warning] No .env file found. "
            "Copy .env.example to .env and fill in your API keys."
        )
except ImportError:
    print("[warning] python-dotenv not installed; relying on shell environment.")

# ---------------------------------------------------------------------------
# Validate settings (raises early if keys are missing)
# ---------------------------------------------------------------------------
from config.settings import get_settings

try:
    settings = get_settings()
except ValueError as exc:
    print(f"\n[error] Configuration error:\n{exc}")
    sys.exit(1)

# ---------------------------------------------------------------------------
# Configure LangSmith tracing
# ---------------------------------------------------------------------------
if settings.langchain_tracing:
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_PROJECT"] = settings.langchain_project
    if settings.langchain_api_key:
        os.environ["LANGCHAIN_API_KEY"] = settings.langchain_api_key
    print(
        f"[info] LangSmith tracing ENABLED → project: '{settings.langchain_project}'"
    )
else:
    print("[info] LangSmith tracing is disabled (set LANGCHAIN_TRACING_V2=true to enable).")

# ---------------------------------------------------------------------------
# Build the graph
# ---------------------------------------------------------------------------
from langchain_core.messages import HumanMessage
from core.graph import build_graph
from core.state import initial_state

print("[info] Building LangGraph…")
graph = build_graph()
print("[info] Graph ready.\n")


# ---------------------------------------------------------------------------
# Helper: pretty-print the final state
# ---------------------------------------------------------------------------

def print_result(final_state: dict) -> None:
    """Print a human-friendly summary of a completed run."""
    sep = "─" * 70

    print(f"\n{sep}")
    print("FINAL OUTPUT")
    print(sep)
    print(final_state.get("final_output", "(no output)"))

    todos = final_state.get("todos", [])
    if todos:
        print(f"\n{sep}")
        print("TODO STATUS")
        print(sep)
        for i, t in enumerate(todos, 1):
            status_icon = {"pending": "○", "in_progress": "◑", "done": "●"}.get(
                t["status"], "?"
            )
            print(f"  {status_icon} [{i}] {t['task']}  ({t['status']})")

    delegation_history = final_state.get("delegation_history", [])
    if delegation_history:
        print(f"\n{sep}")
        print(f"SUB-AGENT DELEGATIONS  ({len(delegation_history)} total)")
        print(sep)
        for i, record in enumerate(delegation_history, 1):
            agent     = record.get("agent_name", "unknown")
            task_txt  = record.get("task", "")
            result    = record.get("result", "")
            icon      = {"web_search_agent": "🌐", "summarization_agent": "📝"}.get(agent, "🤖")
            task_prev  = task_txt[:70] + ("…" if len(task_txt) > 70 else "")
            result_prev = result[:150].replace("\n", " ") + ("…" if len(result) > 150 else "")
            print(f"  {icon} [{i}] {agent}")
            print(f"       Task   : {task_prev}")
            print(f"       Result : {result_prev}")

    files = final_state.get("files", {})
    if files:
        print(f"\n{sep}")
        print(f"VIRTUAL FILE SYSTEM  ({len(files)} file(s))")
        print(sep)
        for fname, content in files.items():
            preview = content[:120].replace("\n", " ")
            print(f"  📄 {fname}  ({len(content)} chars)")
            print(f"     Preview: {preview}{'…' if len(content) > 120 else ''}")

    print(f"{sep}\n")


# ---------------------------------------------------------------------------
# Interactive REPL
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Message type labels and icons for the live progress display
# ---------------------------------------------------------------------------

_TOOL_ICONS = {
    "write_todos":         "📋",
    "tavily_search":       "🔍",
    "delegate_task":       "🤖",
    "write_file":          "💾",
    "read_file":           "📖",
    "edit_file":           "✏️ ",
    "ls":                  "📁",
    "web_search_agent":    "🌐",
    "summarization_agent": "📝",
}


def _describe_tool_call(name: str, args: dict) -> str:
    """Return a one-line human-readable description of a tool call."""
    icon = _TOOL_ICONS.get(name, "🔧")

    if name == "write_todos":
        import json
        try:
            tasks = json.loads(args.get("tasks_json", "[]"))
            return f"{icon} write_todos  ({len(tasks)} tasks planned)"
        except Exception:
            return f"{icon} write_todos"

    elif name == "tavily_search":
        q = args.get("query", "")[:60]
        return f"{icon} tavily_search  → \"{q}\""

    elif name == "delegate_task":
        agent = args.get("agent_name", "?")
        task  = args.get("task", "")[:60]
        agent_icon = _TOOL_ICONS.get(agent, "🤖")
        return f"{icon} delegate_task  → {agent_icon} {agent}  task: \"{task}\""

    elif name in {"write_file", "edit_file"}:
        fname = args.get("filename", "?")
        return f"{icon} {name}  → \"{fname}\""

    elif name == "read_file":
        fname = args.get("filename", "?")
        return f"{icon} read_file  → \"{fname}\""

    return f"{icon} {name}"


def _describe_tool_result(tool_name: str, content: str) -> str | None:
    """
    For delegation results, extract and return a readable sub-agent summary.
    Returns None for non-delegation results (they don't need extra printing).
    """
    if tool_name != "delegate_task":
        return None

    # Parse the result content for delegation payloads
    import json
    try:
        data = json.loads(content)
        if data.get("action") == "delegate_task":
            agent = data.get("agent_name", "?")
            result = data.get("result", "")
            preview = result[:200].replace("\n", " ")
            icon = _TOOL_ICONS.get(agent, "🤖")
            return (
                f"       {icon} {agent} completed\n"
                f"       └─ {preview}{'…' if len(result) > 200 else ''}"
            )
    except Exception:
        pass

    # If the result is a plain "Sub-agent '...' completed successfully." string
    if "completed successfully" in content:
        lines = content.split("\n", 3)
        header = lines[0] if lines else content
        preview = lines[-1][:200].replace("\n", " ") if len(lines) > 1 else ""
        return f"       └─ {header}\n       └─ {preview}"

    return None


def run_once(user_request: str) -> dict:
    """
    Execute a single user request through the full graph.

    Streams every step to the terminal with rich labels showing:
    - Which tool the main agent called (with icon + args preview)
    - For delegate_task: which sub-agent ran and a result preview
    - ToolMessage results for delegation calls
    - The final AIMessage when synthesis is complete

    Parameters
    ----------
    user_request : str
        The research task or question from the user.

    Returns
    -------
    dict
        The final LangGraph state after execution.
    """
    from langchain_core.messages import AIMessage, ToolMessage

    state = initial_state()
    state["messages"] = [HumanMessage(content=user_request)]

    print(f"\n[info] Starting graph execution for request:\n  '{user_request}'\n")

    step = 0
    prev_msg_count = 0
    final_state: dict = {}

    # Track tool names from the previous AI step so ToolMessage results
    # can be annotated with what tool produced them
    last_tool_names: list[str] = []

    for event in graph.stream(state, stream_mode="values"):
        msgs = event.get("messages", [])
        new_msgs = msgs[prev_msg_count:]
        prev_msg_count = len(msgs)

        for msg in new_msgs:
            step += 1

            if isinstance(msg, AIMessage):
                if msg.tool_calls:
                    # One tool call per step (enforced), but loop for safety
                    for tc in msg.tool_calls:
                        desc = _describe_tool_call(tc["name"], tc.get("args", {}))
                        print(f"  [step {step:02d}] ▶  {desc}")
                    last_tool_names = [tc["name"] for tc in msg.tool_calls]

                else:
                    # Final synthesis message
                    print(f"  [step {step:02d}] ✅  Agent producing final answer…")
                    last_tool_names = []

            elif isinstance(msg, ToolMessage):
                # Show sub-agent internals for delegation, skip for others
                if last_tool_names and last_tool_names[-1] == "delegate_task":
                    sub_desc = _describe_tool_result("delegate_task", msg.content)
                    if sub_desc:
                        print(sub_desc)
                    else:
                        # Fallback: show a plain result line
                        preview = msg.content[:120].replace("\n", " ")
                        print(f"       └─ result: {preview}")

        final_state = event

    return final_state


def main() -> None:
    """Run the interactive REPL."""
    print("=" * 70)
    print("  Autonomous Cognitive Engine  –  Deep Research Agent")
    print("  Powered by LangGraph + Groq llama-3.1-8b-instant")
    print("=" * 70)
    print("Type your research request and press Enter.")
    print("Type 'exit' or press Ctrl+C to quit.\n")

    while True:
        try:
            user_input = input("You: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n[info] Goodbye!")
            break

        if not user_input:
            continue
        if user_input.lower() in {"exit", "quit", "q"}:
            print("[info] Goodbye!")
            break

        try:
            final_state = run_once(user_input)
            print_result(final_state)
        except Exception as exc:  # noqa: BLE001
            err_str = str(exc)
            # Friendly rate limit message
            if "rate_limit_exceeded" in err_str or "Rate limit" in err_str:
                import re
                wait = re.search(r"Please try again in (.+?)\.", err_str)
                wait_msg = f" Try again in {wait.group(1)}." if wait else ""
                print(
                    f"\n[rate limit] You have hit the Groq token limit for this model.{wait_msg}"
                    "\n  Options:"
                    "\n  1) Wait for the limit to reset (usually resets daily)"
                    "\n  2) Switch model in .env:  GROQ_MODEL=mixtral-8x7b-32768"
                    "\n  3) Switch model in .env:  GROQ_MODEL=llama-3.1-8b-instant"
                    "\n  4) Upgrade your Groq plan at https://console.groq.com/settings/billing"
                )
            elif "tool_use_failed" in err_str or "Failed to call a function" in err_str:
                print(
                    "\n[tool error] Groq failed to generate a valid tool call after all retries."
                    "\n  This is a known intermittent issue with Groq's llama models."
                    "\n  Options:"
                    "\n  1) Try the same prompt again — retries usually succeed"
                    "\n  2) Switch model in .env:  GROQ_MODEL=llama-3.1-8b-instant"
                    "\n  3) Simplify your prompt slightly"
                )
            elif "decommissioned" in err_str or "model_decommissioned" in err_str:
                import re
                model_match = re.search(r"model `(.+?)`", err_str)
                model_name = model_match.group(1) if model_match else "the selected model"
                print(
                    f"\n[model error] {model_name} has been decommissioned by Groq."
                    "\n  Update GROQ_MODEL in your .env file to an active model."
                    "\n  Current recommended models:"
                    "\n    GROQ_MODEL=llama-3.3-70b-versatile   (best quality)"
                    "\n    GROQ_MODEL=llama-3.1-8b-instant       (fast fallback)"
                    "\n  Check https://console.groq.com/docs/models for the full list."
                )
            else:
                print(f"\n[error] An error occurred during execution: {exc}")
                import traceback
                traceback.print_exc()


if __name__ == "__main__":
    main()
