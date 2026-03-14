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

    files = final_state.get("files", {})
    if files:
        print(f"\n{sep}")
        print(f"VIRTUAL FILE SYSTEM ({len(files)} file(s))")
        print(sep)
        for fname, content in files.items():
            preview = content[:120].replace("\n", " ")
            print(f"  📄 {fname}  ({len(content)} chars)")
            print(f"     Preview: {preview}{'…' if len(content) > 120 else ''}")

    print(f"{sep}\n")


# ---------------------------------------------------------------------------
# Interactive REPL
# ---------------------------------------------------------------------------

def run_once(user_request: str) -> dict:
    """
    Execute a single user request through the full graph.

    Parameters
    ----------
    user_request : str
        The research task or question from the user.

    Returns
    -------
    dict
        The final LangGraph state after execution.
    """
    state = initial_state()
    state["messages"] = [HumanMessage(content=user_request)]

    print(f"\n[info] Starting graph execution for request:\n  '{user_request}'\n")

    # Stream events so the user can see progress
    step = 0
    final_state: dict = {}
    for event in graph.stream(state, stream_mode="values"):
        step += 1
        # Show which node just ran by checking the latest message type
        msgs = event.get("messages", [])
        if msgs:
            last = msgs[-1]
            msg_type = type(last).__name__
            print(f"  [step {step:02d}] {msg_type}", end="")
            if hasattr(last, "tool_calls") and last.tool_calls:
                names = [tc["name"] for tc in last.tool_calls]
                print(f" → tool calls: {names}", end="")
            print()
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
            else:
                print(f"\n[error] An error occurred during execution: {exc}")
                import traceback
                traceback.print_exc()


if __name__ == "__main__":
    main()
