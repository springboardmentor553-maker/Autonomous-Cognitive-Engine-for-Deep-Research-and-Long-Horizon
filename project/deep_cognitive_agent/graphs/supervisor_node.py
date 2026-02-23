"""
Supervisor / Planning Node - Milestone 2

Creates structured TODOs for the user task by calling the
write_todos planning tool.  This is the first node in the
StateGraph:  START → plan → execute → synthesize → END
"""

import json
import re
import time

from langchain_core.messages import AIMessage

from tools.planning.write_todos import write_todos


# ── Helpers ──────────────────────────────────────────────────────────

def _parse_retry_after(err_str: str) -> int:
    """Extract wait seconds from a Groq rate-limit error."""
    match = re.search(r"try again in (?:(\d+)m)?(\d+(?:\.\d+)?)s", err_str)
    if match:
        minutes = int(match.group(1) or 0)
        seconds = float(match.group(2))
        return int(minutes * 60 + seconds) + 2
    return 30


# ── Node Function ────────────────────────────────────────────────────

def plan_node(state: dict, llm) -> dict:
    """
    Planning node: extract the user task from messages and create
    structured TODOs via write_todos.

    Args:
        state: Current AgentState dict.
        llm:   LLM instance (unused here – planning uses its own LLM
               via write_todos, but kept for interface consistency).

    Returns:
        Partial state update with ``todos`` and an informational
        ``messages`` entry.
    """
    # Extract task from the last human message
    task = ""
    for msg in reversed(state.get("messages", [])):
        if hasattr(msg, "content") and hasattr(msg, "type") and msg.type == "human":
            task = msg.content
            break
        elif isinstance(msg, tuple) and msg[0] == "human":
            task = msg[1]
            break

    if not task:
        task = "Perform the assigned task"

    print(f"\n{'='*60}")
    print(f"[Plan Node] Creating TODOs for: {task}")
    print(f"{'='*60}")

    # Call write_todos with retry logic for rate limits
    max_retries = 3
    result = None
    for attempt in range(max_retries):
        try:
            result = write_todos(task)
            break
        except Exception as e:
            err_str = str(e)
            is_rate_limit = "429" in err_str or "rate_limit" in err_str.lower()
            if is_rate_limit and attempt < max_retries - 1:
                wait = _parse_retry_after(err_str)
                print(f"  ⏳ Rate limited. Waiting {wait}s...")
                time.sleep(wait)
                continue
            raise

    todos = result.get("todos", [])

    print(f"\n[Plan Node] Generated {len(todos)} TODOs:")
    for i, todo in enumerate(todos, 1):
        print(f"  {i}. ⬜ {todo['task']}")

    # Return partial state update
    return {
        "todos": todos,
        "messages": [
            AIMessage(
                content=(
                    f"Plan created with {len(todos)} steps: "
                    f"{json.dumps([t['task'] for t in todos])}"
                )
            )
        ],
    }
