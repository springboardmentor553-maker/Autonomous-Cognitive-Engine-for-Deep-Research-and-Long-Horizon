"""
Execution Node - Milestone 2

For each research TODO, uses the LLM to generate detailed content
and writes each summary to the virtual file system via write_file().

Architecture:  START → plan → **execute** → synthesize → END
"""

import re
import time

from langchain_core.messages import AIMessage

from tools.vfs.write_file import write_file


# ── Helpers ──────────────────────────────────────────────────────────

def _parse_retry_after(err_str: str) -> int:
    """Extract wait seconds from a Groq rate-limit error."""
    match = re.search(r"try again in (?:(\d+)m)?(\d+(?:\.\d+)?)s", err_str)
    if match:
        minutes = int(match.group(1) or 0)
        seconds = float(match.group(2))
        return int(minutes * 60 + seconds) + 2
    return 30


def _invoke_llm_with_retry(llm, prompt: str, max_retries: int = 3) -> str:
    """Invoke the LLM with automatic rate-limit retry.

    Returns:
        The text content of the LLM response.
    """
    for attempt in range(max_retries):
        try:
            response = llm.invoke(prompt)
            return response.content
        except Exception as e:
            err_str = str(e)
            is_rate_limit = "429" in err_str or "rate_limit" in err_str.lower()
            if is_rate_limit and attempt < max_retries - 1:
                wait = _parse_retry_after(err_str)
                print(f"  ⏳ Rate limited. Waiting {wait}s...")
                time.sleep(wait)
                continue
            raise


# ── Node Function ────────────────────────────────────────────────────

def execute_node(state: dict, llm) -> dict:
    """
    Execution node: iterate over the first N research TODOs,
    generate a detailed paragraph for each using the LLM,
    and persist each summary in the VFS via write_file().

    The number of research steps defaults to 3 (matching the
    climate-change demo task) but adapts to the todo count.

    Args:
        state: Current AgentState dict.
        llm:   ChatGroq (or compatible) LLM instance.

    Returns:
        Partial state update with ``files``, ``todos``, and ``messages``.
    """
    todos = list(state.get("todos", []))
    files = dict(state.get("files", {}))       # copy current VFS
    vfs_state = {"files": files}                # wrapper for VFS tool

    messages = []
    summary_count = 0

    # Use the first 3 TODOs (or fewer) as research topics
    num_research = min(3, len(todos))

    print(f"\n{'='*60}")
    print(f"[Execute Node] Processing {num_research} research steps")
    print(f"{'='*60}")

    for i in range(num_research):
        todo = todos[i]
        topic = todo["task"]
        summary_count += 1
        filename = f"summary{summary_count}.txt"

        print(f"\n  Step {i+1}/{num_research}: {topic}")

        # ── LLM generates a long research paragraph ──
        prompt = (
            f"Write one long, detailed paragraph (at least 150 words) about "
            f"the following topic. Include specific facts, data points, and "
            f"expert analysis where appropriate.\n\n"
            f"Topic: {topic}\n\n"
            f"Write ONLY the paragraph — no titles, headings, or extra formatting."
        )

        content = _invoke_llm_with_retry(llm, prompt)

        # ── Write to VFS using write_file tool ──
        result = write_file(vfs_state, filename, content)
        print(f"  → {result}  ({len(content)} chars)")

        # Mark TODO as done
        todos[i] = {**todo, "status": "done"}

        messages.append(
            AIMessage(content=f"Researched '{topic}' and saved to {filename}")
        )

        # Small delay between LLM calls to respect rate limits
        if i < num_research - 1:
            time.sleep(2)

    print(f"\n[Execute Node] Wrote {summary_count} files to VFS")

    return {
        "files": vfs_state["files"],
        "todos": todos,
        "messages": messages,
    }
