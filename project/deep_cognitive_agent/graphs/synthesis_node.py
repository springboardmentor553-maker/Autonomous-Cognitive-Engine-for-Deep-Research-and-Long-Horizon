"""
Synthesis Node - Milestone 2

Reads all summaries from the virtual file system using read_file()
and ls(), then invokes the LLM to produce one final structured summary.

Architecture:  START → plan → execute → **synthesize** → END
"""

import re
import time

from langchain_core.messages import AIMessage

from tools.vfs.read_file import read_file
from tools.vfs.ls import ls


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

def synthesize_node(state: dict, llm) -> dict:
    """
    Synthesis node: reads every file in the VFS using read_file(),
    then asks the LLM to combine the content into one structured summary.

    Args:
        state: Current AgentState dict.
        llm:   ChatGroq (or compatible) LLM instance.

    Returns:
        Partial state update with ``final_output``, ``todos``, and
        ``messages``.
    """
    files = dict(state.get("files", {}))
    vfs_state = {"files": files}

    # ── Step 1: List all files in VFS ──
    file_list = ls(vfs_state)
    print(f"\n{'='*60}")
    print(f"[Synthesize Node] Files in VFS: {file_list}")
    print(f"{'='*60}")

    # ── Step 2: Read each file back via read_file ──
    all_content = []
    for fname in sorted(file_list):
        content = read_file(vfs_state, fname)
        preview = content[:120].replace("\n", " ")
        print(f"  → read_file('{fname}'): {len(content)} chars  \"{preview}...\"")
        all_content.append(f"--- {fname} ---\n{content}")

    combined_text = "\n\n".join(all_content)

    # ── Step 3: Generate final structured summary with LLM ──
    prompt = (
        "You are given individual summaries about a topic. "
        "Create ONE final structured summary that combines all key points.\n\n"
        "Use this structure:\n"
        "1. **Overview**: Brief introduction to the topic\n"
        "2. **Key Findings**: Bullet points of the most important facts\n"
        "3. **Analysis**: Deeper analysis connecting the themes across summaries\n"
        "4. **Conclusion**: Final takeaway and implications\n\n"
        f"Individual summaries:\n\n{combined_text}\n\n"
        "Write the structured summary now:"
    )

    print(f"\n[Synthesize Node] Generating combined summary from {len(file_list)} files...")

    final_summary = None
    max_retries = 3
    for attempt in range(max_retries):
        try:
            response = llm.invoke(prompt)
            final_summary = response.content
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

    # ── Mark remaining todos as done ──
    todos = list(state.get("todos", []))
    for i in range(len(todos)):
        if todos[i].get("status") != "done":
            todos[i] = {**todos[i], "status": "done"}

    print(f"[Synthesize Node] Final summary generated ({len(final_summary)} chars)")

    return {
        "final_output": final_summary,
        "todos": todos,
        "messages": [
            AIMessage(
                content=f"Combined structured summary created from {len(file_list)} files."
            )
        ],
    }
