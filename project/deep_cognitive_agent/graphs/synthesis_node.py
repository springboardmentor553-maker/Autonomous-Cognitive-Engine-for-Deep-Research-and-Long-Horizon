"""
Synthesis Node - Milestone 2: Selective Retrieval + Final Summary

Demonstrates selective retrieval discipline:
  - Reads ONLY the key output files (unified_model.txt, comparison.txt)
  - Does NOT re-read all individual research summaries
  - Builds final output from processed/synthesized content only

Architecture:  START → plan → execute → **synthesize** → END
"""

import re
import time

from langchain_core.messages import AIMessage

from tools.vfs.read_file import read_file
from tools.vfs.ls import ls
from utils.helpers import parse_retry_after, is_rate_limit_error, is_server_overload_error


# ── Node Function ────────────────────────────────────────────────────

def synthesize_node(state: dict, llm) -> dict:
    """
    Synthesis node: selectively reads key output files from VFS
    (NOT all files) and generates a final structured summary.

    Selective retrieval logic:
      1. If unified_model.txt exists → read only that (already refined)
      2. If comparison_analysis.txt exists → read that as supplement
      3. Only fall back to all files if no key files found

    This demonstrates:
      ✔ Selective retrieval — not blind loading
      ✔ Memory efficiency — reads processed outputs, not raw data
      ✔ Architectural discipline — clear reasoning for each read

    Args:
        state: Current AgentState dict.
        llm:   ChatGroq (or compatible) LLM instance.

    Returns:
        Partial state update with final_output, todos, trace_log,
        and messages.
    """
    files = dict(state.get("files", {}))
    vfs_state = {"files": files}
    trace_log = list(state.get("trace_log", []))

    # ── Step 1: List files and classify them ──
    file_list = ls(vfs_state)
    trace_log.append({
        "action": "ls",
        "file": None,
        "purpose": "Identify available files for selective retrieval",
        "step": "synthesis",
    })

    print(f"\n{'='*60}")
    print(f"[Synthesize Node] Files in VFS: {file_list}")
    print(f"{'='*60}")

    # Classify files into categories
    key_files = []      # unified models, comparisons
    summary_files = []  # individual research summaries

    for fname in sorted(file_list):
        if "unified" in fname or "model" in fname or "final" in fname:
            key_files.append(fname)
        elif "comparison" in fname or "analysis" in fname:
            key_files.append(fname)
        else:
            summary_files.append(fname)

    # ── Step 2: Selective retrieval — read only key files ──
    files_to_read = key_files if key_files else summary_files

    print(f"\n  Selective retrieval strategy:")
    print(f"    Key files (will read): {key_files}")
    print(f"    Summary files (skipped): {summary_files}")
    if not key_files:
        print(f"    Fallback: reading all {len(summary_files)} summary files")

    contents = []
    for fname in files_to_read:
        content = read_file(vfs_state, fname)
        purpose = (
            "Load key output for final synthesis"
            if fname in key_files
            else "Fallback: load summary for final synthesis"
        )
        trace_log.append({
            "action": "read_file",
            "file": fname,
            "purpose": purpose,
            "step": "synthesis",
        })
        preview = content[:100].replace("\n", " ")
        print(f"    → read_file('{fname}'): {len(content)} chars")
        contents.append(f"--- {fname} ---\n{content}")

    combined_text = "\n\n".join(contents)

    # ── Step 3: Generate final structured summary with LLM ──
    prompt = (
        "You are given the key outputs from a multi-step research and "
        "analysis pipeline. Create ONE final structured summary that "
        "presents the complete findings.\n\n"
        "Use this structure:\n"
        "1. **Overview**: Brief introduction to the topic and scope\n"
        "2. **Key Findings**: Bullet points of the most important "
        "discoveries and insights\n"
        "3. **Analysis**: Deeper analysis connecting themes, patterns, "
        "and implications\n"
        "4. **Recommendations**: Actionable recommendations based on "
        "the analysis\n"
        "5. **Conclusion**: Final takeaway and future considerations\n\n"
        f"Key outputs from the pipeline:\n\n{combined_text}\n\n"
        "Write the final structured summary now:"
    )

    print(f"\n[Synthesize Node] Generating final summary from "
          f"{len(files_to_read)} key files...")
    print(f"  (Skipped {len(summary_files)} raw summary files — "
          f"using processed outputs only)")

    final_summary = None
    max_retries = 3
    for attempt in range(max_retries):
        try:
            response = llm.invoke(prompt)
            final_summary = response.content
            break
        except Exception as e:
            err_str = str(e)
            if attempt < max_retries - 1:
                if is_rate_limit_error(err_str):
                    wait = parse_retry_after(err_str)
                    print(f"  ⏳ Rate limited. Waiting {wait}s...")
                    time.sleep(wait)
                    continue
                if is_server_overload_error(err_str):
                    wait = min(2 ** attempt * 10, 60)
                    print(f"  ⏳ Server overloaded (503). Waiting {wait}s...")
                    time.sleep(wait)
                    continue
            raise

    # ── Mark any remaining todos as done ──
    todos = list(state.get("todos", []))
    for i in range(len(todos)):
        if todos[i].get("status") != "done":
            todos[i] = {**todos[i], "status": "done"}

    trace_log.append({
        "action": "synthesize",
        "file": None,
        "purpose": f"Final structured summary from {len(files_to_read)} key files",
        "step": "synthesis",
    })

    print(f"[Synthesize Node] Final summary generated "
          f"({len(final_summary)} chars)")

    return {
        "final_output": final_summary,
        "todos": todos,
        "trace_log": trace_log,
        "messages": [
            AIMessage(
                content=(
                    f"Final structured summary created from "
                    f"{len(files_to_read)} key files "
                    f"(skipped {len(summary_files)} raw summaries)."
                )
            )
        ],
    }
