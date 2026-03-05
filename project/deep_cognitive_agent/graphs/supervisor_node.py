"""
Supervisor / Planning Node - Milestone 2

Creates structured TODOs for the user task by calling the
write_todos planning tool, then enriches each step with:
  - step_type: research | compare | unify | refine
  - output_file: meaningful filename derived from task content
  - depends_on: list of files this step needs to read

This is the first node in the StateGraph:
  START → plan → execute → synthesize → END
"""

import json
import re
import time

from langchain_core.messages import AIMessage

from tools.planning.write_todos import write_todos
from utils.helpers import parse_retry_after, is_rate_limit_error


# ── Step Classification ──────────────────────────────────────────────

def _classify_step(task_text: str) -> str:
    """Classify a step by its action type using keyword analysis."""
    text_lower = task_text.lower()

    # Check refine/improve keywords first (most specific)
    refine_kw = ["refine", "improve", "enhance", "update", "optimize",
                 "revise", "strengthen", "polish", "finalize"]
    if any(kw in text_lower for kw in refine_kw):
        return "refine"

    # Check unify/propose keywords
    unify_kw = ["propose", "unify", "unified", "combine", "develop a model",
                "create a model", "create a framework", "design a framework",
                "integrate", "merge", "consolidate", "formulate",
                "develop a framework", "build a model"]
    if any(kw in text_lower for kw in unify_kw):
        return "unify"

    # Check compare keywords
    compare_kw = ["compare", "contrast", "differences", "similarities",
                  "analyze across", "identify differences", "cross-reference",
                  "evaluate against", "side-by-side", "comparative"]
    if any(kw in text_lower for kw in compare_kw):
        return "compare"

    # Default to research
    return "research"


def _sanitize_filename(task_text: str) -> str:
    """Create a clean, meaningful filename from task description."""
    # Remove common action verbs from the beginning
    verbs = (r'^(summarize|analyze|research|examine|investigate|evaluate|'
             r'review|collect|gather|identify|study|explore|assess|'
             r'define|outline|describe|document|map|catalog)\s+'
             r'(the\s+|all\s+)?')
    cleaned = re.sub(verbs, '', task_text.lower(), count=1)

    # Extract only alphabetic words
    words = re.findall(r'[a-z]+', cleaned)

    # Remove stop words
    stop = {'the', 'a', 'an', 'of', 'for', 'and', 'in', 'to', 'with',
            'on', 'by', 'from', 'all', 'each', 'across', 'its', 'their',
            'between', 'about', 'key', 'main', 'different', 'various',
            'how', 'what', 'this', 'that', 'these', 'those', 'principles'}
    key_words = [w for w in words if w not in stop and len(w) > 2]

    # Take first 3 key words for filename
    name_part = "_".join(key_words[:3])
    return name_part if name_part else "output"


def _enrich_todos(raw_todos: list) -> list:
    """
    Enrich raw todo steps with step_type, output_file, and depends_on.

    This creates a clean dependency chain:
    - Research steps → independent write_file operations
    - Compare step → reads all research files
    - Unify step → reads comparison file
    - Refine step → reads and edits unified model file
    """
    enriched = []
    research_files = []
    comparison_file = None
    unified_file = None

    for todo in raw_todos:
        task_text = todo["task"]
        step_type = _classify_step(task_text)

        if step_type == "research":
            name = _sanitize_filename(task_text)
            filename = f"{name}_summary.txt"
            enriched.append({
                **todo,
                "step_type": step_type,
                "output_file": filename,
                "depends_on": [],
            })
            research_files.append(filename)

        elif step_type == "compare":
            filename = "comparison_analysis.txt"
            comparison_file = filename
            enriched.append({
                **todo,
                "step_type": step_type,
                "output_file": filename,
                "depends_on": list(research_files),
            })

        elif step_type == "unify":
            filename = "unified_model.txt"
            unified_file = filename
            deps = [comparison_file] if comparison_file else list(research_files)
            enriched.append({
                **todo,
                "step_type": step_type,
                "output_file": filename,
                "depends_on": deps,
            })

        elif step_type == "refine":
            target = unified_file or comparison_file or (
                research_files[-1] if research_files else "output.txt"
            )
            enriched.append({
                **todo,
                "step_type": step_type,
                "output_file": target,
                "depends_on": [target],
            })

    return enriched


# ── Node Function ────────────────────────────────────────────────────

def plan_node(state: dict, llm) -> dict:
    """
    Planning node: extract the user task from messages, create
    structured TODOs via write_todos, and enrich with dependency metadata.

    Returns:
        Partial state update with enriched ``todos``, ``trace_log``,
        and an informational ``messages`` entry.
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
    print(f"[Plan Node] Creating structured plan for:")
    print(f"  \"{task}\"")
    print(f"{'='*60}")

    trace_log = list(state.get("trace_log", []))

    # Call write_todos with retry logic for rate limits
    max_retries = 3
    result = None
    for attempt in range(max_retries):
        try:
            result = write_todos(task)
            break
        except Exception as e:
            err_str = str(e)
            if is_rate_limit_error(err_str) and attempt < max_retries - 1:
                wait = parse_retry_after(err_str)
                print(f"  ⏳ Rate limited. Waiting {wait}s...")
                time.sleep(wait)
                continue
            raise

    raw_todos = result.get("todos", [])
    trace_log.append({
        "action": "write_todos",
        "file": None,
        "purpose": f"Generate structured plan for task: {task[:80]}",
        "step": 0,
    })

    # Enrich steps with types, filenames, and dependencies
    enriched_todos = _enrich_todos(raw_todos)

    print(f"\n[Plan Node] Generated {len(enriched_todos)} enriched steps:")
    for i, todo in enumerate(enriched_todos, 1):
        stype = todo.get("step_type", "?")
        ofile = todo.get("output_file", "?")
        deps = todo.get("depends_on", [])
        dep_str = f" ← reads: {deps}" if deps else ""
        print(f"  {i}. [{stype:8s}] {todo['task']}")
        print(f"             → {ofile}{dep_str}")

    return {
        "todos": enriched_todos,
        "trace_log": trace_log,
        "messages": [
            AIMessage(
                content=(
                    f"Plan created with {len(enriched_todos)} enriched steps: "
                    + json.dumps([
                        {"task": t["task"], "type": t["step_type"],
                         "file": t["output_file"]}
                        for t in enriched_todos
                    ])
                )
            )
        ],
    }
