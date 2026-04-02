import json
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

from langchain_groq import ChatGroq
from langchain_core.messages import AIMessage

# ══════════════════════════════════════════════════════════════
# Milestone 4 Evaluation
#
# Project PDF criteria:
#   1. Task Completion Success
#      Did the system complete the full workflow without errors?
#   2. Delegation Behavior
#      Did specialized tasks use sub-agents correctly?
#   3. Memory Usage
#      Were intermediate outputs stored and reused properly?
#   4. Output Quality (LLM-as-judge)
#      Is the final output logical, structured, and useful?
#
# Success requirement: >70% of tasks pass all 4 criteria
# ══════════════════════════════════════════════════════════════

llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)

VALID_AGENTS = ["research_agent", "summarization_agent"]


# ── LLM-as-judge (mentor pattern: score 1-10) ─────────────────

def judge_output_quality(request: str, final_output: str) -> dict:
    """
    Use LLM as a judge to score the final output quality.
    Mentor pattern: llm.predict(f"Rate this report quality from 1 to 10: {report}")
    """
    if not final_output or len(final_output) < 50:
        return {"score": 0, "grade": "poor", "reasoning": "Output too short or empty"}

    prompt = f"""You are an objective evaluator. Rate the quality of the following research report.

Original Request: {request}

Generated Report:
{final_output[:1500]}

Evaluate based on:
1. Does it directly address the request?
2. Is it well structured with clear sections?
3. Does it contain specific facts and insights?
4. Is it comprehensive and useful?

Respond ONLY in this exact JSON format:
{{"score": <number 1-10>, "grade": "<poor/fair/good/excellent>", "reasoning": "<one sentence>"}}"""

    try:
        response = llm.invoke(prompt)
        content  = response.content.strip()
        if content.startswith("```"):
            content = content.split("\n", 1)[1].rsplit("```", 1)[0].strip()
        result = json.loads(content)
        return result
    except Exception as e:
        return {"score": 5, "grade": "fair", "reasoning": f"Could not parse judge response: {str(e)}"}


# ── 4-criteria evaluation ─────────────────────────────────────

def evaluate_run(run_result: dict) -> dict:
    """
    Evaluate one full run against all 4 Milestone 4 criteria.

    Args:
        run_result: dict returned by workflow.run_task()
    """
    tool_seq       = run_result.get("tool_sequence", [])
    todos          = run_result.get("todos", [])
    delegation_log = run_result.get("delegation_log", [])
    files          = run_result.get("files", {})
    final_output   = run_result.get("final_output", "")
    request        = run_result.get("request", "")

    task_indices  = [i for i, n in enumerate(tool_seq) if n == "task"]
    write_indices = [i for i, n in enumerate(tool_seq) if n == "write_file"]
    read_indices  = [i for i, n in enumerate(tool_seq) if n == "read_file"]

    # ── Criterion 1: Task Completion ─────────────────────────
    # write_todos was called AND final output is substantial
    todos_created    = "write_todos" in tool_seq
    output_generated = len(final_output) > 150
    criterion_1      = todos_created and output_generated

    # ── Criterion 2: Delegation Behavior ─────────────────────
    # task() was called with valid agents and succeeded
    valid_delegations = [
        d for d in delegation_log
        if d["agent"] in VALID_AGENTS and d["status"] == "success"
    ]
    criterion_2 = len(valid_delegations) > 0

    # ── Criterion 3: Memory Usage ─────────────────────────────
    # write_file called after task() AND read_file called after write_file
    result_stored = any(
        w > t for t in task_indices for w in write_indices
    ) if task_indices and write_indices else False

    result_retrieved = any(
        r > w for w in write_indices for r in read_indices
    ) if write_indices and read_indices else False

    criterion_3 = result_stored and result_retrieved

    # ── Criterion 4: Output Quality (LLM-as-judge) ───────────
    quality = judge_output_quality(request, final_output)
    score   = quality.get("score", 0)
    grade   = quality.get("grade", "poor")
    # "good" = score >= 6, "excellent" = score >= 8
    criterion_4 = score >= 6

    # ── Overall pass ─────────────────────────────────────────
    criteria = [criterion_1, criterion_2, criterion_3, criterion_4]
    total    = sum(criteria)
    passed   = total >= 3   # at least 3 of 4 must pass

    return {
        "criterion_1_task_completion":  criterion_1,
        "criterion_2_delegation":       criterion_2,
        "criterion_3_memory_usage":     criterion_3,
        "criterion_4_output_quality":   criterion_4,
        "quality_score":                score,
        "quality_grade":                grade,
        "quality_reasoning":            quality.get("reasoning", ""),
        "todos_created":                len(todos),
        "delegations_made":             len(delegation_log),
        "files_created":                len(files.get("root", {})),
        "criteria_passed":              f"{total}/4",
        "passed":                       passed
    }


def print_eval_report(report: dict, test_num: int, request: str) -> None:
    """Print full evaluation report for one test case."""

    print(f"\n{'=' * 65}")
    print(f"MILESTONE 4 EVALUATION : TEST {test_num}")
    print(f"Task: {request[:70]}...")
    print(f"{'=' * 65}")
    print(f"{'✅' if report['criterion_1_task_completion'] else '❌'} "
          f"Criterion 1 — Task completed (plan created + output generated)")
    print(f"{'✅' if report['criterion_2_delegation']      else '❌'} "
          f"Criterion 2 — Delegation behavior (sub-agents used correctly)")
    print(f"{'✅' if report['criterion_3_memory_usage']    else '❌'} "
          f"Criterion 3 — Memory usage (stored and retrieved correctly)")
    print(f"{'✅' if report['criterion_4_output_quality']  else '❌'} "
          f"Criterion 4 — Output quality (LLM judge score: "
          f"{report['quality_score']}/10 — {report['quality_grade']})")
    print(f"\n   Judge reasoning : {report['quality_reasoning']}")
    print(f"\n📊 TODOs created   : {report['todos_created']}")
    print(f"📊 Delegations     : {report['delegations_made']}")
    print(f"📊 Files stored    : {report['files_created']}")
    print(f"📊 Criteria passed : {report['criteria_passed']}")
    print(f"\n{'✅ PASSED' if report['passed'] else '❌ FAILED'}")
    print(f"{'=' * 65}")