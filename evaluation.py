import json
from langchain_core.messages import AIMessage

# ══════════════════════════════════════════════════════════════════════
# Milestone 3 Evaluation
#
# 4 criteria directly from the project spec (PDF):
#
# Criterion 1: Main agent identifies the need to delegate
#              → write_todos was called AND task() was called
#
# Criterion 2: task tool called with correct parameters
#              → agent_name is a valid registered agent
#              → input_data is not empty
#
# Criterion 3: Sub-agent executes successfully
#              → task() returns status = "success"
#              → result is non-empty
#
# Criterion 4: Result is returned and integrated into the workflow
#              → write_file is called AFTER task() (result is saved)
#              → read_file is called after write_file OR final response
#                is substantial (plan continued beyond delegation)
#
# Pass mark: >80% of test cases (4 out of 5)
# ══════════════════════════════════════════════════════════════════════

VALID_AGENTS = ["research_agent", "summarization_agent"]


def evaluate_test(result: dict) -> dict:

    messages = result.get("messages", [])

    tool_seq     = []
    task_calls   = []
    write_calls  = []
    read_calls   = []
    todos_called = False

    for msg in messages:

        if hasattr(msg, "tool_calls") and msg.tool_calls:
            for call in msg.tool_calls:
                name = call["name"]
                args = call.get("args", {})
                tool_seq.append(name)

                if name == "write_todos":
                    todos_called = True

                elif name == "task":
                    task_calls.append({
                        "agent_name": args.get("agent_name", ""),
                        "input_data": args.get("input_data", ""),
                        "output":     None
                    })

                elif name == "write_file":
                    write_calls.append(args.get("filename", ""))

                elif name == "read_file":
                    read_calls.append(args.get("filename", ""))

        # Attach task() outputs to the matching call
        if hasattr(msg, "name") and msg.name == "task":
            try:
                parsed = json.loads(msg.content)
                for tc in reversed(task_calls):
                    if tc["output"] is None:
                        tc["output"] = parsed
                        break
            except Exception:
                pass

    # Final AI message
    final_ai = next(
        (m for m in reversed(messages) if isinstance(m, AIMessage)), None
    )
    final_response = (
        final_ai.content
        if final_ai and isinstance(final_ai.content, str)
        else ""
    )

    task_indices  = [i for i, n in enumerate(tool_seq) if n == "task"]
    write_indices = [i for i, n in enumerate(tool_seq) if n == "write_file"]
    read_indices  = [i for i, n in enumerate(tool_seq) if n == "read_file"]

    # ── Criterion 1 ───────────────────────────────────────────────────────────
    criterion_1 = todos_called and len(task_calls) > 0

    # ── Criterion 2 ───────────────────────────────────────────────────────────
    criterion_2 = (
        len(task_calls) > 0
        and all(
            tc["agent_name"] in VALID_AGENTS
            and len(tc["input_data"].strip()) > 0
            for tc in task_calls
        )
    )

    # ── Criterion 3 ───────────────────────────────────────────────────────────
    criterion_3 = (
        len(task_calls) > 0
        and all(
            tc["output"] is not None
            and tc["output"].get("status") == "success"
            and len(str(tc["output"].get("result", ""))) > 20
            for tc in task_calls
        )
    )

    # ── Criterion 4 ───────────────────────────────────────────────────────────
    result_saved = any(
        w > t for t in task_indices for w in write_indices
    ) if task_indices and write_indices else False

    result_reused = any(
        r > w for w in write_indices for r in read_indices
    ) if write_indices and read_indices else False

    plan_continued = len(final_response) > 150

    criterion_4 = result_saved and (result_reused or plan_continued)

    # ── Score ─────────────────────────────────────────────────────────────────
    score  = sum([criterion_1, criterion_2, criterion_3, criterion_4])
    passed = score == 4

    return {
        "criterion_1_delegation_identified": criterion_1,
        "criterion_2_correct_parameters":    criterion_2,
        "criterion_3_subagent_executed":     criterion_3,
        "criterion_4_result_integrated":     criterion_4,
        "task_calls":                        len(task_calls),
        "agents_used":                       [tc["agent_name"] for tc in task_calls],
        "score":                             f"{score}/4",
        "passed":                            passed
    }


def print_evaluation_report(report: dict, test_num: int) -> None:

    print(f"\n--- MILESTONE 3 EVALUATION REPORT : TEST {test_num} ---")
    print(f"{'✅' if report['criterion_1_delegation_identified'] else '❌'} "
          f"Criterion 1 — Agent identified need to delegate")
    print(f"{'✅' if report['criterion_2_correct_parameters']    else '❌'} "
          f"Criterion 2 — task() called with correct parameters")
    print(f"{'✅' if report['criterion_3_subagent_executed']     else '❌'} "
          f"Criterion 3 — Sub-agent executed successfully")
    print(f"{'✅' if report['criterion_4_result_integrated']     else '❌'} "
          f"Criterion 4 — Result integrated into workflow")
    print(f"\n📊 Agents used  : {report['agents_used']}")
    print(f"📊 task() calls : {report['task_calls']}")
    print(f"📊 Score        : {report['score']}")
    print(f"{'✅ PASSED' if report['passed'] else '❌ FAILED'}")
    print("-" * 52)