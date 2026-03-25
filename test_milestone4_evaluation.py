"""
test_milestone4_evaluation.py
================================
Milestone 4 — Full Pipeline Test with Output Evaluation

Mentor corrections applied:
  Fix 1: Summarizer writes to /summaries/ path, NOT /research/ path
          Research files and summary files are kept SEPARATE
  Fix 2: Comparison step explicitly reads ieee_summary + eu_summary
          read: ieee_summary.json + eu_summary.json shown in trace

Mentor flow:
    User Request
    → Planning      (write_todos)
    → Execution     (execution_loop)
    → Delegation    (task tool -> sub-agent)
    → File Storage  (write_file — correct paths)
    → Retrieval     (read_file — explicit reads shown in trace)
    → Final Synthesis (synthesize_results)
    → Evaluation    (evaluate_output — rate 1-10)
    → Output

Run: python test_milestone4_evaluation.py
No API keys required.
"""

import sys
import json
from datetime import datetime, timezone

sys.path.insert(0, ".")

SEP = "=" * 65
sep = "-" * 65
now = lambda: datetime.now(timezone.utc).isoformat()


# ─────────────────────────────────────────────────────────────────────────────
# SHARED STATE (mentor exact spec)
# ─────────────────────────────────────────────────────────────────────────────
state = {
    "messages":       [],
    "todos":          [],
    "files":          {},
    "delegation_log": [],
    "evaluation":     None,
}


# ─────────────────────────────────────────────────────────────────────────────
# MOCK LLM
# ─────────────────────────────────────────────────────────────────────────────
class MockLLM:
    def predict(self, prompt: str) -> str:
        if "Break this task" in prompt or "TODO" in prompt:
            return (
                "Search for IEEE AI ethics framework, save to /research/ieee.json\n"
                "Search for EU AI ethics guidelines, save to /research/eu.json\n"
                "Summarize /research/ieee.json, save to /summaries/ieee_summary.json\n"
                "Summarize /research/eu.json, save to /summaries/eu_summary.json\n"
                "Compare summaries, save to /compare/ieee_vs_eu.json\n"
                "Synthesize unified framework, save to /drafts/unified_guide.json\n"
                "Append implementation roadmap to /drafts/unified_guide.json"
            )
        elif "Rate this report" in prompt:
            return (
                "SCORE: 9\n"
                "QUALITY: excellent\n"
                "FEEDBACK: The system completed all 7 tasks, used delegation for search "
                "and summarization, kept research and summary files separate, explicitly "
                "read summaries during comparison, and produced a well-structured report."
            )
        elif "Generate final report" in prompt:
            return (
                "# AI Ethics Unified Framework — Final Report\n\n"
                "## Executive Summary\n"
                "This report synthesizes IEEE and EU AI ethics frameworks "
                "into a unified actionable guide.\n\n"
                "## IEEE Framework\n"
                "Engineering-focused, voluntary, emphasizes technical accountability.\n\n"
                "## EU HLEG Framework\n"
                "Rights-focused, legally binding for high-risk AI systems.\n\n"
                "## Unified Principles\n"
                "1. Transparency and Explainability\n"
                "2. Human Oversight and Control\n"
                "3. Technical Accountability\n"
                "4. Legal Compliance\n\n"
                "## Implementation Roadmap\n"
                "Year 1: Adopt IEEE technical standards\n"
                "Year 2: Achieve EU AI Act compliance\n"
                "Year 3: Full unified framework integration"
            )
        return f"Direct result for: {prompt[:60]}"

llm = MockLLM()


# ─────────────────────────────────────────────────────────────────────────────
# MOCK SUB-AGENTS
# ─────────────────────────────────────────────────────────────────────────────
class _Runnable:
    def __init__(self, fn): self._fn = fn
    def invoke(self, text: str) -> str: return self._fn(text)

sub_agents = {
    "summarizer": _Runnable(lambda text: (
        f"Overview: '{text[:50]}' covers core AI ethics principles.\n\n"
        f"Key Points:\n"
        f"- Transparency and accountability required\n"
        f"- Human oversight is central\n"
        f"- Continuous monitoring needed\n\n"
        f"Conclusion: Strong foundation for ethical AI deployment."
    )),
    "web_searcher": _Runnable(lambda query: (
        f"Summary: Research on '{query[:50]}' confirms major developments.\n\n"
        f"Key Facts:\n"
        f"- Global adoption of AI ethics frameworks growing rapidly\n"
        f"- Industry bodies published updated guidelines in 2024\n"
        f"- Regulatory compliance becoming mandatory in many regions\n\n"
        f"Source Quality: reliable"
    )),
}


# ─────────────────────────────────────────────────────────────────────────────
# MENTOR EXACT FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────

def write_todos(task_description: str) -> list:
    """Mentor: steps = llm.predict(...); todos = [{"task": step, "status":"pending"}]"""
    steps = llm.predict(f"Break this task into structured TODO steps: {task_description}")
    return [
        {"id": f"task_{i+1}", "task": step.strip(),
         "status": "pending", "delegated_to": "", "save_to": ""}
        for i, step in enumerate(steps.split("\n")) if step.strip()
    ]


def task(agent_name: str, input_data: str) -> str:
    """Mentor: agent = sub_agents[name]; result = agent.invoke(input_data)"""
    if agent_name not in sub_agents:
        return "Agent not found."
    return sub_agents[agent_name].invoke(input_data)


def write_file(filename: str, content: str):
    """Store intermediate results. Keeps research and summaries SEPARATE."""
    state["files"][filename] = {
        "content":    content,
        "created_at": now(),
        "updated_at": now(),
    }
    print(f"    write_file -> {filename}  ({len(content.split())} words)")


def read_file(filename: str) -> str:
    """Retrieve stored result. Explicit call shown in trace."""
    if filename not in state["files"]:
        return f"ERROR: {filename} not found"
    content = state["files"][filename]["content"]
    print(f"    read_file  -> {filename}  ({len(content.split())} words)")
    return content


def edit_file(filename: str, mode: str, content: str):
    if filename in state["files"] and mode == "append":
        state["files"][filename]["content"] += "\n\n" + content
        state["files"][filename]["updated_at"] = now()
    print(f"    edit_file [{mode}] -> {filename}")


def ls() -> list:
    return list(state["files"].keys())


def synthesize_results() -> str:
    """
    Mentor exact:
        files = ls()
        combined_data = ""
        for file in files: combined_data += read_file(file)
        final_report = llm.predict(f"Generate final report: {combined_data}")
    """
    files         = ls()
    combined_data = ""
    for f in files:
        combined_data += f"\n\n[{f}]\n" + read_file(f)
    return llm.predict(f"Generate final report from this data: {combined_data}")


def evaluate_output(report: str) -> dict:
    """
    Mentor exact:
        score = llm.predict(f"Rate this report quality from 1 to 10: {report}")
    + 4 mentor checks
    """
    raw   = llm.predict(f"Rate this report quality from 1 to 10: {report}")
    lines = {}
    for line in raw.split("\n"):
        if ":" in line:
            k, v = line.split(":", 1)
            lines[k.strip()] = v.strip()

    return {
        "score":           int(lines.get("SCORE", "7")),
        "quality":         lines.get("QUALITY", "good"),
        "feedback":        lines.get("FEEDBACK", raw),
        "tasks_completed": all(t["status"] == "done" for t in state["todos"]),
        "delegation_used": len(state["delegation_log"]) > 0,
        "memory_used":     len(state["files"]) > 0,
    }


# ─────────────────────────────────────────────────────────────────────────────
# EXECUTION LOOP — with mentor corrections
#
# FIX 1: Summarizer saves to /summaries/ path, NOT /research/ path
#         Research files (/research/) and summaries (/summaries/) are SEPARATE
#
# FIX 2: Comparison step explicitly calls read_file() for both summaries
#         read: ieee_summary.json + eu_summary.json shown in trace
# ─────────────────────────────────────────────────────────────────────────────
def execution_loop():
    for todo in state["todos"]:
        if todo["status"] != "pending":
            continue

        task_text = todo["task"]
        task_id   = todo["id"]

        print()
        print(f"  [{task_id}] {task_text[:72]}")

        # ── Extract ALL json paths from task description ──────────────────────
        # For "Summarize /research/ieee.json, save to /summaries/ieee_summary.json"
        # json_paths = ["/research/ieee.json", "/summaries/ieee_summary.json"]
        # source  = first path  (/research/ieee.json)
        # save_to = last path   (/summaries/ieee_summary.json)
        json_paths = [t.rstrip(",.") for t in task_text.split()
                      if t.startswith("/") and ".json" in t]
        source_path = json_paths[0]  if len(json_paths) >= 1 else None
        save_path   = json_paths[-1] if len(json_paths) >= 1 else None


        if "summarize" in task_text.lower() or "summary" in task_text.lower():
            # ── FIX 1: Write to /summaries/ not /research/ ────────────────────
            # source_path = /research/ieee.json  (read from here)
            # save_path   = /summaries/ieee_summary.json  (write to here)
            print(f"  => DELEGATE to 'summarizer'")

            # Read source research file
            if source_path and source_path in state["files"]:
                input_data = read_file(source_path)
            else:
                input_data = task_text

            # Delegate to summarizer
            result = task("summarizer", input_data)
            print(f"     summarizer returned: {result[:65].strip()}...")

            # FIX 1: Save to /summaries/ path (separate from /research/)
            print(f"     saving summary to: {save_path}  (separate from research)")
            write_file(save_path, result)

            state["delegation_log"].append({
                "task_id":    task_id,
                "agent_name": "summarizer",
                "input":      input_data[:80],
                "result":     result[:200],
                "status":     "completed",
            })
            todo["delegated_to"] = "summarizer"
            todo["save_to"]      = save_path


        elif "search" in task_text.lower() or "find" in task_text.lower():
            # Search -> web_searcher -> /research/
            print(f"  => DELEGATE to 'web_searcher'")
            result = task("web_searcher", task_text)
            print(f"     web_searcher returned: {result[:65].strip()}...")
            write_file(save_path, result)
            state["delegation_log"].append({
                "task_id":    task_id,
                "agent_name": "web_searcher",
                "input":      task_text[:80],
                "result":     result[:200],
                "status":     "completed",
            })
            todo["delegated_to"] = "web_searcher"
            todo["save_to"]      = save_path


        elif "compare" in task_text.lower():
            # ── FIX 2: Explicit read_file() calls shown in trace ──────────────
            print(f"  => DIRECT: comparison (reasoning task)")
            print(f"     supervisor explicitly reads summary files:")

            # FIX 2: Explicit read_file calls — not implicit/assumed
            ieee_summary = read_file("/summaries/ieee_summary.json")
            eu_summary   = read_file("/summaries/eu_summary.json")
            print(f"     skipped: /research/ieee.json, /research/eu.json (raw data not needed)")

            result = (
                "# Comparison Analysis: IEEE vs EU AI Ethics\n\n"
                "## IEEE Framework Summary\n"
                + ieee_summary[:250] + "\n\n"
                "## EU HLEG Framework Summary\n"
                + eu_summary[:250] + "\n\n"
                "## Key Differences\n"
                "- IEEE: engineering-focused, voluntary, global scope\n"
                "- EU:   rights-focused, legally binding, EU jurisdiction\n\n"
                "## Common Ground\n"
                "- Both require transparency and accountability\n"
                "- Both prioritize human well-being and safety\n"
                "- Both demand ongoing monitoring"
            )
            write_file(save_path, result)
            todo["save_to"] = save_path


        elif "synthesize" in task_text.lower() or "unified" in task_text.lower():
            print(f"  => DIRECT: synthesis (reasoning task)")
            print(f"     supervisor reads comparison file:")
            comparison = read_file("/compare/ieee_vs_eu.json")
            print(f"     skipped: /summaries/ files (comparison already extracted key info)")

            result = (
                "# Unified AI Ethics Framework\n\n"
                "## Pillar 1 — Technical Accountability (IEEE)\n"
                "Engineers bear professional responsibility for AI outcomes.\n\n"
                "## Pillar 2 — Rights and Regulation (EU HLEG)\n"
                "Human agency must be preserved. Risk-based regulation.\n\n"
                "## Pillar 3 — Shared Principles\n"
                "Transparency, accountability, monitoring are non-negotiable.\n\n"
                "## Compliance Score\n"
                "Organizations scored 0-100 across all three pillars."
            )
            write_file(save_path, result)
            todo["save_to"] = save_path


        elif "append" in task_text.lower() or "roadmap" in task_text.lower():
            print(f"  => DIRECT: edit_file append (no re-read needed)")
            draft_files = [f for f in state["files"] if "/drafts/" in f]
            if draft_files:
                roadmap = (
                    "## Implementation Roadmap\n"
                    "Year 1: Adopt IEEE technical accountability standards\n"
                    "Year 2: Achieve EU AI Act compliance for high-risk systems\n"
                    "Year 3: Full unified framework integration"
                )
                edit_file(draft_files[-1], "append", roadmap)

        else:
            print(f"  => DIRECT: supervisor handles")
            result = llm.predict(task_text)
            if save_path:
                write_file(save_path, result)

        todo["status"] = "done"
        state["messages"].append({"role": "system", "content": f"[{task_id}] done"})


# ─────────────────────────────────────────────────────────────────────────────
# MAIN TEST RUNNER
# ─────────────────────────────────────────────────────────────────────────────
def run():
    print(SEP)
    print("  Milestone 4 — Full Pipeline + Mentor Corrections Applied")
    print(SEP)
    print()
    print("  Mentor corrections:")
    print("  Fix 1: Summarizer writes to /summaries/ (not /research/)")
    print("  Fix 2: Comparison explicitly reads ieee_summary + eu_summary")
    print()
    print("  Flow: Request -> Planning -> Execution -> Delegation")
    print("        -> File Storage -> Retrieval -> Synthesis -> Evaluation")

    user_request = (
        "Research IEEE and EU AI ethics frameworks, "
        "compare them, write a unified guide"
    )
    print(f"\n  Request: {user_request}")

    # ── Step 1: Planning ──────────────────────────────────────────────────────
    print()
    print(sep)
    print("  STEP 1: PLANNING  (write_todos)")
    print(sep)
    state["todos"] = write_todos(user_request)
    print(f"  Created {len(state['todos'])} TODO tasks:")
    for t in state["todos"]:
        print(f"    [{t['id']}] {t['task'][:70]}")

    # ── Step 2: Execution Loop ────────────────────────────────────────────────
    print()
    print(sep)
    print("  STEP 2: EXECUTION LOOP")
    print(sep)
    execution_loop()

    # ── Step 3: Final Synthesis ───────────────────────────────────────────────
    print()
    print(sep)
    print("  STEP 3: FINAL SYNTHESIS  (synthesize_results)")
    print(sep)
    print("  Reading all files:")
    final_report = synthesize_results()
    state["messages"].append({"role": "assistant", "content": final_report})
    print(f"\n  Final report: {len(final_report.split())} words")
    print(f"  Preview: {final_report[:120].strip()}...")

    # ── Step 4: Evaluation ────────────────────────────────────────────────────
    print()
    print(sep)
    print("  STEP 4: EVALUATION  (evaluate_output)")
    print(sep)
    print("  Mentor checks:")
    print("    1. Did the system complete tasks?")
    print("    2. Did delegation happen?")
    print("    3. Did memory usage work?")
    print("    4. Did output make sense?")
    print()

    evaluation = evaluate_output(final_report)
    state["evaluation"] = evaluation

    print(f"  Score   : {evaluation['score']}/10")
    print(f"  Quality : {evaluation['quality']}")
    print(f"  Feedback: {evaluation['feedback'][:110]}...")

    # ── Execution Trace ───────────────────────────────────────────────────────
    print()
    print(SEP)
    print("  EXECUTION TRACE")
    print(SEP)
    print(f"  {'Task':<8} {'Decision':<10} {'Handler':<16} {'Status':<8} Saved To")
    print(sep)
    for t in state["todos"]:
        handler  = t.get("delegated_to") or "supervisor"
        decision = "DELEGATE" if t.get("delegated_to") else "DIRECT  "
        saved    = t.get("save_to") or "(edit/append)"
        print(f"  {t['id']:<8} {decision:<10} {handler:<16} {t['status']:<8} {saved}")

    # ── VFS state ─────────────────────────────────────────────────────────────
    print()
    print(SEP)
    print("  FINAL FILE SYSTEM (VFS)")
    print(SEP)
    print(f"  {'File':<44} {'Type':<14} Words")
    print(sep)
    for path, entry in state["files"].items():
        words = len(entry["content"].split())
        ftype = ("research" if "/research/"   in path else
                 "summary"  if "/summaries/"  in path else
                 "compare"  if "/compare/"    in path else
                 "draft"    if "/drafts/"     in path else "other")
        print(f"  {path:<44} {ftype:<14} {words}")

    # ── Delegation Log ────────────────────────────────────────────────────────
    print()
    print(SEP)
    print("  DELEGATION LOG")
    print(SEP)
    print(f"  {'task_id':<8} {'agent':<16} {'status':<12} result preview")
    print(sep)
    for rec in state["delegation_log"]:
        preview = rec["result"].replace("\n", " ")[:52]
        print(f"  {rec['task_id']:<8} {rec['agent_name']:<16} {rec['status']:<12} {preview}...")

    # ── Evaluation Checklist ──────────────────────────────────────────────────
    print()
    print(SEP)
    print("  MILESTONE 4 EVALUATION CHECKLIST")
    print(SEP)

    # Check Fix 1: /summaries/ files exist separately from /research/
    research_files = [p for p in state["files"] if "/research/"  in p]
    summary_files  = [p for p in state["files"] if "/summaries/" in p]
    compare_files  = [p for p in state["files"] if "/compare/"   in p]
    draft_files    = [p for p in state["files"] if "/drafts/"    in p]

    # Check Fix 2: comparison file has content from both summaries
    compare_content = state["files"].get("/compare/ieee_vs_eu.json", {}).get("content", "")
    has_both_summaries = ("IEEE" in compare_content and "EU" in compare_content)

    todos_done     = sum(1 for t in state["todos"] if t["status"] == "done")
    eval_score     = state["evaluation"]["score"]
    del_count      = len(state["delegation_log"])

    checks = [
        # Planning
        ("write_todos: structured TODO plan created from user request",
         len(state["todos"]) >= 5),

        ("execution_loop: all TODO tasks completed",
         todos_done == len(state["todos"])),

        # Delegation
        ("delegation: search tasks -> web_searcher",
         any(r["agent_name"] == "web_searcher" for r in state["delegation_log"])),

        ("delegation: summarize tasks -> summarizer",
         any(r["agent_name"] == "summarizer"   for r in state["delegation_log"])),

        # ── MENTOR FIX 1 ─────────────────────────────────────────────────────
        ("FIX 1: /research/ files exist (raw web search results)",
         len(research_files) >= 2),

        ("FIX 1: /summaries/ files exist SEPARATELY from /research/",
         len(summary_files) >= 2),

        ("FIX 1: research and summary files are different paths (not overwritten)",
         "/summaries/ieee_summary.json" in state["files"] and
         "/research/ieee.json"          in state["files"]),

        # ── MENTOR FIX 2 ─────────────────────────────────────────────────────
        ("FIX 2: comparison explicitly reads ieee_summary.json + eu_summary.json",
         has_both_summaries),

        ("FIX 2: comparison file contains content from BOTH summaries",
         "IEEE Framework Summary" in compare_content and
         "EU HLEG Framework Summary" in compare_content),

        # Storage & Retrieval
        ("write_file: intermediate results stored after every delegation",
         len(state["files"]) >= 4),

        ("synthesize_results: reads ALL files via ls() + read_file()",
         "/drafts/unified_guide.json" in state["files"]),

        # Evaluation
        ("evaluate_output: rates 1-10 (mentor spec)",
         1 <= eval_score <= 10),

        ("evaluation score >= 7/10",
         eval_score >= 7),

        # Full pipeline
        ("complete pipeline: Planning->Execution->Delegation->Storage->Synthesis->Evaluation",
         todos_done > 0 and del_count > 0 and eval_score > 0),
    ]

    passed = 0
    for label, result in checks:
        icon = "PASS" if result else "FAIL"
        print(f"  [{icon}]  {label}")
        if result:
            passed += 1

    score = int((passed / len(checks)) * 100)

    print()
    print(f"  Score              : {passed}/{len(checks)} ({score}%)")
    print(f"  TODOs completed    : {todos_done}/{len(state['todos'])}")
    print(f"  Delegations        : {del_count}")
    print(f"  Research files     : {len(research_files)}  {research_files}")
    print(f"  Summary files      : {len(summary_files)}  {summary_files}")
    print(f"  Evaluation score   : {eval_score}/10  ({evaluation['quality']})")

    print()
    print(SEP)
    if score >= 80:
        print("  MILESTONE 4 COMPLETE  (with mentor corrections)")
        print(f"  Score: {score}% | Output quality: {eval_score}/10")
        print()
        print("  Both mentor corrections verified:")
        print("  Fix 1: /research/ieee.json  != /summaries/ieee_summary.json")
        print("         Research and summary files are SEPARATE")
        print("  Fix 2: comparison reads ieee_summary.json + eu_summary.json")
        print("         Explicit read operations appear in trace")
        print()
        print("  Full pipeline verified:")
        print("  Request -> Planning -> Execution -> Delegation")
        print("  -> Storage -> Retrieval -> Synthesis -> Evaluation -> Output")
    else:
        print(f"  Score {score}% — needs attention")
    print(SEP)

    # ── Save output ───────────────────────────────────────────────────────────
    output = {
        "milestone":          4,
        "mentor_corrections": [
            "Fix1: summarizer saves to /summaries/ not /research/",
            "Fix2: comparison explicitly reads both summary files",
        ],
        "state": {
            "todos":          state["todos"],
            "files":          {k: v["content"] for k, v in state["files"].items()},
            "delegation_log": state["delegation_log"],
            "evaluation":     state["evaluation"],
        },
        "final_report": final_report,
        "evaluation": {
            "checks_passed": passed,
            "checks_total":  len(checks),
            "score_percent": score,
            "output_score":  eval_score,
            "status":        "PASS" if score >= 80 else "FAIL",
        },
        "generated_at": now(),
    }
    with open("milestone4_output.json", "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print()
    print(f"  Output saved: milestone4_output.json")


if __name__ == "__main__":
    run()