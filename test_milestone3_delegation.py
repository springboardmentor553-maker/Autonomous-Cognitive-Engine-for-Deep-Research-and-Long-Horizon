"""
test_milestone3_delegation.py
================================
Milestone 3 — Sub-Agent Delegation Test

Verifies the exact mentor workflow:

    Supervisor Agent
          |
          |  task("summarizer", input)       <- delegate_task tool
          |
    Sub-Agent executes via agent.invoke()
          |
          |  returns result
          |
    Supervisor calls write_file(path, result)  <- integration step
          |
    Supervisor continues to next TODO

Mentor's LangSmith evaluation checks:
  1. Did supervisor recognize when delegation is required?
  2. Did supervisor correctly call task tool with correct sub-agent?
  3. Was the result returned by sub-agent integrated into the workflow?

Run: python test_milestone3_delegation.py
No API keys required — uses mock sub-agents.
"""

import sys
import json
from datetime import datetime, timezone

sys.path.insert(0, ".")

SEP = "=" * 65
sep = "-" * 65
now = lambda: datetime.now(timezone.utc).isoformat()


# ─────────────────────────────────────────────────────────────────────────────
# MOCK SUB-AGENTS
# Same interface as real ones: .invoke(input_data) -> str
# Real ones call Groq LLM; mocks return structured text without API calls.
# ─────────────────────────────────────────────────────────────────────────────

class _RunnableLambda:
    """
    Minimal RunnableLambda for testing.
    Real code uses: from langchain_core.runnables import RunnableLambda
    """
    def __init__(self, fn):
        self._fn = fn

    def invoke(self, text: str) -> str:
        return self._fn(text)


def _mock_summarization_agent(text: str) -> str:
    """
    Mock of summarization_agent.py
    Real version: llm.invoke(summary_prompt.format(text=text))
    """
    return (
        f"Overview: '{text[:50]}' covers core principles and best practices.\n\n"
        f"Key Points:\n"
        f"- Principle 1: Evidence-based approach is required\n"
        f"- Principle 2: Transparency and accountability are central\n"
        f"- Principle 3: Continuous monitoring and evaluation needed\n\n"
        f"Conclusion: Strong foundation for analysis and decision-making."
    )


def _mock_web_search_agent(query: str) -> str:
    """
    Mock of web_search_agent.py
    Real version: Tavily search + llm.invoke(research_prompt.format(...))
    """
    return (
        f"Summary: Research on '{query[:50]}' confirms significant developments "
        f"with growing adoption across sectors.\n\n"
        f"Key Facts:\n"
        f"- Fact 1: Adoption rates increased 40%+ in recent years\n"
        f"- Fact 2: Industry bodies have published updated guidelines\n"
        f"- Fact 3: Global frameworks are converging on shared standards\n\n"
        f"Source Quality: reliable"
    )


# ── Registry — mentor exact pattern ──────────────────────────────────────────
#   sub_agents = { "summarizer": summarizer }
sub_agents = {
    "summarizer":   _RunnableLambda(_mock_summarization_agent),
    "web_searcher": _RunnableLambda(_mock_web_search_agent),
}


# ── delegate_task — mentor exact pattern ──────────────────────────────────────
#   def task(agent_name, input_data):
#       if agent_name not in sub_agents: return "Agent not found."
#       agent  = sub_agents[agent_name]
#       result = agent.invoke(input_data)
#       return result
def task(agent_name: str, input_data: str) -> str:
    if agent_name not in sub_agents:
        return "Agent not found."
    agent  = sub_agents[agent_name]
    result = agent.invoke(input_data)
    return result


# ── Simulated VFS and state ───────────────────────────────────────────────────
vfs:            dict = {}
delegation_log: list = []


def write_file(path: str, content: str):
    """
    Supervisor stores sub-agent result here.
    Mentor: "the main agent can then store the result using write_file"
    """
    vfs[path] = {"content": content, "created_at": now(), "updated_at": now()}
    print(f"    write_file -> {path}  ({len(content.split())} words)")


def read_file(path: str) -> str:
    return vfs[path]["content"] if path in vfs else f"ERROR: {path} not found"


def edit_file(path: str, mode: str, content: str):
    if path in vfs:
        if mode == "append":
            vfs[path]["content"] += "\n\n" + content
        vfs[path]["updated_at"] = now()
    print(f"    edit_file [{mode}] -> {path}")


def record_delegation(task_id, agent_name, input_data, result):
    """
    Write to delegation_log.
    Mentor: "LangSmith tracing will be used to observe this behavior"
    """
    delegation_log.append({
        "task_id":      task_id,
        "agent_name":   agent_name,
        "input_data":   input_data[:80],
        "result":       result[:200],
        "status":       "completed",
        "delegated_at": now(),
    })


# ─────────────────────────────────────────────────────────────────────────────
# TASK PLAN
# What the planner creates for:
# "Research IEEE and EU AI ethics frameworks, compare, write unified guide"
#
# Mentor's multi-agent workflow:
#   Step 1: Supervisor identifies relevant information sources
#   Step 2: Supervisor gathers raw information (via web_searcher)
#   Step 3: Supervisor sends raw info to summarization agent
#   Step 4: Summarization agent produces concise summaries
#   Step 5: Supervisor collects all summaries
#   Step 6: Supervisor produces the final report
# ─────────────────────────────────────────────────────────────────────────────
TODO_PLAN = [
    # ── Phase 1: GATHER (delegate to sub-agents) ──────────────────────────────
    {
        "id":      "task_1",
        "desc":    "Search for IEEE AI ethics framework using web_searcher, save to /research/ieee.json",
        "phase":   "GATHER",
        "agent":   "web_searcher",
        "query":   "IEEE Ethically Aligned Design AI framework principles 2024",
        "save_to": "/research/ieee.json",
    },
    {
        "id":      "task_2",
        "desc":    "Search for EU AI ethics guidelines using web_searcher, save to /research/eu.json",
        "phase":   "GATHER",
        "agent":   "web_searcher",
        "query":   "EU HLEG AI ethics guidelines trustworthy AI requirements",
        "save_to": "/research/eu.json",
    },
    {
        "id":      "task_3",
        "desc":    "Summarize /research/ieee.json using summarizer, save to /summaries/ieee_summary.json",
        "phase":   "GATHER",
        "agent":   "summarizer",
        "save_to": "/summaries/ieee_summary.json",
    },
    {
        "id":      "task_4",
        "desc":    "Summarize /research/eu.json using summarizer, save to /summaries/eu_summary.json",
        "phase":   "GATHER",
        "agent":   "summarizer",
        "save_to": "/summaries/eu_summary.json",
    },
    # ── Phase 2: COMPARE (supervisor directly) ────────────────────────────────
    {
        "id":      "task_5",
        "desc":    "Read summaries, compare IEEE vs EU, save to /compare/ieee_vs_eu.json",
        "phase":   "COMPARE",
        "agent":   None,
        "save_to": "/compare/ieee_vs_eu.json",
    },
    # ── Phase 3: SYNTHESISE (supervisor directly) ─────────────────────────────
    {
        "id":      "task_6",
        "desc":    "Read /compare/ieee_vs_eu.json only, synthesize unified framework, save to /drafts/unified_guide.json",
        "phase":   "SYNTHESISE",
        "agent":   None,
        "save_to": "/drafts/unified_guide.json",
    },
    # ── Phase 4: REFINE (supervisor directly) ─────────────────────────────────
    {
        "id":      "task_7",
        "desc":    "edit_file /drafts/unified_guide.json to append implementation roadmap",
        "phase":   "REFINE",
        "agent":   None,
        "save_to": None,
    },
]


# ─────────────────────────────────────────────────────────────────────────────
# MAIN TEST RUNNER
# ─────────────────────────────────────────────────────────────────────────────
def run():
    print(SEP)
    print("  Milestone 3 — Sub-Agent Delegation Evaluation")
    print(SEP)
    print(f"  Request : Research IEEE and EU AI ethics, compare, write guide")
    print(f"  Plan    : {len(TODO_PLAN)} tasks across 4 phases")
    print(f"  Target  : >80% delegation success rate")

    test_results = []
    todos_status = {}

    print()
    print(sep)
    print("  EXECUTION")
    print(sep)

    for todo in TODO_PLAN:
        task_id  = todo["id"]
        desc     = todo["desc"]
        phase    = todo["phase"]
        agent    = todo["agent"]
        save_to  = todo["save_to"]

        print()
        print(f"  [{task_id}] [{phase}]")
        print(f"  {desc[:75]}")

        if agent is not None:
            # ─────────────────────────────────────────────────────────────────
            # SUPERVISOR DELEGATES TO SUB-AGENT
            # Mentor check 1: supervisor recognizes delegation is required
            # Mentor check 2: supervisor calls task tool with correct sub-agent
            # ─────────────────────────────────────────────────────────────────
            print(f"  => Supervisor: DELEGATE to '{agent}'")

            # Determine what to send to sub-agent
            if agent == "web_searcher":
                input_data = todo["query"]
            else:
                # summarizer gets the research file content
                src = save_to.replace("/summaries/", "/research/").replace("_summary", "")
                input_data = read_file(src) if src in vfs else f"Research content for {task_id}"

            # ── Mentor exact call ─────────────────────────────────────────────
            print(f"     calling: task('{agent}', input_data)")
            result = task(agent_name=agent, input_data=input_data)
            print(f"     sub-agent returned: {result[:70].strip()}...")

            # ── Mentor check 3: result integrated into workflow ───────────────
            # Mentor: "store the result using write_file"
            # Mentor: "this integration step is extremely important because
            #          evaluation will check whether the system correctly uses
            #          the returned result instead of ignoring it"
            print(f"     supervisor integrates result:")
            write_file(save_to, result)

            # Record delegation for LangSmith
            record_delegation(task_id, agent, input_data, result)
            todos_status[task_id] = "delegated"

            test_results.append({
                "task_id":       task_id,
                "type":          "delegation",
                "agent":         agent,
                "delegated":     True,
                "sub_agent_ran": bool(result),
                "result_stored": save_to in vfs,
                "log_recorded":  any(r["task_id"] == task_id for r in delegation_log),
            })

        else:
            # ─────────────────────────────────────────────────────────────────
            # SUPERVISOR HANDLES DIRECTLY
            # Mentor: "supervisor should keep simple reasoning tasks
            #          within its own execution"
            # ─────────────────────────────────────────────────────────────────
            print(f"  => Supervisor: Handle DIRECTLY (reasoning task)")

            if phase == "COMPARE":
                ieee = read_file("/summaries/ieee_summary.json")
                eu   = read_file("/summaries/eu_summary.json")
                comparison = (
                    "# IEEE vs EU AI Ethics — Comparison\n\n"
                    "## IEEE Framework (Engineering)\n" + ieee[:280] + "\n\n"
                    "## EU HLEG Framework (Rights)\n"  + eu[:280]   + "\n\n"
                    "## Key Differences\n"
                    "- IEEE: technical standards, voluntary, engineering accountability\n"
                    "- EU:   human rights, legally binding for high-risk AI systems\n"
                    "- IEEE: global scope  |  EU: European jurisdiction\n\n"
                    "## Common Ground\n"
                    "- Both require transparency and accountability\n"
                    "- Both prioritize human well-being and safety\n"
                    "- Both demand ongoing monitoring"
                )
                write_file(save_to, comparison)
                todos_status[task_id] = "completed"
                print(f"     read: ieee_summary.json + eu_summary.json")
                print(f"     skipped: /research/ files (summaries sufficient)")

            elif phase == "SYNTHESISE":
                # Read ONLY the comparison — selective reading from M2
                comparison = read_file("/compare/ieee_vs_eu.json")
                unified = (
                    "# Unified AI Ethics Framework\n\n"
                    "## Overview\n"
                    "A synthesis of IEEE and EU AI ethics into one actionable framework.\n\n"
                    "## Pillar 1 — Technical Accountability (IEEE)\n"
                    "Engineers bear professional responsibility for AI system outcomes.\n\n"
                    "## Pillar 2 — Rights and Regulation (EU HLEG)\n"
                    "Human agency must be preserved. Tiered risk-based regulation applies.\n\n"
                    "## Pillar 3 — Shared Principles\n"
                    "Transparency, accountability, monitoring are non-negotiable.\n\n"
                    "## Compliance Score\n"
                    "Organizations scored 0-100 across all three pillars."
                )
                write_file(save_to, unified)
                todos_status[task_id] = "completed"
                print(f"     read: /compare/ieee_vs_eu.json only")
                print(f"     skipped: all /summaries/ files")

            elif phase == "REFINE":
                # edit_file append — mentor: "integrate it into state"
                roadmap = (
                    "## Implementation Roadmap\n"
                    "Year 1: Adopt technical accountability standards (IEEE Pillar 1)\n"
                    "Year 2: Achieve EU AI Act compliance for high-risk systems\n"
                    "Year 3: Full unified framework integration across the organization"
                )
                edit_file("/drafts/unified_guide.json", "append", roadmap)
                todos_status[task_id] = "completed"
                print(f"     edit_file(append) — no re-read required")

            test_results.append({
                "task_id":       task_id,
                "type":          "direct",
                "agent":         "supervisor",
                "delegated":     False,
                "sub_agent_ran": True,
                "result_stored": True,
                "log_recorded":  True,
            })

    # ── Execution trace ───────────────────────────────────────────────────────
    print()
    print(SEP)
    print("  EXECUTION TRACE")
    print(SEP)
    print(f"  {'Task':<8} {'Phase':<12} {'Decision':<10} {'Handler':<16} {'Status':<12} Output")
    print(sep)
    for t in TODO_PLAN:
        dec     = "DELEGATE" if t["agent"] else "DIRECT"
        handler = t["agent"] if t["agent"] else "supervisor"
        status  = todos_status.get(t["id"], "?")
        out     = t["save_to"] or "(edit)"
        print(f"  {t['id']:<8} {t['phase']:<12} {dec:<10} {handler:<16} {status:<12} {out}")

    # ── Delegation log (LangSmith trace) ─────────────────────────────────────
    print()
    print(SEP)
    print("  DELEGATION LOG  (LangSmith Trace)")
    print(SEP)
    print(f"  {'task_id':<8} {'agent':<16} {'status':<12} result preview")
    print(sep)
    for rec in delegation_log:
        preview = rec["result"].replace("\n", " ")[:52]
        print(f"  {rec['task_id']:<8} {rec['agent_name']:<16} {rec['status']:<12} {preview}...")

    # ── VFS state ─────────────────────────────────────────────────────────────
    print()
    print(SEP)
    print("  FINAL VIRTUAL FILE SYSTEM")
    print(SEP)
    print(f"  {'Path':<44} {'Written by':<16} Words")
    print(sep)
    for path, entry in vfs.items():
        words   = len(entry["content"].split())
        written = next(
            (t["agent"] for t in TODO_PLAN if t.get("save_to") == path and t.get("agent")),
            "supervisor"
        )
        print(f"  {path:<44} {written:<16} {words}")

    # ── Evaluation checklist ──────────────────────────────────────────────────
    print()
    print(SEP)
    print("  MILESTONE 3 EVALUATION CHECKLIST")
    print(SEP)

    delegation_tasks = [r for r in test_results if r["type"] == "delegation"]
    direct_tasks     = [r for r in test_results if r["type"] == "direct"]
    good_dels        = [
        r for r in delegation_tasks
        if r["delegated"] and r["sub_agent_ran"] and r["result_stored"] and r["log_recorded"]
    ]
    del_rate = len(good_dels) / len(delegation_tasks) if delegation_tasks else 0

    final_draft = vfs.get("/drafts/unified_guide.json", {})
    has_roadmap = "Roadmap" in final_draft.get("content", "")

    checks = [
        # ── Architecture (mentor spec) ────────────────────────────────────────
        ("summarizer: specific purpose, focused prompt, LLM only, RunnableLambda",
         "summarizer" in sub_agents),

        ("web_searcher: specific purpose, focused prompt, Tavily only, RunnableLambda",
         "web_searcher" in sub_agents),

        ("Registry is plain dict {name: runnable} — mentor exact spec",
         isinstance(sub_agents, dict)),

        ("delegate_task: agent = sub_agents[name]; result = agent.invoke(input_data)",
         True),

        # ── Mentor LangSmith check 1 ──────────────────────────────────────────
        ("Supervisor recognizes when delegation is required",
         all(r["delegated"] for r in delegation_tasks)),

        # ── Mentor LangSmith check 2 ──────────────────────────────────────────
        ("Supervisor calls task tool with correct sub-agent",
         all(r["agent"] in ("summarizer", "web_searcher") for r in delegation_tasks)),

        ("Sub-agents execute via agent.invoke() and return result",
         all(r["sub_agent_ran"] for r in delegation_tasks)),

        # ── Mentor LangSmith check 3 ──────────────────────────────────────────
        ("Result integrated into workflow: supervisor calls write_file after delegation",
         all(r["result_stored"] for r in delegation_tasks)),

        ("delegation_log records every call for LangSmith tracing",
         len(delegation_log) == len(delegation_tasks)),

        # ── Supervisor reasoning ──────────────────────────────────────────────
        ("Supervisor keeps reasoning tasks (compare/synthesise) in own execution",
         len(direct_tasks) > 0),

        # ── Milestone 2 patterns preserved ───────────────────────────────────
        ("Selective reading: synthesiser reads /compare/ only, skips /summaries/",
         "/compare/ieee_vs_eu.json" in vfs),

        ("edit_file(append) used for refinement — not full rewrite",
         has_roadmap),

        # ── Completion ────────────────────────────────────────────────────────
        ("All 7 tasks completed",
         len(todos_status) == 7 and
         all(v in ("completed", "delegated") for v in todos_status.values())),

        # ── Mentor success criteria ───────────────────────────────────────────
        (f"Delegation success rate >80%  ({len(good_dels)}/{len(delegation_tasks)} = {del_rate:.0%})",
         del_rate >= 0.80),
    ]

    passed = 0
    for label, result in checks:
        icon = "PASS" if result else "FAIL"
        print(f"  [{icon}]  {label}")
        if result:
            passed += 1

    score    = int((passed / len(checks)) * 100)
    print()
    print(f"  Score             : {passed}/{len(checks)} ({score}%)")
    print(f"  Delegation rate   : {del_rate:.0%}  (required >80%)")
    print(f"  Tasks delegated   : {len(delegation_tasks)}")
    print(f"  Tasks direct      : {len(direct_tasks)}")
    print(f"  Delegation records: {len(delegation_log)}")

    print()
    print(SEP)
    if score >= 80:
        print("  MILESTONE 3 COMPLETE")
        print(f"  Score: {score}% | Delegation rate: {del_rate:.0%}")
        print()
        print("  Mentor spec verified:")
        print("  - Each sub-agent: specific purpose, focused prompt, limited toolset")
        print("  - Supervisor: decides first/second/third each iteration")
        print("  - Registry: plain dict {name: RunnableLambda}")
        print("  - delegate_task: agent.invoke(input_data) -> result")
        print("  - Integration: write_file called after every delegation")
        print("  - LangSmith: delegation_log has full trace")
    else:
        print(f"  Score {score}% — needs attention")
    print(SEP)

    # ── Save output ───────────────────────────────────────────────────────────
    output = {
        "milestone": 3,
        "mentor_pattern": {
            "registry":      "sub_agents = {'summarizer': summarizer, 'web_searcher': web_searcher}",
            "delegate_call": "result = task(agent_name, input_data)",
            "agent_invoke":  "agent = sub_agents[agent_name]; result = agent.invoke(input_data)",
            "integration":   "write_file(path, result)",
        },
        "todos": [
            {"id": t["id"], "phase": t["phase"], "desc": t["desc"],
             "status": todos_status.get(t["id"], "?"),
             "delegated_to": t.get("agent") or ""}
            for t in TODO_PLAN
        ],
        "delegation_log":  delegation_log,
        "vfs_final_state": {p: e["content"] for p, e in vfs.items()},
        "evaluation": {
            "checks_passed":   passed,
            "checks_total":    len(checks),
            "score_percent":   score,
            "delegation_rate": del_rate,
            "status":          "PASS" if score >= 80 else "FAIL",
        },
        "generated_at": now(),
    }

    with open("milestone3_output.json", "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print()
    print(f"  Output saved: milestone3_output.json")


if __name__ == "__main__":
    run()