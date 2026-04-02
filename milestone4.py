import os
import json
import time
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

from dotenv import load_dotenv

from workflow import build_agent, run_task
from milestone4_eval import evaluate_run, print_eval_report

load_dotenv()

os.environ["LANGCHAIN_TRACING_V2"] = os.getenv("LANGCHAIN_TRACING_V2", "true")
os.environ["LANGCHAIN_API_KEY"]     = os.getenv("LANGCHAIN_API_KEY", "")
os.environ["LANGCHAIN_PROJECT"]     = os.getenv("LANGCHAIN_PROJECT", "milestone_4_integration")

TEST_INPUTS = [
    "Generate a research report on the impact of artificial intelligence in healthcare diagnostics.",
    "Research quantum computing applications in cybersecurity and produce a structured summary report.",
    "Investigate the current state of climate change policies globally and generate a comprehensive report.",
    "Research the role of blockchain technology in supply chain management and summarize key findings.",
    "Generate a report on recent advancements in renewable energy storage technologies.",
]


def run_evaluation():

    agent = build_agent()

    print(f"\n{'=' * 65}")
    print(f"MILESTONE 4 — FULL INTEGRATION & USE CASE EXECUTION")
    print(f"Running {len(TEST_INPUTS)} end-to-end test cases")
    print(f"Success threshold: >70% (4 out of 5)")
    print(f"{'=' * 65}\n")

    passed_count = 0
    all_reports  = []

    for i, test_task in enumerate(TEST_INPUTS):

        print(f"\nTEST {i+1}/{len(TEST_INPUTS)}: {test_task[:70]}...")
        print("-" * 65)

        try:
            # ── Retry on tool_use_failed ──────────────────────
            # Llama occasionally generates malformed JSON for task()
            # Retrying once usually resolves it
            max_retries = 2
            run_result  = None

            for attempt in range(max_retries):
                try:
                    run_result = run_task(agent, test_task)
                    break
                except Exception as e:
                    if "tool_use_failed" in str(e) and attempt < max_retries - 1:
                        print(f"⚠️  tool_use_failed — retrying in 30 seconds...")
                        time.sleep(30)
                    else:
                        raise e

            if run_result is None:
                raise Exception("All retry attempts failed")

            # ── Execution summary ─────────────────────────────
            delegations = run_result["delegation_log"]
            files       = run_result["files"].get("root", {})

            print(f"\n📋 TODOs created   : {len(run_result['todos'])}")
            print(f"🤝 Delegations     : {len(delegations)}")
            for d in delegations:
                icon = "✅" if d["status"] == "success" else "❌"
                print(f"   {icon} → {d['agent']} : {d['result'][:80]}...")
            print(f"📁 Files stored    : {list(files.keys())}")

            final = run_result["final_output"]
            if final:
                print(f"\n💬 Final output snippet:\n{final[:300]}...")

            # ── Evaluate ──────────────────────────────────────
            report = evaluate_run(run_result)
            print_eval_report(report, i + 1, test_task)

            # ── Save ──────────────────────────────────────────
            os.makedirs("outputs", exist_ok=True)
            fname = f"outputs/m4_test_{i+1}.json"
            with open(fname, "w") as f:
                json.dump({
                    "task":           test_task,
                    "todos":          run_result["todos"],
                    "delegation_log": run_result["delegation_log"],
                    "files_created":  list(files.keys()),
                    "final_output":   run_result["final_output"][:500],
                    "evaluation":     report
                }, f, indent=2)
            print(f"💾 Saved to {fname}")

            if report["passed"]:
                passed_count += 1

            all_reports.append(report)

        except Exception as e:
            print(f"❌ CRITICAL ERROR in Test {i+1}: {str(e)}")

        print("-" * 65)

        if i < len(TEST_INPUTS) - 1:
            print(f"Waiting 60 seconds to avoid rate limit...")
            time.sleep(60)

    # ── Final summary ─────────────────────────────────────────
    total   = len(TEST_INPUTS)
    pct     = (passed_count / total) * 100
    passing = pct >= 70

    avg_quality = (
        sum(r["quality_score"] for r in all_reports) / len(all_reports)
        if all_reports else 0
    )
    avg_delegations = (
        sum(r["delegations_made"] for r in all_reports) / len(all_reports)
        if all_reports else 0
    )

    print(f"\n{'=' * 65}")
    print(f"MILESTONE 4 FINAL RESULT")
    print(f"{'=' * 65}")
    print(f"Tests Passed         : {passed_count}/{total} ({pct:.0f}%)")
    print(f"Required             : >70% (4 out of 5)")
    print(f"Avg Output Quality   : {avg_quality:.1f}/5")
    print(f"Avg Delegations/Test : {avg_delegations:.1f}")
    print(f"\nCriteria Breakdown:")

    for label, key in [
        ("Task Completion",  "criterion_1_task_completion"),
        ("Delegation",       "criterion_2_delegation"),
        ("Memory Usage",     "criterion_3_memory_usage"),
        ("Output Quality",   "criterion_4_output_quality"),
    ]:
        count = sum(1 for r in all_reports if r.get(key))
        print(f"  {label:20} : {count}/{total} passed")

    print(f"\n{'✅ MILESTONE 4 COMPLETE' if passing else '❌ MILESTONE 4 INCOMPLETE'}")
    print(f"{'=' * 65}\n")

    os.makedirs("outputs", exist_ok=True)
    with open("outputs/m4_summary.json", "w") as f:
        json.dump({
            "total_tests":        total,
            "passed":             passed_count,
            "pass_rate":          f"{pct:.0f}%",
            "avg_quality_score":  round(avg_quality, 1),
            "milestone_complete": passing,
            "all_reports":        all_reports
        }, f, indent=2)
    print("📊 Full summary saved to outputs/m4_summary.json")


if __name__ == "__main__":
    run_evaluation()