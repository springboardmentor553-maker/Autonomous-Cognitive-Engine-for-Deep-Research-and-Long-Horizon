"""
Milestone 2 — Main Runner

Executes all 7 Milestone 2 tasks sequentially, saves JSON outputs,
and prints a summary report at the end.

Usage:
    python main.py
"""

import os
import sys
import time
import json
from dotenv import load_dotenv

load_dotenv()

from app import create_milestone2_agent, run_agent, save_result_to_json, display_result
from tasks import TASKS


DELAY_BETWEEN_TASKS = 65   # seconds — avoids Gemini rate limits
OUTPUT_DIR = "outputs"


def run_all_tasks():
    print("=" * 65)
    print("  MILESTONE 2 — ReAct Agent with Virtual File System")
    print("=" * 65)

    agent = create_milestone2_agent()
    results_summary = []

    for i, task_def in enumerate(TASKS):
        task_id    = task_def["id"]
        task_label = task_def["label"]
        task_desc  = task_def["description"]

        print(f"\n{'━'*65}")
        print(f"  [{i+1}/{len(TASKS)}] {task_label}")
        print(f"{'━'*65}")

        thread_id = f"m2-{task_id}"
        start_time = time.time()

        try:
            result = run_agent(agent, task_desc, thread_id=thread_id)
            elapsed = round(time.time() - start_time, 1)

            # Display result in terminal
            display_result(result)

            # Save JSON output
            json_filename = f"{task_id}_output.json"
            save_result_to_json(result, json_filename, output_dir=OUTPUT_DIR)

            # Collect summary row
            results_summary.append({
                "id": task_id,
                "label": task_label,
                "status": "✅ OK",
                "todos": len(result["todos"]),
                "files": len(result["files"]),
                "time_s": elapsed,
            })

        except Exception as e:
            elapsed = round(time.time() - start_time, 1)
            print(f"\n  ❌ ERROR in {task_id}: {e}")
            results_summary.append({
                "id": task_id,
                "label": task_label,
                "status": f"❌ ERROR: {e}",
                "todos": 0,
                "files": 0,
                "time_s": elapsed,
            })

        # Rate limit buffer between tasks (skip after last task)
        if i < len(TASKS) - 1:
            print(f"\n  ⏳ Waiting {DELAY_BETWEEN_TASKS}s before next task...")
            time.sleep(DELAY_BETWEEN_TASKS)

    # ── Final summary report ──────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("  MILESTONE 2 — FINAL SUMMARY REPORT")
    print("=" * 65)
    print(f"  {'ID':<10} {'TODOs':>5} {'Files':>5} {'Time':>7}  Status")
    print(f"  {'─'*10} {'─'*5} {'─'*5} {'─'*7}  {'─'*30}")
    for row in results_summary:
        print(
            f"  {row['id']:<10} {row['todos']:>5} {row['files']:>5} "
            f"{row['time_s']:>6.1f}s  {row['status']}"
        )

    success = sum(1 for r in results_summary if r["status"].startswith("✅"))
    total   = len(results_summary)
    print(f"\n  Passed: {success}/{total}  ({round(success/total*100)}%)")

    # Save summary JSON
    summary_path = os.path.join(OUTPUT_DIR, "milestone2_summary.json")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(results_summary, f, indent=2, ensure_ascii=False)
    print(f"\n  Summary saved → {summary_path}")

    print("\n  Check LangSmith for full traces.")
    print("=" * 65)


if __name__ == "__main__":
    run_all_tasks()
