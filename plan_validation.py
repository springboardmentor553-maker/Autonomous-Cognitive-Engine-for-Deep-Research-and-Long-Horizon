"""
Plan Validation: Unified Workflow (Milestone 1 + Milestone 2) — OpenAI
Tests planning + file-based context offloading across 5 task scenarios.
Success Criteria: >80% pass rate.
"""
import os
import sys
import time
import shutil
from pathlib import Path
from langchain_core.messages import HumanMessage
from workflow.flow import create_agent_executor, create_system_prompt
from brains.filetools import clear_virtual_fs, FILE_SYSTEM_DIR, get_fs_stats

os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"]    = "milestone2-multi-prompt"

OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

print(f"✓ OpenAI Model: {OPENAI_MODEL}")
print(f"✓ LangSmith tracing ENABLED — project: milestone2-multi-prompt")
print(f"✓ Storage: {FILE_SYSTEM_DIR.absolute()}\n")

# ── Test Scenarios (unchanged from original) ──────────────────────────────────
TEST_SCENARIOS = [
    {
        "name": "Country Cultures",
        "task": """Analyze the cultures of Germany, India, and Japan using file storage.
... (task text unchanged) ...
YOUR TASK - Create EXACTLY 5 steps:
Step 1: Summarize German culture (100-150 words) → "germany_culture.txt"
Step 2: Summarize Indian culture (100-150 words) → "india_culture.txt"
Step 3: Summarize Japanese culture (100-150 words) → "japan_culture.txt"
Step 4: Read all 3 culture files using read_file()
Step 5: Create comparative analysis → "final_comparison.txt"
""",
        "expected_files": ["germany_culture.txt", "india_culture.txt", "japan_culture.txt", "final_comparison.txt"],
        "min_write_ops": 4, "min_read_ops": 3
    },
    {
        "name": "AI Frameworks",
        "task": """Analyze and compare TensorFlow, PyTorch, and JAX.
YOUR TASK - Create EXACTLY 5 steps:
Step 1: Summarize TensorFlow → "tensorflow_summary.txt"
Step 2: Summarize PyTorch → "pytorch_summary.txt"
Step 3: Summarize JAX → "jax_summary.txt"
Step 4: Read all 3 framework files
Step 5: Create comparison report → "framework_comparison.txt"
""",
        "expected_files": ["tensorflow_summary.txt", "pytorch_summary.txt", "jax_summary.txt", "framework_comparison.txt"],
        "min_write_ops": 4, "min_read_ops": 3
    },
    {
        "name": "Climate Regions",
        "task": """Analyze climate change impacts on Arctic, Amazon, and Sahara.
YOUR TASK - Create EXACTLY 5 steps:
Step 1: Summarize Arctic impacts → "arctic_summary.txt"
Step 2: Summarize Amazon impacts → "amazon_summary.txt"
Step 3: Summarize Sahara impacts → "sahara_summary.txt"
Step 4: Read all 3 regional files
Step 5: Create global analysis → "climate_analysis.txt"
""",
        "expected_files": ["arctic_summary.txt", "amazon_summary.txt", "sahara_summary.txt", "climate_analysis.txt"],
        "min_write_ops": 4, "min_read_ops": 3
    },
    {
        "name": "Programming Languages",
        "task": """Compare Python, JavaScript, and Go for web development.
YOUR TASK - Create EXACTLY 5 steps:
Step 1: Summarize Python → "python_summary.txt"
Step 2: Summarize JavaScript → "javascript_summary.txt"
Step 3: Summarize Go → "go_summary.txt"
Step 4: Read all 3 language files
Step 5: Create comparison report → "language_comparison.txt"
""",
        "expected_files": ["python_summary.txt", "javascript_summary.txt", "go_summary.txt", "language_comparison.txt"],
        "min_write_ops": 4, "min_read_ops": 3
    },
    {
        "name": "Historical Revolutions",
        "task": """Compare the French, American, and Russian Revolutions.
YOUR TASK - Create EXACTLY 5 steps:
Step 1: Summarize French Revolution → "french_revolution.txt"
Step 2: Summarize American Revolution → "american_revolution.txt"
Step 3: Summarize Russian Revolution → "russian_revolution.txt"
Step 4: Read all 3 revolution files
Step 5: Create comparative analysis → "revolution_comparison.txt"
""",
        "expected_files": ["french_revolution.txt", "american_revolution.txt", "russian_revolution.txt", "revolution_comparison.txt"],
        "min_write_ops": 4, "min_read_ops": 3
    },
]


def analyze_tool_sequence(messages):
    sequence = []
    for msg in messages:
        if hasattr(msg, "tool_calls") and msg.tool_calls:
            for tc in msg.tool_calls:
                name = tc.get("name", "unknown")
                args = tc.get("args", {})
                sequence.append({
                    "tool":     name,
                    "filename": args.get("filename", "?"),
                    "content_length": len(args.get("content", "")),
                    "action":   f"{name.upper()}: {args.get('filename', '?')}"
                })
    return sequence


def validate_scenario_result(scenario_name, result, expected_files, min_write_ops, min_read_ops):
    todos    = result.get("todos", [])
    messages = result.get("messages", [])
    sequence = analyze_tool_sequence(messages)

    write_ops = [s for s in sequence if s["tool"] == "write_file"]
    read_ops  = [s for s in sequence if s["tool"] == "read_file"]

    files_created = set()
    if FILE_SYSTEM_DIR.exists():
        files_created = {f.name for f in FILE_SYSTEM_DIR.iterdir() if f.is_file()}

    checks = {
        "planning_completed":    "write_todos" in [s["tool"] for s in sequence],
        "exactly_5_todos":       len(todos) == 5,
        "write_file_used":       len(write_ops) >= min_write_ops,
        "read_file_used":        len(read_ops) >= min_read_ops,
        "expected_files_created":all(f in files_created for f in expected_files),
        "meaningful_filenames":  not any(
            f in fname.lower() for fname in files_created
            for f in ["file1", "file2", "data", "temp"]
        ),
    }
    passed = sum(1 for v in checks.values() if v)
    return {
        "scenario": scenario_name, "checks": checks,
        "passed": passed, "total": len(checks),
        "score": passed / len(checks) * 100,
        "files_created": len(files_created),
        "write_ops": len(write_ops), "read_ops": len(read_ops),
    }


def run_plan_validation():
    print("=" * 100)
    print("PLAN VALIDATION: MULTI-PROMPT TEST (MILESTONE 1 + 2)")
    print("=" * 100)
    print(f"Testing {len(TEST_SCENARIOS)} scenarios | Success Criteria: >80% pass rate\n")

    FILE_SYSTEM_DIR.mkdir(exist_ok=True)
    agent         = create_agent_executor()
    system_prompt = create_system_prompt()
    results       = []

    for i, scenario in enumerate(TEST_SCENARIOS, 1):
        print(f"\n{'='*100}\nSCENARIO {i}/{len(TEST_SCENARIOS)}: {scenario['name']}\n{'='*100}")
        clear_virtual_fs()

        try:
            result = agent.invoke(
                {"messages": [HumanMessage(content=f"{system_prompt}\n\n{scenario['task']}")]},
                {"configurable": {"thread_id": f"scenario-{i}"}, "recursion_limit": 50}
            )
            validation = validate_scenario_result(
                scenario['name'], result,
                scenario['expected_files'], scenario['min_write_ops'], scenario['min_read_ops']
            )
            results.append(validation)
            icon = "🎉" if validation['score'] >= 80 else "⚠️"
            print(f"\n{icon} SCORE: {validation['passed']}/{validation['total']} ({validation['score']:.1f}%)")
        except Exception as e:
            print(f"\n❌ ERROR: {e}")
            results.append({"scenario": scenario['name'], "score": 0, "passed": 0,
                             "total": 6, "error": str(e)})

        time.sleep(1)

    # Summary
    passed_scenarios = sum(1 for r in results if r['score'] >= 80)
    overall          = passed_scenarios / len(results) * 100

    print(f"\n{'='*100}\nOVERALL RESULTS\n{'='*100}")
    print(f"\nScenarios Passed: {passed_scenarios}/{len(results)}  |  Success Rate: {overall:.1f}%\n")
    print(f"{'Scenario':<25} {'Score':<10} {'Files':<8} {'Write':<8} {'Read':<8} Status")
    print("-" * 80)
    for r in results:
        status = "✅ PASS" if r['score'] >= 80 else "❌ FAIL"
        print(f"{r['scenario']:<25} {r['score']:.1f}%{'':<5} {r.get('files_created',0):<8}"
              f"{r.get('write_ops',0):<8} {r.get('read_ops',0):<8} {status}")

    print(f"\n{'='*100}")
    if overall >= 80:
        print("🎉 MILESTONE 2 VALIDATION: PASSED ✓")
    else:
        print("⚠️  MILESTONE 2 VALIDATION: NEEDS IMPROVEMENT")
    print(f"{'='*100}\n")
    return overall >= 80


if __name__ == "__main__":
    success = run_plan_validation()
    sys.exit(0 if success else 1)
