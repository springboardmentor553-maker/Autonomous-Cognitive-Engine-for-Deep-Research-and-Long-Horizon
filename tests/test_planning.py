"""
Test suite for the Planning Agent (Milestone 1)
Strict verification of planning-first behavior.
"""
import time
import os
import sys
import json
import argparse
from datetime import datetime

# ============================================================
# PATH CONFIGURATION
#import os
import sys

# Get the path to 'deep_cognitive_agent' (one level up from 'tests')
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# Add it to the path
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Now the import will work
from app import create_planning_agent, run_agent


# ============================================================
# TEST INPUTS (MENTOR REQUIRED SET)
# ============================================================

TEST_INPUTS = [
    "Create a research outline for renewable energy trends",
    "Design a structured learning roadmap for data science",
    "Break down the steps for developing a web application",
    "Plan a comparative study between electric and hydrogen vehicles",
    "Create a technical writing outline for AI ethics",
    "Design an autonomous AI research agent architecture",
    "Create a scalable cloud deployment plan for a web platform",
    "Develop a roadmap for implementing cybersecurity policies in a company.",
    "Create a structured plan for launching a new mobile application.",
    "Design a detailed research strategy for studying climate change impact.",
    "Break down the process of building and deploying a machine learning system.",
    "Prepare a comparative analysis structure between electric and hydrogen vehicles.",
    "Design a roadmap for scaling a SaaS platform.",
    "Create a structured content plan for technical documentation.",
    "Plan the full lifecycle of building a data pipeline.",
    "Develop a structured approach for conducting market analysis.",
    "Design a research breakdown for analyzing AI ethics challenges."
]



# ============================================================
# SAVE OUTPUT
# ============================================================

def save_test_output(test_number, input_text, result):
    output_dir = "test_results"
    os.makedirs(output_dir, exist_ok=True)

    filename = f"test_{test_number}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    filepath = os.path.join(output_dir, filename)

    output_data = {
        "test_number": test_number,
        "input": input_text,
        "timestamp": datetime.now().isoformat(),
        "result": result, 
    }

    with open(filepath, "w", encoding="utf-8") as f:
        # ✅ Added default=str to prevent 'HumanMessage is not JSON serializable'
        json.dump(output_data, f, indent=4, ensure_ascii=False, default=str)

    return filepath


# ============================================================
# VALIDATION (STRICT — mentor rules)
# ============================================================

def validate_todos(result):
    if not isinstance(result, dict):
        return False, "Result not dictionary"

    todos = result.get("todos")

    if not isinstance(todos, list):
        return False, "Todos not list"

    # ✅ EXACT FIVE STEPS (mentor requirement)
    if len(todos) != 5:
        return False, f"Expected 5 steps, got {len(todos)}"

    for todo in todos:
        if not isinstance(todo, dict):
            return False, "Todo not dict"

        if "task" not in todo or "status" not in todo:
            return False, "Missing required fields"

    return True, "Valid 5-step structured plan"


# ============================================================
# TOOL INVOCATION CHECK
# ============================================================

def tool_was_called(result):
    messages = result.get("messages", [])

    for msg in messages:
        # 1. Check if the message has a 'name' property (ToolMessage)
        # We check both the function name and the decorated name
        msg_name = getattr(msg, "name", None)
        if msg_name in ["write_todos", "write_todos_tool"]:
            return True
            
        # 2. Check if the AI message contains 'tool_calls' (AIMessage)
        if hasattr(msg, "tool_calls") and msg.tool_calls:
            for tc in msg.tool_calls:
                if tc.get("name") in ["write_todos", "write_todos_tool"]:
                    return True

    return False

# ============================================================
# RUN SINGLE TEST (3 RUNS EACH — mentor rule)
# ============================================================

def run_single_test(test_number, input_text, agent):
    print("=" * 60)
    print(f"TEST {test_number}/7")
    print("=" * 60)
    print(f"Task: {input_text}\n")

    success_runs = 0

    for run in range(3):  # ✅ mentor requirement
        print(f"Run {run+1}")
        time.sleep(5)
        result = run_agent(agent, input_text)

        valid, message = validate_todos(result)
        tool_called = tool_was_called(result)

        if valid and tool_called:
            print("SUCCESS:", message)
            success_runs += 1
        else:
            print("FAILED")
            print(f"Validation: {message}")
            print(f"Tool invoked: {tool_called}")

    filepath = save_test_output(test_number, input_text, result)
    print(f"\nSaved → {filepath}\n")

    return success_runs == 3


# ============================================================
# RUN ALL TESTS
# ============================================================

def run_all_tests():
    print("\nMILESTONE 1 — STRICT PLANNING VALIDATION\n")

    agent = create_planning_agent()
    success_count = 0

    for i, task in enumerate(TEST_INPUTS, 1):
        success = run_single_test(i, task, agent)
        if success:
            success_count += 1

    total = len(TEST_INPUTS)
    accuracy = (success_count / total) * 100

    print("=" * 60)
    print("FINAL SUMMARY")
    print("=" * 60)

    print(f"Successful Tasks: {success_count}/{total}")
    print(f"Accuracy: {accuracy:.2f}%")

    if accuracy >= 80:
        print("✅ Mentor Requirement PASSED")
    else:
        print("❌ Needs refinement (<80%)")


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", type=int, choices=range(1, 8))
    parser.add_argument("--all", action="store_true")

    args = parser.parse_args()

    # Create agent once
    agent = create_planning_agent()

    if args.test:
        run_single_test(args.test, TEST_INPUTS[args.test - 1], agent)
    else:
        run_all_tests()


if __name__ == "__main__":
    main()