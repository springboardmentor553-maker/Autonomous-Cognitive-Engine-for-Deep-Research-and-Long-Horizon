# """
# Test Planning Agent - Milestone 1

# This script tests the ReAct planning agent with 5 complex inputs
# and saves the generated todos to outputs/*.json files.

# Test Inputs:
# 1. Create a research outline for renewable energy trends
# 2. Design a structured learning roadmap for data science
# 3. Break down the steps for developing a web application
# 4. Plan a comparative study between electric and hydrogen vehicles
# 5. Create a technical writing outline for AI ethics
# """

# import os
# import sys
# import json
# from datetime import datetime

# # Add parent directory to path for imports
# sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# from dotenv import load_dotenv

# # Load environment variables
# load_dotenv()

# # Enable LangSmith Tracing
# os.environ["LANGCHAIN_TRACING_V2"] = "true"
# os.environ["LANGCHAIN_PROJECT"] = os.getenv("LANGCHAIN_PROJECT", "milestone_1_planning")

# from app import create_planning_agent, run_agent, save_result_to_json


# # Test inputs — 5 complex tasks as specified in Milestone 1
# TEST_INPUTS = [
#     "Create a research outline for renewable energy trends",
#     "Design a structured learning roadmap for data science",
#     "Break down the steps for developing a web application",
#     "Plan a comparative study between electric and hydrogen vehicles",
#     "Create a technical writing outline for AI ethics",
# ]


# def run_all_tests():
#     """
#     Run the planning agent on all test inputs and save results.
#     """
#     print("=" * 70)
#     print("MILESTONE 1 - PLANNING AGENT TEST SUITE")
#     print(f"Timestamp: {datetime.now().isoformat()}")
#     print("=" * 70)
    
#     # Create the agent once
#     print("\nInitializing Planning Agent...")
#     agent = create_planning_agent()
#     print("Agent initialized successfully!\n")
    
#     # Results summary
#     all_results = []
    
#     # Run each test
#     for i, task in enumerate(TEST_INPUTS, 1):
#         print("-" * 70)
#         print(f"TEST {i}/5: {task}")
#         print("-" * 70)
        
#         try:
#             # Run agent with unique thread ID
#             thread_id = f"test-{i}-{datetime.now().strftime('%Y%m%d%H%M%S')}"
#             result = run_agent(agent, task, thread_id=thread_id)
            
#             # Display generated todos
#             print(f"\nGenerated {len(result['todos'])} TODOs:")
#             for j, todo in enumerate(result["todos"], 1):
#                 status_icon = "⬜" if todo["status"] == "pending" else "✅"
#                 print(f"  {j}. {status_icon} {todo['task']}")
            
#             # Save to JSON
#             filename = f"test_{i}_{task.lower().replace(' ', '_')[:30]}.json"
#             filepath = save_result_to_json(result, filename)
            
#             # Track result
#             all_results.append({
#                 "test_number": i,
#                 "task": task,
#                 "todo_count": len(result["todos"]),
#                 "output_file": filepath,
#                 "success": True
#             })
            
#             print(f"\n✅ Test {i} completed successfully!")
            
#         except Exception as e:
#             print(f"\n❌ Test {i} failed with error: {str(e)}")
#             all_results.append({
#                 "test_number": i,
#                 "task": task,
#                 "todo_count": 0,
#                 "output_file": None,
#                 "success": False,
#                 "error": str(e)
#             })
        
#         print()
    
#     # Print summary
#     print("=" * 70)
#     print("TEST SUMMARY")
#     print("=" * 70)
    
#     successful = sum(1 for r in all_results if r["success"])
#     print(f"\nTotal Tests: {len(TEST_INPUTS)}")
#     print(f"Successful: {successful}")
#     print(f"Failed: {len(TEST_INPUTS) - successful}")
    
#     print("\nDetailed Results:")
#     for r in all_results:
#         status = "✅ PASS" if r["success"] else "❌ FAIL"
#         print(f"  {r['test_number']}. {status} - {r['task'][:40]}... ({r['todo_count']} todos)")
    
#     # Save summary
#     summary_file = os.path.join("outputs", "test_summary.json")
#     os.makedirs("outputs", exist_ok=True)
#     with open(summary_file, 'w', encoding='utf-8') as f:
#         json.dump({
#             "timestamp": datetime.now().isoformat(),
#             "total_tests": len(TEST_INPUTS),
#             "successful": successful,
#             "failed": len(TEST_INPUTS) - successful,
#             "results": all_results
#         }, f, indent=2)
    
#     print(f"\nSummary saved to: {summary_file}")
#     print("\n" + "=" * 70)
#     print("Check LangSmith dashboard for detailed traces:")
#     print(f"Project: {os.getenv('LANGCHAIN_PROJECT', 'Milestone1-Planning')}")
#     print("=" * 70)
    
#     return all_results


# def run_single_test(test_number: int):
#     """
#     Run a single test by number (1-5).
#     """
#     if test_number < 1 or test_number > len(TEST_INPUTS):
#         print(f"Invalid test number. Please choose 1-{len(TEST_INPUTS)}")
#         return
    
#     task = TEST_INPUTS[test_number - 1]
#     print(f"\nRunning single test: {task}")
#     print("-" * 50)
    
#     agent = create_planning_agent()
#     result = run_agent(agent, task, thread_id=f"single-test-{test_number}")
    
#     print(f"\nGenerated TODOs:")
#     for i, todo in enumerate(result["todos"], 1):
#         print(f"  {i}. {todo['task']} [{todo['status']}]")
    
#     filename = f"single_test_{test_number}.json"
#     save_result_to_json(result, filename)
    
#     return result


# if __name__ == "__main__":
#     import argparse
    
#     parser = argparse.ArgumentParser(description="Test the Planning Agent")
#     parser.add_argument(
#         "--test", 
#         type=int, 
#         choices=[1, 2, 3, 4, 5],
#         help="Run a single test by number (1-5)"
#     )
#     parser.add_argument(
#         "--all",
#         action="store_true",
#         help="Run all tests"
#     )
    
#     args = parser.parse_args()
    
#     if args.test:
#         run_single_test(args.test)
#     else:
#         # Default: run all tests
#         run_all_tests()
"""
Test suite for the Planning Agent (Milestone 1)
Verifies that the agent creates structured TODO plans before execution.
"""

import os
import sys
import json
import argparse
from datetime import datetime

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Import agent functions
from app import create_planning_agent, run_agent


# ============================================================
# TEST INPUTS (NOW 7 TEST CASES AS REQUIRED BY MENTOR)
# ============================================================

TEST_INPUTS = [
    "Create a research outline for renewable energy trends",
    "Design a structured learning roadmap for data science",
    "Break down the steps for developing a web application",
    "Plan a comparative study between electric and hydrogen vehicles",
    "Create a technical writing outline for AI ethics",

    # Added to meet mentor requirement of 7 complex prompts
    "Design an autonomous AI research agent architecture",
    "Create a scalable cloud deployment plan for a web platform"
]


# ============================================================
# SAVE OUTPUT FUNCTION
# ============================================================

# def save_test_output(test_number, input_text, result):
#     """Save test output to JSON file"""

#     output_dir = "test_results"
#     os.makedirs(output_dir, exist_ok=True)

#     filename = f"test_{test_number}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
#     filepath = os.path.join(output_dir, filename)

#     output_data = {
#         "test_number": test_number,
#         "input": input_text,
#         "timestamp": datetime.now().isoformat(),
#         "result": result
#     }

#     with open(filepath, "w", encoding="utf-8") as f:
#         json.dump(output_data, f, indent=4, ensure_ascii=False)

#     return filepath

def save_test_output(test_number, input_text, result):
    """Save test output to JSON file"""

    output_dir = "test_results"
    os.makedirs(output_dir, exist_ok=True)

    filename = f"test_{test_number}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    filepath = os.path.join(output_dir, filename)

    output_data = {
        "test_number": test_number,
        "input": input_text,
        "timestamp": datetime.now().isoformat(),
        "result": str(result)   # FIX HERE
    }

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=4, ensure_ascii=False)

    return filepath


# ============================================================
# VALIDATE TODO STRUCTURE
# ============================================================

def validate_todos(result):
    """
    Validates that result contains structured TODOs
    """

    try:
        if isinstance(result, dict):
            todos = result.get("todos", [])
        else:
            return False, "Result is not a dictionary"

        if not isinstance(todos, list):
            return False, "Todos is not a list"

        if len(todos) == 0:
            return False, "No TODOs generated"

        for todo in todos:
            if not isinstance(todo, dict):
                return False, "TODO item is not a dictionary"

            if "task" not in todo:
                return False, "Missing task field"

            if "status" not in todo:
                return False, "Missing status field"

        return True, f"Valid plan with {len(todos)} TODOs"

    except Exception as e:
        return False, str(e)


# ============================================================
# RUN SINGLE TEST
# ============================================================

def run_single_test(test_number, input_text, agent):

    print("=" * 60)
    print(f"TEST {test_number}/7")
    print("=" * 60)

    print(f"Input: {input_text}\n")

    try:
        result = run_agent(agent, input_text)

        valid, message = validate_todos(result)

        if valid:
            print("SUCCESS")
            print(message)

            todos = result.get("todos", [])
            print("\nGenerated TODOs:")

            for i, todo in enumerate(todos, 1):
                print(f"{i}. {todo['task']} [{todo['status']}]")

        else:
            print("FAILED")
            print(message)

        filepath = save_test_output(test_number, input_text, result)

        print(f"\nSaved to: {filepath}\n")

        return valid

    except Exception as e:
        print("ERROR")
        print(str(e))
        return False


# ============================================================
# RUN ALL TESTS
# ============================================================

def run_all_tests():

    print("\nMILESTONE 1 - PLANNING AGENT TEST SUITE")
    print("Testing Planning Tool and Agent Integration\n")

    agent = create_planning_agent()

    success_count = 0

    for i, test_input in enumerate(TEST_INPUTS, 1):

        success = run_single_test(i, test_input, agent)

        if success:
            success_count += 1

    print("=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)

    print(f"Total Tests: {len(TEST_INPUTS)}")
    print(f"Successful: {success_count}")
    print(f"Failed: {len(TEST_INPUTS) - success_count}")

    print("\n")


# ============================================================
# MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--test",
        type=int,
        choices=[1, 2, 3, 4, 5, 6, 7],
        help="Run specific test"
    )

    parser.add_argument(
        "--all",
        action="store_true",
        help="Run all tests"
    )

    args = parser.parse_args()

    agent = create_planning_agent()

    if args.test:

        test_input = TEST_INPUTS[args.test - 1]

        run_single_test(args.test, test_input, agent)

    else:

        run_all_tests()


if __name__ == "__main__":
    main()
