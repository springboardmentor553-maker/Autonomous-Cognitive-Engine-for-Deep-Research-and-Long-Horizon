"""
Test Planning Agent - Milestone 1

This script tests the ReAct planning agent with 5 complex inputs
and saves the generated todos to outputs/*.json files.

Test Inputs:
1. Comparative study of EV vs Hydrogen vehicles
2. Build an AI chatbot architecture
3. Stock market research strategy
4. Renewable energy research outline
5. AI agents in healthcare report
"""

import os
import sys
import json
from datetime import datetime

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Enable LangSmith Tracing
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = os.getenv("LANGCHAIN_PROJECT", "Milestone1-Planning")

from app import create_planning_agent, run_agent, save_result_to_json


# Test inputs as specified
TEST_INPUTS = [
    "Comparative study of EV vs Hydrogen vehicles",
    "Build an AI chatbot architecture",
    "Stock market research strategy",
    "Renewable energy research outline",
    "AI agents in healthcare report"
]


def run_all_tests():
    """
    Run the planning agent on all test inputs and save results.
    """
    print("=" * 70)
    print("MILESTONE 1 - PLANNING AGENT TEST SUITE")
    print(f"Timestamp: {datetime.now().isoformat()}")
    print("=" * 70)
    
    # Create the agent once
    print("\nInitializing Planning Agent...")
    agent = create_planning_agent()
    print("Agent initialized successfully!\n")
    
    # Results summary
    all_results = []
    
    # Run each test
    for i, task in enumerate(TEST_INPUTS, 1):
        print("-" * 70)
        print(f"TEST {i}/5: {task}")
        print("-" * 70)
        
        try:
            # Run agent with unique thread ID
            thread_id = f"test-{i}-{datetime.now().strftime('%Y%m%d%H%M%S')}"
            result = run_agent(agent, task, thread_id=thread_id)
            
            # Display generated todos
            print(f"\nGenerated {len(result['todos'])} TODOs:")
            for j, todo in enumerate(result["todos"], 1):
                status_icon = "⬜" if todo["status"] == "pending" else "✅"
                print(f"  {j}. {status_icon} {todo['task']}")
            
            # Save to JSON
            filename = f"test_{i}_{task.lower().replace(' ', '_')[:30]}.json"
            filepath = save_result_to_json(result, filename)
            
            # Track result
            all_results.append({
                "test_number": i,
                "task": task,
                "todo_count": len(result["todos"]),
                "output_file": filepath,
                "success": True
            })
            
            print(f"\n✅ Test {i} completed successfully!")
            
        except Exception as e:
            print(f"\n❌ Test {i} failed with error: {str(e)}")
            all_results.append({
                "test_number": i,
                "task": task,
                "todo_count": 0,
                "output_file": None,
                "success": False,
                "error": str(e)
            })
        
        print()
    
    # Print summary
    print("=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)
    
    successful = sum(1 for r in all_results if r["success"])
    print(f"\nTotal Tests: {len(TEST_INPUTS)}")
    print(f"Successful: {successful}")
    print(f"Failed: {len(TEST_INPUTS) - successful}")
    
    print("\nDetailed Results:")
    for r in all_results:
        status = "✅ PASS" if r["success"] else "❌ FAIL"
        print(f"  {r['test_number']}. {status} - {r['task'][:40]}... ({r['todo_count']} todos)")
    
    # Save summary
    summary_file = os.path.join("outputs", "test_summary.json")
    os.makedirs("outputs", exist_ok=True)
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump({
            "timestamp": datetime.now().isoformat(),
            "total_tests": len(TEST_INPUTS),
            "successful": successful,
            "failed": len(TEST_INPUTS) - successful,
            "results": all_results
        }, f, indent=2)
    
    print(f"\nSummary saved to: {summary_file}")
    print("\n" + "=" * 70)
    print("Check LangSmith dashboard for detailed traces:")
    print(f"Project: {os.getenv('LANGCHAIN_PROJECT', 'Milestone1-Planning')}")
    print("=" * 70)
    
    return all_results


def run_single_test(test_number: int):
    """
    Run a single test by number (1-5).
    """
    if test_number < 1 or test_number > len(TEST_INPUTS):
        print(f"Invalid test number. Please choose 1-{len(TEST_INPUTS)}")
        return
    
    task = TEST_INPUTS[test_number - 1]
    print(f"\nRunning single test: {task}")
    print("-" * 50)
    
    agent = create_planning_agent()
    result = run_agent(agent, task, thread_id=f"single-test-{test_number}")
    
    print(f"\nGenerated TODOs:")
    for i, todo in enumerate(result["todos"], 1):
        print(f"  {i}. {todo['task']} [{todo['status']}]")
    
    filename = f"single_test_{test_number}.json"
    save_result_to_json(result, filename)
    
    return result


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Test the Planning Agent")
    parser.add_argument(
        "--test", 
        type=int, 
        choices=[1, 2, 3, 4, 5],
        help="Run a single test by number (1-5)"
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Run all tests"
    )
    
    args = parser.parse_args()
    
    if args.test:
        run_single_test(args.test)
    else:
        # Default: run all tests
        run_all_tests()
