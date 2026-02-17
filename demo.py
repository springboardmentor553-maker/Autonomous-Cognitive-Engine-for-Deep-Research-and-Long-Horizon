print("=" * 80)
print("SCRIPT STARTING")
print("=" * 80)

import sys
import os

print("[1/10] Importing dotenv...")
from dotenv import load_dotenv
load_dotenv()
print("    ✓ Done")

print("[2/10] Checking API key...")
google_key = os.getenv("GOOGLE_API_KEY")
if not google_key:
    print("    ✗ GOOGLE_API_KEY not found in .env!")
    sys.exit(1)
print(f"    ✓ Found: {google_key[:20]}...")

print("[3/10] Importing langchain_core...")
from langchain_core.messages import HumanMessage
print("    ✓ Done")

print("[4/10] Importing workflow.flow...")
try:
    from workflow.flow import create_agent_executor
    print("    ✓ Done")
except Exception as e:
    print(f"    ✗ Failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "=" * 80)
print("MILESTONE 1: Task Planning Demo")
print("=" * 80)

system_prompt = """You are an AI agent. When given a complex task, use the write_todos tool to break it into 3-7 sub-tasks. Each sub-task should be clear and actionable."""

demo_request = "Compare different programming languages for web development"

print(f"\nRequest: {demo_request}")
print("\n[5/10] Creating agent executor...")

try:
    agent = create_agent_executor()
    print("    ✓ Agent created")
except Exception as e:
    print(f"    ✗ Failed to create agent: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n[6/10] Preparing message...")
message = HumanMessage(content=f"{system_prompt}\n\nTask: {demo_request}")
print("    ✓ Message prepared")

print("\n[7/10] Invoking agent (may take 10-60 seconds)...")
print("    Please wait...")

try:
    result = agent.invoke(
        {"messages": [message]},
        {"configurable": {"thread_id": "demo"}, "recursion_limit": 10}
    )
    print("    ✓ Agent completed!")
except Exception as e:
    print(f"    ✗ Agent failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n[8/10] Extracting results...")
todos = result.get("todos", [])
print(f"    Found {len(todos)} TODOs")

print("\n[9/10] Displaying results...")
print("\n" + "=" * 80)
print("RESULTS")
print("=" * 80)

if todos:
    print(f"\n✓ Created {len(todos)} TODO items:\n")
    for i, todo in enumerate(todos, 1):
        print(f"  {i}. {todo['description']}")
else:
    print("\n✗ No TODOs were created")
    print("\nDEBUG - Full result:")
    print(result)

print("\n[10/10] Complete!")
print("\n" + "=" * 80)
print("✓ MILESTONE 1 DEMO FINISHED")
print("=" * 80)