print("=" * 60)
print("DEBUGGING SCRIPT")
print("=" * 60)

# Test 1: Basic imports
print("\n[1/8] Testing basic imports...")
try:
    import os
    import sys
    print("    ✓ os, sys imported")
except Exception as e:
    print(f"    ✗ Failed: {e}")
    exit(1)

# Test 2: dotenv
print("[2/8] Testing dotenv...")
try:
    from dotenv import load_dotenv
    load_dotenv()
    print("    ✓ dotenv imported and loaded")
except Exception as e:
    print(f"    ✗ Failed: {e}")
    exit(1)

# Test 3: Check .env file
print("[3/8] Checking .env file...")
google_key = os.getenv("GOOGLE_API_KEY")
if google_key:
    print(f"    ✓ GOOGLE_API_KEY found: {google_key[:20]}...")
else:
    print("    ✗ GOOGLE_API_KEY not found!")
    print("    Make sure .env file exists with:")
    print("    GOOGLE_API_KEY=your_key_here")
    exit(1)

# Test 4: LangChain imports
print("[4/8] Testing LangChain imports...")
try:
    from langchain_core.messages import HumanMessage
    print("    ✓ langchain_core imported")
except Exception as e:
    print(f"    ✗ Failed: {e}")
    exit(1)

# Test 5: Google AI import
print("[5/8] Testing Google AI import...")
try:
    from langchain_google_genai import ChatGoogleGenerativeAI
    print("    ✓ langchain_google_genai imported")
except Exception as e:
    print(f"    ✗ Failed: {e}")
    print("    Run: uv pip install langchain-google-genai")
    exit(1)

# Test 6: Local imports
print("[6/8] Testing local imports...")
try:
    from workflow.memory_state import AgentState
    print("    ✓ workflow.memory_state imported")
except Exception as e:
    print(f"    ✗ Failed: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

try:
    from brains.mainagent import write_todos
    print("    ✓ brains.mainagent imported")
except Exception as e:
    print(f"    ✗ Failed: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

try:
    from workflow.flow import create_agent_executor
    print("    ✓ workflow.flow imported")
except Exception as e:
    print(f"    ✗ Failed: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

# Test 7: Create LLM
print("[7/8] Testing LLM creation...")
try:
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash-exp",
        temperature=0.7
    )
    print("    ✓ LLM created")
except Exception as e:
    print(f"    ✗ Failed: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

# Test 8: Simple LLM call
print("[8/8] Testing simple LLM call (may take 5-10 seconds)...")
try:
    response = llm.invoke([HumanMessage(content="Say 'test successful'")])
    print(f"    ✓ LLM responded: {response.content[:50]}")
except Exception as e:
    print(f"    ✗ Failed: {e}")
    