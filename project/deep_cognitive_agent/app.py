import json
import sys
import os

# 1. Path setup: Ensure Python can find 'write_todos_root' in the parent folder
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.join(current_dir, "..")
sys.path.append(parent_dir)

from langchain_core.tools import tool
from langchain_groq import ChatGroq
from langgraph.prebuilt import create_react_agent

# Try/Except block to catch the path error if it persists
try:
    from write_todos_root import write_todos_tool
except ImportError:
    print("❌ Error: 'write_todos_root.py' not found in the parent directory.")
    sys.exit(1)

def create_planning_agent():
    # It's better to let ChatGroq find the key from your .env automatically
    llm = ChatGroq(
        model="llama-3.1-8b-instant", 
        temperature=0
    )
    
    system_prompt = (
        "You are a planning assistant. You MUST use 'write_todos_tool' for every task. "
        "Provide ONLY the tool call. Do not explain your actions."
    )
    
    # Delete BOTH old returns and use this single one:
    return create_react_agent(
        llm, 
        tools=[write_todos_tool], 
        state_modifier=system_prompt 
    )
    
    # Changed 'state_modifier' to 'prompt' to fix your TypeError
    return create_react_agent(
        llm, 
        tools=[write_todos_tool], 
        prompt=system_prompt 
    )
    
    return create_react_agent(
        llm, 
        tools=[write_todos_tool], 
        state_modifier=system_prompt
    )

def run_agent(agent, input_text):
    inputs = {"messages": [("user", input_text)]}
    print(f"\n🤖 Processing Task: '{input_text}'")
    
    try:
        result = agent.invoke(inputs)
        
        todos_list = []
        for msg in result["messages"]:
            if msg.type == "tool" or (hasattr(msg, "name") and "write_todos" in msg.name):
                content = msg.content
                data = json.loads(content) if isinstance(content, str) else content
                if isinstance(data, dict) and "todos" in data:
                    todos_list = data["todos"]
                    print(f"   ✅ Success: Extracted {len(todos_list)} items.")
        
        return {"messages": result.get("messages", []), "todos": todos_list}

    except Exception as e:
        print(f"❌ LLM Error: {e}")
        return {"messages": [], "todos": []}

# --- THIS IS THE NEW EXECUTION BLOCK ---
if __name__ == "__main__":
    print("🚀 --- AUTONOMOUS COGNITIVE ENGINE STARTING ---")
    
    # Initialize the agent
    my_agent = create_planning_agent()
    
    # THE 17 TASKS: Replace these strings with your actual tasks
    tasks_to_run = [
        "Task 1: Initialize research framework",
        "Task 2: Scan for data sources",
        "Task 3: Verify connectivity",
        "Task 4: Authenticate modules",
        # ... Add tasks 5 through 17 here ...
        "Task 17: Generate final report"
    ]
    
    # Loop through and run every task
    for i, task_text in enumerate(tasks_to_run, 1):
        print(f"\n[PROGRESS {i}/17]")
        run_agent(my_agent, task_text)

    print("\n🏁 --- ALL TASKS COMPLETED ---")