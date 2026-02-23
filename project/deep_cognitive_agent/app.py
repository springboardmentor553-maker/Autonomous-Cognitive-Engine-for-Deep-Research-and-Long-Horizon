import json
import sys
import os

# This adds the parent directory to the path so it can find the 'agents' folder
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import json
from agents.planning_agent import create_planning_agent

def run_agent(agent, input_text):
    inputs = {"messages": [("user", input_text)]}
    result = agent.invoke(inputs)
    
    todos_list = []
    for msg in result["messages"]:
        # Look for the output of the write_todos tool
        if hasattr(msg, "name") and msg.name == "write_todos":
            try:
                # 1. Parse the content if it's a string
                data = json.loads(msg.content) if isinstance(msg.content, str) else msg.content
                
                # 2. Extract the 'todos' list from the dictionary the tool returns
                if isinstance(data, dict) and "todos" in data:
                    todos_list = data["todos"]
            except Exception as e:
                print(f"Error parsing tool output: {e}")

    # Convert messages to serializable format to avoid the HumanMessage error
    serializable_messages = []
    for m in result["messages"]:
        serializable_messages.append({
            "type": m.type,
            "content": m.content,
            "name": getattr(m, "name", None)
        })

    return {
        "messages": serializable_messages,
        "todos": todos_list
    }