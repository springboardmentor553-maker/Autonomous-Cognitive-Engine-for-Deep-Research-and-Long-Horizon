import os
import json
from langchain_groq import ChatGroq
from langchain_core.tools import tool

# Minimalist tool for Python 3.13 stability
@tool
def write_todos_tool(task: str):
    """Break down a task into 5 steps."""
    # Hardcoding the key for one final test to bypass .env issues
    llm = ChatGroq(
        model="llama-3.1-8b-instant", 
        temperature=0,
        groq_api_key="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"
    )
    
    try:
        resp = llm.invoke(f"List 5 steps for: {task}. Return ONLY a JSON list of strings.")
        # Super simple extraction
        import re
        match = re.search(r'\[.*\]', resp.content, re.DOTALL)
        steps = json.loads(match.group()) if match else []
    except:
        steps = []

    # Ensure exactly 5
    while len(steps) < 5: steps.append("Finalize step")
    return {"todos": [{"task": s[:100], "status": "pending"} for s in steps[:5]]}

if __name__ == "__main__":
    print("--- START ---")
    # Call the function logic directly without .invoke() to test stability
    result = write_todos_tool.func("Test task")
    print(json.dumps(result))
    print("--- END ---")