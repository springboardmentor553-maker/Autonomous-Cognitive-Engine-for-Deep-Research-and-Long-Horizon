# Updated app.py
import os
import json
import ast
from typing import List, Dict
from dotenv import load_dotenv
import time

load_dotenv()

# Set project name for Milestone 2
os.environ["LANGCHAIN_PROJECT"] = "Milestone3 -> Sub-agents"

from langchain_core.tools import Tool
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver
from langchain_google_genai import ChatGoogleGenerativeAI
from tools.execution.delegation_tool import task_delegate

# Import Existing Planning Tool
from tools.planning.write_todos import write_todos
# Import New Milestone 2 Tools
from tools.execution.file_tools import write_file, read_file, ls, edit_file

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)

# Registering all tools
# Note: creating a list of tools including your Milestone 1 tool
tools = [write_todos, write_file, read_file, ls, edit_file, task_delegate]

# ENHANCED SYSTEM PROMPT (Based on Mentor Notes)
SYSTEM_PROMPT = """You are a Lead Research Supervisor. 

You manage a team of specialists: researcher, summarizer, comparator, and refiner.

DELEGATION PROTOCOL:
1. **Initial Planning**: Always start with 'write_todos'.
2. **Autonomous Workflow**:
   - For raw data gathering, use the 'researcher'.
   - For cleaning up dense research, use the 'summarizer'.
   - For comparing multiple entities, use the 'comparator'.
   - ALWAYS use the 'refiner' as the final step to produce the final answer.
3. **Integration & Storage**: 
   - Every time a sub-agent returns a result, you MUST call 'write_file' to save it to the Virtual File System.
   - Use 'ls' to maintain awareness of the files your team has created.

Your goal is to coordinate these experts to solve complex, long-horizon research tasks while maintaining a clean context window."""


def create_planning_agent():
    memory = MemorySaver()
    # Updated to include all tools and the new prompt
    agent = create_react_agent(
        model=llm,
        tools=tools,
        checkpointer=memory,
        prompt=SYSTEM_PROMPT
    )
    return agent

# --- I have kept your run_agent and save_result_to_json functions identical to your M1 code ---
def run_agent(agent, task: str, thread_id: str = "default") -> Dict:
    config = {"configurable": {"thread_id": thread_id}}
    input_message = {"messages": [("user", task)]}
    final_state = None
    todos = []
    
    # We use stream to capture the tool calls properly
    for event in agent.stream(input_message, config, stream_mode="values"):
        time.sleep(2)
        final_state = event
        if "messages" in event:
            for msg in event["messages"]:
                if hasattr(msg, 'name') and msg.name == "write_todos":
                    try:
                        content = msg.content
                        if isinstance(content, str):
                            clean_content = content.strip()
                            todos = ast.literal_eval(clean_content) if clean_content.startswith('[') else []
                        elif isinstance(content, list):
                            todos = content
                    except:
                        pass
    return {
        "task": task,
        "messages": final_state.get("messages", []) if final_state else [],
        "todos": todos
    }

def save_result_to_json(result: Dict, filename: str, output_dir: str = "outputs"):
    os.makedirs(output_dir, exist_ok=True)
    serializable_result = {
        "task": result["task"],
        "todos": result["todos"],
        "message_count": len(result["messages"])
    }
    for msg in reversed(result["messages"]):
        if hasattr(msg, 'content') and hasattr(msg, 'type') and msg.type == "ai":
            serializable_result["final_response"] = msg.content
            break
    filepath = os.path.join(output_dir, filename)
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(serializable_result, f, indent=2, ensure_ascii=False)
    print(f"Saved result to {filepath}")
    return filepath

if __name__ == "__main__":
    print("=" * 60)
    print("Milestone 2: Context Offloading Engine")
    print("=" * 60)
    
    agent = create_planning_agent()
    
    # Mentor's test case: Process 3 distinct pieces of info
    test_task = """
    I have three reports:
    1. Climate report: Global temps rose 1.2C.
    2. Energy report: Solar usage is up 20%.
    3. Policy report: New green tax implemented.
    
    Summarize each into separate files, then read ONLY the climate and energy files 
    to tell me the correlation between temp rise and solar adoption.
    """
    
    result = run_agent(agent, test_task, thread_id="m2-test-1")
    save_result_to_json(result, "m2_output.json")