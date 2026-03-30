import os
import json
import time
from typing import Dict, List
from dotenv import load_dotenv

load_dotenv()

# NOTE: The LANGCHAIN_PROJECT name is no longer hardcoded here.
# It is now dynamically set in your test_milestone4.py file!

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage
from graphs.main_graph import build_cognitive_engine

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)

def create_planning_agent():
    """Returns our custom compiled LangGraph Cognitive Engine."""
    return build_cognitive_engine()

def generate_initial_plan(task: str) -> List[Dict]:
    """Phase 1: Breaks down the user task into a structured list of todos."""
    prompt = f"""
    You are the Lead Strategic Planner. Break down the following task into a sequential list of sub-tasks.
    Assign each sub-task to the most appropriate role: 'researcher', 'summarizer', 'comparator', or 'refiner'.
    
    CRITICAL: You must output ONLY a valid JSON list of dictionaries with 'task' and 'status' keys.
    The status must always be "pending".
    
    Task: {task}
    """
    
    response = llm.invoke(prompt)
    
    try:
        content = response.content.replace("```json", "").replace("```", "").strip()
        todos = json.loads(content)
        return todos
    except Exception as e:
        print(f"Error parsing plan, using fallback: {e}")
        return [
            {"task": f"researcher: Gather data on {task}", "status": "pending"},
            {"task": "refiner: Polish the final output", "status": "pending"}
        ]

def run_agent(agent, task: str, thread_id: str = "default") -> Dict:
    """Phase 2: Runs the Cognitive Engine with the generated plan."""
    print("\n🧠 PHASE 1: Generating Execution Plan...")
    todos = generate_initial_plan(task)
    
    print(f"📋 Plan generated with {len(todos)} steps:")
    for i, t in enumerate(todos, 1):
        print(f"  {i}. {t['task']}")
        
    initial_state = {
        "messages": [HumanMessage(content=task)],
        "todos": todos
    }
    
    config = {"configurable": {"thread_id": thread_id}}
    final_state = None
    
    print("\n⚙️ PHASE 2: Starting Multi-Agent Engine Execution...")
    
    for event in agent.stream(initial_state, config, stream_mode="values"):
        final_state = event
        
        current_todos = event.get("todos", [])
        pending = [t for t in current_todos if t.get("status") == "pending"]
        done = [t for t in current_todos if t.get("status") == "done"]
        
        if len(current_todos) > 0:
            print(f"🔄 Graph Update: {len(done)} tasks done, {len(pending)} pending...")
        time.sleep(1) 
        
    return {
        "task": task,
        "messages": final_state.get("messages", []) if final_state else [],
        "todos": final_state.get("todos", []) if final_state else []
    }

def save_result_to_json(result: Dict, filename: str, output_dir: str = "outputs"):
    """Saves the final state and extracts the Final Answer."""
    os.makedirs(output_dir, exist_ok=True)
    serializable_result = {
        "task": result["task"],
        "todos": result["todos"],
        "message_count": len(result["messages"])
    }
    
    # --- THE MAGIC HAPPENS HERE ---
    # Because we injected AIMessage(content=result) in execution_node.py,
    # this loop will perfectly catch the last agent's output (usually the Refiner)
    # and save it as the 'final_response' in your JSON file!
    for msg in reversed(result["messages"]):
        if hasattr(msg, 'content') and hasattr(msg, 'type') and msg.type == "ai":
            serializable_result["final_response"] = msg.content
            break
            
    filepath = os.path.join(output_dir, filename)
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(serializable_result, f, indent=2, ensure_ascii=False)
    print(f"✅ Saved result to {filepath}")
    return filepath

if __name__ == "__main__":
    print("=" * 60)
    print("Cognitive Engine Ready")
    print("=" * 60)