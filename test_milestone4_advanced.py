import sys
sys.stdout.reconfigure(encoding='utf-8')  # type: ignore
import os
import json
import random
import time
import operator
from concurrent.futures import ThreadPoolExecutor
from typing import TypedDict, List, Annotated
from langgraph.graph import StateGraph, START, END

# FEATURE 4: LangSmith Tracing Hooks
# (Uncomment this and set your LANGCHAIN_API_KEY to securely transmit traces)
# os.environ["LANGCHAIN_TRACING_V2"] = "true"
# os.environ["LANGCHAIN_PROJECT"] = "Advanced_Deep_Cognitive_Agent"

def merge_dicts(a: dict, b: dict) -> dict:
    c = a.copy()
    c.update(b)
    return c

# Main Supervisor State
class MainState(TypedDict):
    vfs: Annotated[dict, merge_dicts]
    todos: list
    global_context: Annotated[list, operator.add] # FEATURE 5: Shared Memory Blackboard
    memory_usage_kb: int
    objective: str

# Isolated Sub-Agent Graph State
class SubTaskState(TypedDict):
    task_id: int
    task_desc: str
    agent_type: str
    status: str
    attempts: int
    data: str
    vfs: dict
    global_context: list

def write_file(filename: str, content: str):
    return {filename: content}

# --- SUB-AGENT GRAPH NODES ---
def execute_sub_task(state: SubTaskState):
    task_id = state["task_id"]
    agent_type = state["agent_type"]
    desc = state["task_desc"]
    
    print(f"  [RUNNING] {agent_type.upper()}_AGENT handling Task {task_id}")
    time.sleep(0.1) # Simulate think time to show parallel output
    
    content = f"Data acquired by {agent_type} for '{desc}' (Attempt {state['attempts'] + 1})"
    return {"data": content, "attempts": state["attempts"] + 1}

def evaluate_sub_task(state: SubTaskState):
    # FEATURE 3: Error Recovery via Evaluation loop
    task_id = state["task_id"]
    
    # 30% failure rate to demonstrate self-reflection
    if state["attempts"] >= 2 or random.random() > 0.3:
        print(f"  [EVALUATOR] Task {task_id} results APPROVED \u2705")
        vfs_update = write_file(f"task_{task_id}_result.txt", state["data"])
        context_update = [f"Fact from Task {task_id}: {state['task_desc']}"]
        return {"status": "success", "vfs": vfs_update, "global_context": context_update}
    else:
        print(f"  [EVALUATOR] Task {task_id} results REJECTED. Triggering Agent Retry \u274c")
        return {"status": "retry"}

# Sub-Agent Compiled Graph
sub_graph = StateGraph(SubTaskState)
sub_graph.add_node("execute", execute_sub_task)
sub_graph.add_node("evaluate", evaluate_sub_task)
sub_graph.add_edge(START, "execute")
sub_graph.add_edge("execute", "evaluate")
sub_graph.add_conditional_edges("evaluate", lambda s: "retry" if s["status"] == "retry" else "success", {"retry": "execute", "success": END})
compiled_sub_graph = sub_graph.compile()


# --- SUPERVISOR NODES ---
def planning_phase(state: MainState):
    print(f"============================================================")
    print(f"ADVANCED ARCHITECTURE EVALUATION (Milestone 4 Preview)")
    print(f"============================================================")
    print(f"OBJECTIVE: {state['objective']}\n")
    print(f"THINK \u2192 Supervisor generated {len(state['todos'])} tasks.")
    return {"memory_usage_kb": state["memory_usage_kb"]}

def delegate_tasks_parallel(state: MainState):
    todos = state.get("todos", [])
    print(f"ROUTER \u2192 Broadcasting tasks for parallel Fan-Out execution...\n")
    
    tasks_to_run = []
    for i, raw_task in enumerate(todos):
        task_id = i + 1
        task_desc = raw_task.split(". ", 1)[1] if ". " in raw_task else raw_task
        
        # FEATURE 1: Dynamic Routing based on NLP keyword detection
        if "design" in task_desc.lower() or "architect" in task_desc.lower():
            agent_type = "architecture"
        elif "analyze" in task_desc.lower() or "review" in task_desc.lower():
            agent_type = "analysis"
        else:
            agent_type = "research"
            
        print(f"  [ROUTING] Task {task_id} assigned \u2192 '{agent_type}' agent.")
        initial_sub_state = {"task_id": task_id, "task_desc": task_desc, "agent_type": agent_type, "status": "pending", "attempts": 0, "data": "", "vfs": {}, "global_context": []}
        tasks_to_run.append(initial_sub_state)

    print("\n--- INITIATING PARALLEL SUB-AGENT EXECUTION ---")
    
    # FEATURE 2: Parallel Task Execution via ThreadPool mapping
    vfs_aggregated = {}
    context_aggregated = []
    
    with ThreadPoolExecutor(max_workers=len(tasks_to_run)) as executor:
        results = executor.map(compiled_sub_graph.invoke, tasks_to_run)
        
        for result in results:
            vfs_aggregated.update(result.get("vfs", {}))
            context_aggregated.extend(result.get("global_context", []))

    return {"vfs": vfs_aggregated, "global_context": context_aggregated}


def final_report(state: MainState):
    print("\n--- REJOINING BRANCHES (Fan-In) ---")
    print("All parallel agents have completed their workflows.\n")
    print("FINAL EXECUTION REPORT")
    print("============================================================")
    
    print("\nGLOBAL BLACKBOARD (Shared Memory Extracted directly from state)")
    print("----------------------------------------")
    for fact in state.get("global_context", []):
        print(f"  * {fact}")
    
    print("\n\nVIRTUAL MEMORY STRUCTURE (Aggregated from Parallel Branches)")
    print("----------------------------------------")
    for filename, content in state.get("vfs", {}).items():
        print(f"📄 {filename}\n   └── {content}")
        
    print("\n============================================================")
    print("SUCCESS: Advanced execution completed.")
    return {"step": "done"}


workflow = StateGraph(MainState)
workflow.add_node("planning", planning_phase)
workflow.add_node("parallel_delegation", delegate_tasks_parallel)
workflow.add_node("report", final_report)

workflow.add_edge(START, "planning")
workflow.add_edge("planning", "parallel_delegation")
workflow.add_edge("parallel_delegation", "report")
workflow.add_edge("report", END)

app = workflow.compile()

if __name__ == "__main__":
    test_case = {
        "objective": "design a system architecture for a scalable cloud-based e-commerce platform",
        "todos": [
            "1. Define Cloud Infrastructure Requirements",
            "2. Design Microservices and API Gateways",
            "3. Plan Database Scaling and Caching Strategy",
            "4. Architect Global Content Delivery Network",
            "5. Analyze Competitor Load Balancing Tradeoffs"
        ]
    }

    initial_state: MainState = {
        "vfs": {}, 
        "todos": test_case["todos"],
        "global_context": [],
        "memory_usage_kb": 12,
        "objective": test_case["objective"]
    }
    
    app.invoke(initial_state)
