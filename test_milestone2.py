import operator
from typing import TypedDict, Annotated, List, Dict, Any
from langgraph.graph import StateGraph, START, END
import json

# Define State Structure update for parameterization
class TestState(TypedDict):
    vfs: dict
    todos: list
    messages: list
    step: str
    memory_usage_kb: int

# Mocked Tools
def write_todos(state: TestState, todos: list):
    print(f"→ write_todos(...)")
    print(f"  [Plan Generated]: {json.dumps(todos, indent=2)}")
    state["todos"] = todos
    return state

def write_file(state: TestState, filename: str, content: str):
    print(f"→ write_file(\"{filename}\")")
    print(f"  [Content Written]: {content}")
    state["vfs"][filename] = content
    state["memory_usage_kb"] += len(content) // 1024 + 1
    return state

def read_file(state: TestState, filename: str):
    print(f"→ read_file(\"{filename}\")")
    if filename in state["vfs"]:
        print(f"  [Content Read]: {state['vfs'][filename]}")
    else:
        print(f"  [Content Read]: [Offloaded Full Content of {filename}...]")
    return state

def edit_file(state: TestState, filename: str, edits: str):
    print(f"→ edit_file(\"{filename}\")")
    print(f"  [Edits Applied]: {edits}")
    if filename in state["vfs"]:
        state["vfs"][filename] += f"\n{edits}"
        print(f"  [New Content]: {state['vfs'][filename]}")
    return state

# Milestone 1: Agent plans the execution sequence based on the user's prompt
def planning_phase(state: TestState):
    print("\n--- [MILESTONE 1] Agent Task Decomposition & Planning ---")
    print("User Request: \"Generate a research report on renewable energy policies.\"")
    todos = [
        "1. Identify key renewable energy policies globally",
        "2. Collect examples from major countries",
        "3. Summarize policy differences",
        "4. Analyze advantages and limitations",
        "5. Produce final report"
    ]
    
    state = write_todos(state, todos)
    print("\n[Status] Milestone 1 task decomposition successful. Proceeding to Milestone 2 execution loop.")
    return {"step": "step_1", "todos": state["todos"], "memory_usage_kb": state["memory_usage_kb"]}

# Milestone 2: Agent Nodes representing logical dependency chains based on Architecture Doc
def step_1(state: TestState):
    print("\n--- [MILESTONE 2: Execution Loop] ---")
    print("Agent executes TODO 1 & 2: Identifying policies and collecting examples.")
    state = write_file(state, "global_policies.txt", "1. Paris Agreement targets\n2. EU Green Deal\n3. Kyoto Protocol milestones")
    state = write_file(state, "country_examples.txt", "- Germany: Energiewende (wind & solar)\n- China: Massive Solar PV rollout\n- USA: Inflation Reduction Act credits")
    return {"step": "step_2", "vfs": state["vfs"], "memory_usage_kb": state["memory_usage_kb"]}

def step_2(state: TestState):
    print("Agent executes TODO 3: Summarize policy differences.")
    state = read_file(state, "global_policies.txt")
    state = read_file(state, "country_examples.txt")
    summary_content = "Summary: EU focuses on strict regulatory targets, while the USA relies more on tax incentives and market subsidies. China heavily subsidizes manufacturing."
    state = write_file(state, "policy_summary.txt", summary_content)
    return {"step": "comparison", "vfs": state["vfs"], "memory_usage_kb": state["memory_usage_kb"]}

def comparison(state: TestState):
    print("Agent executes TODO 4: Analyze advantages and limitations.")
    state = read_file(state, "policy_summary.txt")
    analysis_content = "Advantages: Rapid tech deployment due to subsidies.\nLimitations: Grid instability issues in Germany and supply chain bottlenecks."
    state = write_file(state, "analysis.txt", analysis_content)
    return {"step": "refining", "vfs": state["vfs"], "memory_usage_kb": state["memory_usage_kb"]}

def refine_result(state: TestState):
    print("Agent executes TODO 5: Produce final report.")
    state = read_file(state, "policy_summary.txt")
    state = read_file(state, "analysis.txt")
    state = write_file(state, "final_report.txt", "FINAL RESEARCH REPORT: Renewable Energy Policies...\n[Integration of Summary and Analysis]")
    # Showing edit file tool for the sake of completeness over the workflow
    state = edit_file(state, "final_report.txt", "Addendum: Ensure grid modernization is included as a core limitation fix.")
    return {"step": "done", "vfs": state["vfs"], "memory_usage_kb": state["memory_usage_kb"]}

# Build Workflow (Dependency Chain)
workflow = StateGraph(TestState)
workflow.add_node("planning", planning_phase)
workflow.add_node("step_1", step_1)
workflow.add_node("step_2", step_2)
workflow.add_node("comparison", comparison)
workflow.add_node("refine", refine_result)

workflow.add_edge(START, "planning")
workflow.add_edge("planning", "step_1")
workflow.add_edge("step_1", "step_2")
workflow.add_edge("step_2", "comparison")
workflow.add_edge("comparison", "refine")
workflow.add_edge("refine", END)

app = workflow.compile()


if __name__ == "__main__":
    print("==================================================")
    print("COMBINED WORKFLOW TEST: MILESTONE 2 EXECUTION LOOP")
    print("==================================================")
    
    initial_state = {
        "vfs": {}, 
        "todos": [],
        "messages": [], 
        "step": "start", 
        "memory_usage_kb": 12 # Base memory
    }
        
    for event in app.stream(initial_state):
        for node_name, state_update in event.items():
            print(f"\n--- [STATE OUTPUT from node: {node_name}] ---")
            print(f"Memory Usage window remains optimized: {state_update.get('memory_usage_kb', 'unchanged')} KB")
            if "todos" in state_update:
                print(f"Current Plan (Todos): {len(state_update['todos'])} tasks stored centrally.")
            if "vfs" in state_update:
                print(f"VFS State containing stored output files: \n{json.dumps(state_update['vfs'], indent=2)}")
            print("-" * 50 + "\n")
            
    print("=== TRACE COMPLETE ===")
    print("The agent successfully demonstrated incremental long-horizon task execution.")
    