import os
from langchain_groq import ChatGroq
from langgraph.prebuilt import create_react_agent
from langchain_core.tools import tool

@tool
def write_todos_tool(plan_content: str):
    """Save the plan text to the local file."""
    os.makedirs("test_results", exist_ok=True)
    with open("test_results/todos.txt", "a", encoding="utf-8") as f:
        f.write(f"PLAN: {str(plan_content)}\n" + "-"*30 + "\n")
    return "Tool success: Plan saved to disk."

if __name__ == "__main__":
    print("🚀 --- ENGINE STARTING: STABILITY MODE ---")
    
    llm = ChatGroq(
        model="llama-3.1-8b-instant",
        groq_api_key="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",
        temperature=0.1 # Slight temperature helps it follow formatting better
    )
    
    # We make the prompt extremely simple and strict
    system_message = (
        "You are a file-writing robot. Your ONLY job is to take a task and write a 3-step plan for it. "
        "You MUST call 'write_todos_tool' with your plan. "
        "Do not talk to the user. Do not say 'Here is the plan'. "
        "JUST CALL THE TOOL."
    )
    
    agent = create_react_agent(llm, tools=[write_todos_tool], prompt=system_message)
    
    tasks = [
        "Task 1: Initialize research framework", "Task 2: Scan for data sources",
        "Task 3: Verify connectivity", "Task 4: Authenticate modules",
        "Task 5: Data ingestion setup", "Task 6: Schema mapping",
        "Task 7: API endpoint validation", "Task 8: Load balancing configuration",
        "Task 9: Security protocol handshake", "Task 10: Metadata extraction",
        "Task 11: Dependency resolution", "Task 12: Cache warming",
        "Task 13: Query optimization", "Task 14: Latency benchmarking",
        "Task 15: Error log aggregation", "Task 16: Module synchronization",
        "Task 17: Generate final report"
    ]
    
    for i, task_text in enumerate(tasks, 1):
        print(f"[PROGRESS {i}/17] 🤖 {task_text}...", end=" ", flush=True)
        
        try:
            # We allow 10 steps now to give it room to correct itself if it makes a mistake
            config = {"recursion_limit": 10}
            agent.invoke({"messages": [("user", task_text)]}, config=config)
            print("✅ DONE")
            
        except Exception as e:
            # We catch the error but keep the engine running
            print("⚠️ SKIPPED")
        
    print("\n🏁 --- PROCESSING COMPLETE ---")