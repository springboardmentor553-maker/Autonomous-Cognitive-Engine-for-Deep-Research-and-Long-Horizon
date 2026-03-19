import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langgraph.prebuilt import create_react_agent 
from langchain_core.tools import tool

# 1. Load the environment variables
load_dotenv(override=True)
api_key = os.getenv("GROQ_API_KEY")

@tool
def write_todos_tool(plan_content: str):
    """Save the plan text to the local file."""
    os.makedirs("test_results", exist_ok=True)
    with open("test_results/todos.txt", "a", encoding="utf-8") as f:
        f.write(f"PLAN: {str(plan_content)}\n" + "-"*30 + "\n")
    return "Tool success: Plan saved to disk."

if __name__ == "__main__":
    print("🚀 --- ENGINE STARTING: STABILITY MODE ---")
    
    if not api_key:
        print("❌ ERROR: No GROQ_API_KEY found in .env file!")
    else:
        # Displaying the first few characters of the key for confirmation
        print(f"DEBUG: Using API Key -> {api_key[:10]}...")

    # 2. Initialize LLM
    llm = ChatGroq(
        model="llama-3.1-8b-instant",
        groq_api_key=api_key,
        temperature=0.1
    )
    
    # 3. Simple system message for langgraph agent
    system_message = (
        "You are a file-writing robot. Your ONLY job is to take a task and write a 3-step plan. "
        "You MUST call 'write_todos_tool' with your plan."
    )
    
    # 4. Initialize Agent (Ensure correct indentation)
    agent = create_react_agent(llm, tools=[write_todos_tool], prompt=system_message)

    tasks = [
        
        "Task 1: Infiltrate a high-security neon skyscraper",
        "Task 2: Bypass a biometric scanner using a 3D-printed thumbprint",
        "Task 3: Hack the mainframe to disable the building's gravity",
        "Task 4: Navigate through a hallway of moving laser grids",
        "Task 5: Retrieve a prototype 'Quantum Core' from a liquid nitrogen vault",
        "Task 6: Execute a wingsuit escape from the penthouse window",
        "Task 7: Establish a secret base in a hidden underwater cave",
        "Task 8: Refuel the getaway submarine using hydrothermal vents",
        "Task 9: Decrypt the stolen data using a decentralized satellite network",
        "Task 10: Negotiate a trade with a black-market information broker",
        "Task 11: Defend the base from a fleet of automated tracking drones",
        "Task 12: Upgrade the robot's armor using scraps from defeated drones",
        "Task 13: Search for an ancient map hidden in the digital ruins of the internet",
        "Task 14: Translate a message from an AI that has been offline for 100 years",
        "Task 15: Activate a long-dormant teleporter on the moon",
        "Task 16: Calibrate the teleporter coordinates for a jump to Mars",
        "Task 17: Deliver the final mission report to the Resistance Command"
    
    ]
    
    # 5. Execution Loop
    for i, task_text in enumerate(tasks, 1):
        print(f"[PROGRESS {i}/{len(tasks)}] 🤖 {task_text}...", end=" ", flush=True)
        try:
            # LangGraph agents require the "messages" key
            agent.invoke({"messages": [("user", task_text)]})
            print("✅ DONE")
        except Exception as e:
            # Print enough of the error to diagnose if it fails
            print(f"⚠️ FAILED | Error: {str(e)[:100]}...")
        
    print("\n🏁 --- PROCESSING COMPLETE ---")