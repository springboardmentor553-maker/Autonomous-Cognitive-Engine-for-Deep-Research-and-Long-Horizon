import os
import json
import time # IMPORT ADDED
import warnings
from dotenv import load_dotenv

# Suppress the specific deprecation warning for this run
warnings.filterwarnings("ignore", category=DeprecationWarning)

from langchain_groq import ChatGroq
from langchain_core.tools import StructuredTool
from langchain_core.prompts import PromptTemplate
from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.prebuilt import create_react_agent

# 1. Load Environment Variables
load_dotenv()

# Configure LangSmith Tracing
os.environ["LANGCHAIN_TRACING_V2"] = os.getenv("LANGCHAIN_TRACING_V2", "true")
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY", "")
os.environ["LANGCHAIN_PROJECT"] = os.getenv("LANGCHAIN_PROJECT", "milestone_1_planning")

# 2. Initialize LLM (Groq AI)
llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0.1
)

# 3. Define the Planning Logic
planning_prompt = PromptTemplate(
    input_variables=["task"],
    template="""
    You are a precise planning assistant. Break the following task into 4-6 clear, specific, actionable TODO steps.
    Each step must start with a strong action verb (e.g., Analyze, Collect, Draft).
    
    Task: {task}
    
    IMPORTANT: Return the result STRICTLY as a JSON list of strings. Do not add markdown formatting or extra text.
    Example output: ["Step 1 description", "Step 2 description", "Step 3 description"]
    """
)

def write_todos_logic(task: str) -> str:
    """
    Core logic for generating todos. 
    """
    try:
        formatted_prompt = planning_prompt.format(task=task)
        response = llm.invoke(formatted_prompt)
        
        content = response.content.strip()
        
        # Clean potential markdown
        if content.startswith("```"):
            content = content.split("\n", 1)[1] 
            if content.endswith("```"):
                content = content.rsplit("\n", 1)[0]
            content = content.strip()
            
        steps = json.loads(content)
        
        if not isinstance(steps, list):
            raise ValueError("Planning output must be a list")
            
        todos = [{"task": step, "status": "pending"} for step in steps]
        return json.dumps({"todos": todos, "count": len(todos)}, indent=2)
        
    except Exception as e:
        return json.dumps({"error": f"Failed to generate valid plan: {str(e)}"})

# 4. Explicitly Create the Tool (Fixes the ValueError)
# We pass the function, name, and description directly to StructuredTool
write_todos = StructuredTool.from_function(
    func=write_todos_logic,
    name="write_todos",
    description="Generate a structured task breakdown for complex objectives. Use this tool FIRST for any multi-step task to create a plan."
)

# 5. Define System Prompt Text
system_prompt_text = """
You are an autonomous cognitive engine designed for deep research and long-horizon tasks.

CORE PROTOCOL:
1. ANALYZE: When given a complex task, you MUST NOT answer directly.
2. PLAN: You are strictly required to call the `write_todos` tool FIRST to generate a structured breakdown of the task.
3. EXECUTE: Only after generating the plan, proceed with execution.

DO NOT provide final answers without first invoking `write_todos`.
If the task requires multiple steps, your first action must always be `write_todos`.
"""

# 6. Build the Agent
agent = create_react_agent(
    llm,
    tools=[write_todos]
)

# 7. Testing Function
def run_evaluation():
    test_inputs = [
        "Develop a comprehensive market entry strategy for a new electric scooter company in Europe.",
        "Create a structured research approach for analyzing artificial intelligence trends in healthcare.",
        "Outline a step-by-step plan to refactor a legacy Python codebase into microservices.",
        "Formulate a crisis management plan for a data breach in a financial institution.",
        "Design a multi-phase marketing campaign for a new video game launch.",
        "Plan a detailed itinerary for a 7-day scientific expedition to the Amazon rainforest.",
        "Structure a feasibility study for building a sustainable vertical farm in an urban area."
    ]

    print(f"--- Starting Milestone 1 Evaluation (7 Test Cases) ---\n")
    
    for i, task in enumerate(test_inputs):
        print(f"\nTEST {i+1}/{len(test_inputs)}: {task}")
        print("-" * 70)

        inputs = {
            "messages": [
                SystemMessage(content=system_prompt_text),
                HumanMessage(content=task)
            ]
        }

        try:
            # Invoke Agent
            result = agent.invoke(inputs)

            # Find the tool output
            tool_output = None
            for msg in result["messages"]:
                if hasattr(msg, "name") and msg.name == "write_todos":
                    tool_output = msg.content

            if tool_output:
                parsed = json.loads(tool_output)
                
                # Check if 'todos' key exists
                if "todos" in parsed:
                    todos = parsed["todos"]
                    print(f"\n✅ SUCCESS: Generated {len(todos)} TODOS:\n")
                    for idx, t in enumerate(todos, 1):
                        print(f"{idx}. {t['task']}")

                    # Save to file
                    os.makedirs("outputs", exist_ok=True)
                    fname = f"outputs/test_{i+1}.json"
                    with open(fname, "w") as f:
                        json.dump(parsed, f, indent=2)
                    print(f"\nSaved result to {fname}")
                
                elif "error" in parsed:
                    print(f"\n❌ TOOL ERROR: {parsed['error']}")
                else:
                    print(f"\n❌ UNEXPECTED FORMAT: {tool_output}")
            
            else:
                print("❌ ERROR: Agent did not call the write_todos tool.")

        except Exception as e:
            print(f"❌ CRITICAL ERROR in Test Case {i+1}: {str(e)}")

        print("-" * 70)
        
        # FIX: Sleep for 10 seconds between tests to avoid Rate Limit (429)
        if i < len(test_inputs) - 1:
            print("Waiting 10 seconds to avoid rate limit...")
            time.sleep(10)

if __name__ == "__main__":
    run_evaluation()