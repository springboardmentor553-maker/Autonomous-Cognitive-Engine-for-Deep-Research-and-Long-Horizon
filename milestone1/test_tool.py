import json
import re
from pydantic import BaseModel, Field
from langchain_groq import ChatGroq
from langchain_core.tools import tool
from dotenv import load_dotenv

load_dotenv()

class PlanningInput(BaseModel):
    task: str = Field(description="The complex task that needs a 5-step plan")

@tool(args_schema=PlanningInput)
def write_todos_tool(task: str):
    """Generate a structured 5-step task breakdown."""
    llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0)
    prompt_text = f"Return ONLY a JSON list of 5 strings for this task: {task}"
    
    try:
        response = llm.invoke(prompt_text)
        content = response.content
        json_match = re.search(r'\[.*\]', content, re.DOTALL)
        steps = json.loads(json_match.group()) if json_match else content.split("\n")
    except Exception:
        steps = [f"Step {i+1} for {task}" for i in range(5)]

    steps = (steps[:5] + ["Finalize review"] * 5)[:5]
    return {"todos": [{"task": s, "status": "pending"} for s in steps]}