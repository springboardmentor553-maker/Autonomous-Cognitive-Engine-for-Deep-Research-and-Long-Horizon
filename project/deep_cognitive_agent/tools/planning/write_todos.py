from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from typing import List, Dict
import os
import ast

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0,
    google_api_key=os.getenv("GOOGLE_API_KEY"),
)

@tool
def write_todos(task: str) -> List[Dict]:
    """
    Break a complex task into at least 5 actionable steps.
    """

    prompt = f"""
Break the following complex task into at least 5 detailed,
clear, logically ordered steps.

Return ONLY numbered steps.

Task:
{task}
"""

    response = llm.invoke(prompt)
    text = response.content

    lines = text.strip().split("\n")
    steps = [line.strip() for line in lines if line.strip()]

    todos = []
    for step in steps:
        clean_step = step.lstrip("0123456789.)- ").strip()
        if clean_step:
            todos.append({
                "task": clean_step,
                "status": "pending"
            })

    return todos
