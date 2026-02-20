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
You are a professional planning assistant.

Break the following complex task into EXACTLY FIVE
clear, logically ordered, non-redundant steps.

STRICT RULES:
- Exactly 5 steps (no more, no less)
- Each step must begin with a strong action verb
- Steps must be specific and actionable
- Do not repeat the task sentence
- Do not include introduction or conclusion text
- Return ONLY a numbered list

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

    # Force exactly 5
    if len(todos) > 5:
        todos = todos[:5]
    elif len(todos) < 5:
        return []
        
    return todos
