"""
write_todos Tool - Milestone 1 (preserved for Milestone 2)

Calls the LLM to break any complex task into exactly 5 structured,
actionable TODO steps and returns them as a list of dicts.
"""

import os
import json
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate

load_dotenv()

# Lazy-initialized LLM (initialized on first call, not at import time)
_llm = None

def _get_llm():
    global _llm
    if _llm is None:
        _llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)
    return _llm

planning_prompt = PromptTemplate(
    input_variables=["task"],
    template="""
Break the following task into exactly 5 clear, specific, actionable TODO steps.
Each step must start with a strong action verb.
Avoid repetition.
Return the output STRICTLY as a JSON array of strings — no markdown, no extra text.

Task: {task}

Return format example:
["Step one here", "Step two here", "Step three here", "Step four here", "Step five here"]
"""
)


def write_todos(task: str) -> str:
    """
    Generate exactly 5 structured TODO steps for the given task.

    Args:
        task: The complex task description.

    Returns:
        String representation of a list of todo dicts with 'task' and 'status'.
    """
    prompt_text = planning_prompt.format(task=task)
    response = _get_llm().invoke(prompt_text)
    
    if isinstance(response.content, list):
        # Join list elements if they are strings, or extract text from dicts
        raw = ""
        for item in response.content:
            if isinstance(item, str):
                raw += item
            elif isinstance(item, dict) and "text" in item:
                raw += item["text"]
    else:
        raw = response.content.strip()

    # Strip markdown code fences if present
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
        raw = raw.strip()

    try:
        steps = json.loads(raw)
        if not isinstance(steps, list):
            raise ValueError("Expected a list")
    except Exception:
        # Fallback: split by newlines
        steps = [line.strip("- •0123456789.) ").strip()
                 for line in raw.split("\n") if line.strip()]

    # Ensure exactly 5 steps
    steps = steps[:5]
    while len(steps) < 5:
        steps.append(f"Review and finalize step {len(steps) + 1}")

    todos = [{"task": step, "status": "pending"} for step in steps]
    return str(todos)
