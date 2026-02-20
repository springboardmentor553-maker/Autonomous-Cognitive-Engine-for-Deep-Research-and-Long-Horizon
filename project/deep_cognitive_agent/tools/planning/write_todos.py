"""
Dynamic Planning Tool - write_todos

This tool uses the LLM (Groq Llama 3.3 70B) to break down complex tasks into
structured TODO steps. It enforces STRICT JSON output from the LLM.
It does NOT use hardcoded responses - the LLM generates the plan dynamically.
"""

import json
import re
import time
from typing import List, Dict
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


# Lazy-initialized LLM so importing this module doesn't crash
# when the API key hasn't been set yet.
_llm = None


def _get_llm():
    global _llm
    if _llm is None:
        _llm = ChatGroq(
            model="llama-3.1-8b-instant",
            temperature=0,
            groq_api_key=os.getenv("GROQ_API_KEY"),
        )
    return _llm

# Planning prompt template — enforces strict JSON output
planning_prompt = PromptTemplate(
    input_variables=["task"],
    template="""Break the following task into 4-6 clear, specific, actionable TODO steps.

Each step must start with a strong action verb such as:
Analyze, Collect, Break down, Design, Compare, Draft, Evaluate, Implement, Validate, Test, Review.

Avoid repetition. Steps must be non-overlapping, meaningful, and in logical order.

STRICT OUTPUT RULES:
- Return ONLY valid JSON.
- Do NOT include markdown, code fences, or explanations.
- The output MUST be a JSON array of strings.
- NOT a dictionary. NOT nested. NOT numbered.

Correct example:
["Analyze the task requirements and constraints", "Break the task into logical sub-tasks", "Determine the tools, data, or resources needed", "Sequence the sub-tasks into an executable order", "Review the plan for completeness and clarity"]

Task: {task}"""
)


def _parse_retry_after(err_str: str) -> int:
    """Extract recommended wait seconds from a Groq rate-limit error."""
    match = re.search(r"try again in (?:(\d+)m)?(\d+(?:\.\d+)?)s", err_str)
    if match:
        minutes = int(match.group(1) or 0)
        seconds = float(match.group(2))
        return int(minutes * 60 + seconds) + 2
    return 30


def write_todos(task: str) -> Dict:
    """
    Use this tool to decompose complex tasks into structured to-do lists
    before any execution. This tool MUST be called for complex tasks.

    Dynamically generates TODO items using the LLM with strict JSON parsing.

    Args:
        task: The complex task to break down into steps

    Returns:
        Dict with a 'todos' key containing a list of structured todo dicts.
        Example: {"todos": [{"task": "Research topic X", "status": "pending"}, ...]}
    """
    # Use LLM to generate the plan dynamically with retry on rate limits
    formatted_prompt = planning_prompt.format(task=task)

    max_retries = 3
    response = None
    for attempt in range(max_retries):
        try:
            response = _get_llm().invoke(formatted_prompt)
            break
        except Exception as e:
            err_str = str(e)
            is_rate_limit = "429" in err_str or "rate_limit" in err_str.lower()
            if is_rate_limit and attempt < max_retries - 1:
                wait = _parse_retry_after(err_str)
                print(f"  ⏳ [write_todos] Rate limited. Waiting {wait}s before retry {attempt + 2}/{max_retries}...")
                time.sleep(wait)
                continue
            raise

    # Extract content from AIMessage
    response_text = response.content if hasattr(response, 'content') else str(response)

    # Strip markdown code fences if the LLM accidentally wraps them
    cleaned = response_text.strip()
    if cleaned.startswith("```"):
        # Remove opening fence (e.g. ```json)
        cleaned = cleaned.split("\n", 1)[1] if "\n" in cleaned else cleaned[3:]
    if cleaned.endswith("```"):
        cleaned = cleaned[:-3]
    cleaned = cleaned.strip()

    # Strict JSON parsing — no silent fallback
    try:
        steps = json.loads(cleaned)
    except json.JSONDecodeError as e:
        raise ValueError(
            f"LLM did not return valid JSON. "
            f"Raw response:\n{response_text}\n\nJSON error: {e}"
        )

    if not isinstance(steps, list):
        raise ValueError(
            f"Planning output must be a list. Got: {type(steps)}"
        )

    if not all(isinstance(s, str) for s in steps):
        raise ValueError(
            f"All steps must be strings. Got types: {[type(s).__name__ for s in steps]}"
        )

    if len(steps) < 4 or len(steps) > 6:
        raise ValueError(
            f"Plan must contain 4-6 steps. Got {len(steps)} steps."
        )

    # Convert JSON list into structured dictionaries
    todos = [{"task": step, "status": "pending"} for step in steps]

    return {"todos": todos}


# For direct testing
if __name__ == "__main__":
    test_task = "Build an AI chatbot architecture"
    result = write_todos(test_task)
    print(f"Task: {test_task}")
    print(f"Generated {len(result['todos'])} TODOs:")
    for i, todo in enumerate(result["todos"], 1):
        print(f"  {i}. {todo['task']} [{todo['status']}]")
