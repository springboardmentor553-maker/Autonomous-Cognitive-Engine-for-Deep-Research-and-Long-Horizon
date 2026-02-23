"""
Dynamic Planning Tool - write_todos

This tool uses the LLM (Groq Llama 3.3 70B) to break down complex tasks into
structured TODO steps. It enforces STRICT JSON output from the LLM.
It does NOT use hardcoded responses - the LLM generates the plan dynamically.
"""

import json
from typing import List, Dict
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


# Initialize LLM for planning (Groq free tier - Llama 3.3 70B)
llm = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0,
    groq_api_key="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",
)

# Planning prompt template — enforces strict JSON output
# Updated planning prompt template — forces EXACTLY 5 steps
planning_prompt = PromptTemplate(
    input_variables=["task"],
    template="""You are a planning agent.

Break the following complex task into EXACTLY 5 logically ordered, clear, non-repeating, actionable steps.

STRICT OUTPUT RULES:
- Return ONLY valid JSON.
- The output MUST be a JSON array of EXACTLY 5 strings.
- Do NOT include markdown, code fences, or explanations.

Example output:
["Step one", "Step two", "Step three", "Step four", "Step five"]

Task: {task}"""
)


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
    # Use LLM to generate the plan dynamically
    formatted_prompt = planning_prompt.format(task=task)
    response = llm.invoke(formatted_prompt)

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

    if not isinstance(steps, list) or not all(isinstance(s, str) for s in steps):
        raise ValueError(
            f"LLM returned JSON but not a list of strings. Got: {type(steps)}"
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

# Run the test file
if __name__ == "__main__":
    test_task = "Build an AI chatbot architecture"
    result = write_todos(test_task)
    print(f"Task: {test_task}")
    print(f"Generated {len(result['todos'])} TODOs:")
    for i, todo in enumerate(result["todos"], 1):
        print(f"  {i}. {todo['task']} [{todo['status']}]")
