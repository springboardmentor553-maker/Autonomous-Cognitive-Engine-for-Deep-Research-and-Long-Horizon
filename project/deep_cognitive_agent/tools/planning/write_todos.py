"""
Dynamic Planning Tool - write_todos

<<<<<<< HEAD
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
    model="llama-3.3-70b-versatile",
    temperature=0,
    groq_api_key=os.getenv("GROQ_API_KEY"),
)

# Planning prompt template — enforces strict JSON output
planning_prompt = PromptTemplate(
    input_variables=["task"],
    template="""You are a planning agent.

Break the following complex task into 4 to 6 logically ordered, clear, non-repeating, actionable steps.

STRICT OUTPUT RULES:
- Return ONLY valid JSON.
- Do NOT include markdown, code fences, or explanations.
- The output MUST be a JSON array of strings.

Example output:
["Research the topic", "Identify key components", "Draft an outline", "Review and refine"]

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
=======
This tool uses the LLM to break down complex tasks into structured TODO steps.
It does NOT use hardcoded responses - the LLM generates the plan dynamically.
"""

from typing import List, Dict
# from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
import os
from dotenv import load_dotenv
load_dotenv() # This sucks in the LangSmith keys from your .env file


# Initialize LLM for planning
from langchain_google_genai import ChatGoogleGenerativeAI
# ...
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)

# Planning prompt template
planning_prompt = PromptTemplate(
    input_variables=["task"],
    template="""
You are a planning agent.

Break the following complex task into a logically ordered,
clear, non-repeating, actionable list of steps.

Return ONLY numbered steps. Each step should be specific and actionable.
Do not include any introduction or conclusion text.

Task: {task}
"""
)


def write_todos(task: str) -> List[Dict]:
    """
    Dynamically generate TODO items for a given task using LLM.
    
    Args:
        task: The complex task to break down into steps
        
    Returns:
        List[Dict]: A list of todo items with 'task' and 'status' keys
        Example: [{"task": "Research topic X", "status": "pending"}, ...]
>>>>>>> milestone-1-planner
    """
    # Use LLM to generate the plan dynamically
    formatted_prompt = planning_prompt.format(task=task)
    response = llm.invoke(formatted_prompt)
<<<<<<< HEAD

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
=======
    
    # Extract content from AIMessage
    response_text = response.content if hasattr(response, 'content') else str(response)
    
    # Parse the numbered steps from the response
    lines = response_text.strip().split("\n")
    steps = [line.strip() for line in lines if line.strip()]
    
    # Convert to structured todo format
    todos = []
    for step in steps:
        # Remove leading numbers, dots, and whitespace (e.g., "1. ", "2) ", etc.)
        clean_step = step.lstrip("0123456789.)- ").strip()
        if clean_step:  # Only add non-empty steps
            todos.append({
                "task": clean_step,
                "status": "pending"
            })
    
    return todos
>>>>>>> milestone-1-planner


# For direct testing
if __name__ == "__main__":
    test_task = "Build an AI chatbot architecture"
    result = write_todos(test_task)
    print(f"Task: {test_task}")
<<<<<<< HEAD
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
=======
    print(f"Generated TODOs:")
    for i, todo in enumerate(result, 1):
>>>>>>> milestone-1-planner
        print(f"  {i}. {todo['task']} [{todo['status']}]")
