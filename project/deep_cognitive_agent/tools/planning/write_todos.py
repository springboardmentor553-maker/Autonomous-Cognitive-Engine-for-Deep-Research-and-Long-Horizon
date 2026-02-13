"""
Dynamic Planning Tool - write_todos

This tool uses the LLM to break down complex tasks into structured TODO steps.
It does NOT use hardcoded responses - the LLM generates the plan dynamically.
"""

from typing import List, Dict
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
import os


# Initialize LLM for planning
llm = ChatOpenAI(model="gpt-4o", temperature=0)

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
    """
    # Use LLM to generate the plan dynamically
    formatted_prompt = planning_prompt.format(task=task)
    response = llm.invoke(formatted_prompt)
    
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


# For direct testing
if __name__ == "__main__":
    test_task = "Build an AI chatbot architecture"
    result = write_todos(test_task)
    print(f"Task: {test_task}")
    print(f"Generated TODOs:")
    for i, todo in enumerate(result, 1):
        print(f"  {i}. {todo['task']} [{todo['status']}]")
