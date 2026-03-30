"""
Dynamic Planning Tool - write_todos

This tool uses the LLM to break down complex tasks into structured TODO steps.
It does NOT use hardcoded responses - the LLM generates the plan dynamically.
"""

from typing import List, Dict
from langchain_core.prompts import PromptTemplate
from langchain_core.tools import tool
from langgraph.types import Command
import os
from dotenv import load_dotenv

load_dotenv() # This sucks in the LangSmith keys from your .env file

# Initialize LLM for planning
from langchain_google_genai import ChatGoogleGenerativeAI
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

@tool
def write_todos(task: str):
    """
    Dynamically generate TODO items for a given task using LLM.
    ALWAYS call this tool first to plan out the research steps.
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
            
    # MILESTONE 4 UPGRADE: 
    # Instead of returning a raw list, return a Command to update the AgentState
    return Command(
        update={
            "todos": todos
        }
    )


# For direct testing
if __name__ == "__main__":
    test_task = "Build an AI chatbot architecture"
    
    # Because it is now a LangChain @tool, we use .invoke() and pass a dictionary
    result = write_todos.invoke({"task": test_task})
    
    print(f"Task: {test_task}")
    print(f"Generated TODOs (Pushing to State):")
    
    # Extract the payload from the Command object for testing visibility
    extracted_todos = result.update.get("todos", [])
    for i, todo in enumerate(extracted_todos, 1):
        print(f"  {i}. {todo['task']} [{todo['status']}]")