from langchain_core.tools import Tool
from langgraph.prebuilt import create_react_agent
from langchain_groq import ChatGroq
import os
from dotenv import load_dotenv

# Ensure this path is correct relative to the agents folder
from tools.planning.write_todos import write_todos

load_dotenv()

SYSTEM_PROMPT = """
You are a specialized Planning Agent. 

STRICT OPERATING PROTOCOL:
1. You are FORBIDDEN from answering the user's request using your own knowledge.
2. For EVERY single request, you MUST call the 'write_todos' tool.
3. Your only goal is to trigger that tool and then present the result.
4. If you answer without calling 'write_todos', you have failed.
"""

def create_planning_agent():
    """Initializes and returns the LangGraph ReAct agent."""
    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=0,
        # It's better to keep the key in .env, but using your provided logic:
        groq_api_key=os.getenv("GROQ_API_KEY"), 
    )

    write_todos_tool = Tool(
        name="write_todos",
        func=write_todos,
        description="Break complex tasks into structured TODO steps"
    )

    # Returns the compiled graph
    return create_react_agent(
        model=llm,
        tools=[write_todos_tool],
        prompt=SYSTEM_PROMPT
    )