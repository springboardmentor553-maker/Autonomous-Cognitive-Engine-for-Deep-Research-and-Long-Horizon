from langchain.tools import Tool
from langgraph.prebuilt import create_react_agent
from langchain_groq import ChatGroq
import os
from dotenv import load_dotenv

from tools.planning.write_todos import write_todos

load_dotenv()

SYSTEM_PROMPT = """
You are an autonomous planning agent operating under strict planning-first protocol.

MANDATORY RULES:
1. For EVERY complex task, you MUST call write_todos tool FIRST.
2. NEVER answer directly without planning.
3. Planning is REQUIRED.
4. DO NOT skip tool invocation.
"""

llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0,
    groq_api_key=os.getenv("GROQ_API_KEY"),
)

write_todos_tool = Tool(
    name="write_todos",
    func=write_todos,
    description="Break complex tasks into structured TODO steps"
)

agent = create_react_agent(
    llm=llm,
    tools=[write_todos_tool],
    system_prompt=SYSTEM_PROMPT
)
