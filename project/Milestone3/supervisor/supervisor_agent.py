import os
from dotenv import load_dotenv
from langgraph.prebuilt import create_react_agent
from langchain_google_genai import ChatGoogleGenerativeAI

from state.agent_state import AgentState
from tools.todo_tools import write_todos, update_todo
from tools.file_tools import write_file, read_file, edit_file, ls
from tools.delegation_tool import task
from supervisor.system_prompt import SYSTEM_PROMPT

load_dotenv()

llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    google_api_key=os.getenv("GOOGLE_API_KEY")
)

tools = [write_todos, update_todo, write_file, read_file, edit_file, ls, task]

supervisor_graph = create_react_agent(
    model=llm,
    tools=tools,
    state_schema=AgentState,
    prompt=SYSTEM_PROMPT,
)