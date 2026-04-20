"""
Researcher Agent - OpenAI
"""
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from pathlib import Path

load_dotenv()
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
FS_DIR          = Path("virtual_fs")
FS_DIR.mkdir(exist_ok=True)

def _write(filename, content):
    (FS_DIR / filename).write_text(content, encoding="utf-8")

def create_researcher():
    llm = ChatOpenAI(model=OPENAI_MODEL,
                     temperature=0.7, max_tokens=300)

    system_message = "You are a research agent. Write concise, factual research findings in 3-4 short paragraphs. Be brief."

    def researcher_node(state):
        user_task    = state.get("user_task", "Research topic")
        current_step = state.get("current_step", 1)
        print(f"[RESEARCHER] Processing step {current_step}")

        response = llm.invoke([
            SystemMessage(content=system_message),
            HumanMessage(content=f"Research Task: {user_task}\nStep {current_step} of 3.\nWrite 3-4 short paragraphs (max 150 words total).")
        ])
        content  = response.content
        filename = f"research_step{current_step}.txt"
        _write(filename, content)
        print(f"[RESEARCHER] Created: {filename}")

        created_files = state.get("created_files", [])
        created_files.append(filename)
        return {**state, "created_files": created_files,
                "messages": state.get("messages", []) + [response]}

    return researcher_node
