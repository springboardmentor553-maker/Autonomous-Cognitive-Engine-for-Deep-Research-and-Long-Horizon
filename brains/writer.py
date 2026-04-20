"""
Writer Agent - OpenAI
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

def _read(filename):
    p = FS_DIR / filename
    return p.read_text(encoding="utf-8") if p.exists() else ""

def _list():
    return [f.name for f in FS_DIR.iterdir() if f.is_file()]

def create_writer():
    llm = ChatOpenAI(model=OPENAI_MODEL,
                     temperature=0.7, max_tokens=400)

    system_message = "You are a professional writer. Write a concise, well-structured report. Be brief and clear."

    def writer_node(state):
        user_task = state.get("user_task", "Write report")
        print("[WRITER] Creating final report")

        research_files   = [f for f in _list() if f.startswith("research_")]
        research_content = ""
        for fname in sorted(research_files):
            text = _read(fname)[:300]  # trim each file to 300 chars
            research_content += f"\n[{fname}]\n{text}\n"

        response = llm.invoke([
            SystemMessage(content=system_message),
            HumanMessage(content=f"Task: {user_task}\n\nResearch:\n{research_content}\n\nWrite a report with title, 3 sections, conclusion. Max 300 words.")
        ])
        content  = response.content
        _write("final_report.txt", content)
        print("[WRITER] Created: final_report.txt")

        created_files = state.get("created_files", [])
        created_files.append("final_report.txt")
        return {**state, "created_files": created_files, "final_output": content,
                "messages": state.get("messages", []) + [response]}

    return writer_node
