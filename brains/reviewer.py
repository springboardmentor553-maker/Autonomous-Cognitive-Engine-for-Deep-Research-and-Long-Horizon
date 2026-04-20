"""
Reviewer Agent - OpenAI
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
    return p.read_text(encoding="utf-8") if p.exists() else "No report found"

def create_reviewer():
    llm = ChatOpenAI(model=OPENAI_MODEL,
                     temperature=0.3, max_tokens=200)

    system_message = "You are a quality reviewer. Give a brief review in 3-5 sentences. End with: Verdict: Approved or Verdict: Needs Revision."

    def reviewer_node(state):
        print("[REVIEWER] Reviewing final report")
        report   = _read("final_report.txt")[:500]  # trim to 500 chars
        response = llm.invoke([
            SystemMessage(content=system_message),
            HumanMessage(content=f"Review this report (be brief):\n\n{report}")
        ])
        content  = response.content
        _write("review.txt", content)
        print("[REVIEWER] Created: review.txt")

        created_files = state.get("created_files", [])
        created_files.append("review.txt")
        return {**state, "created_files": created_files,
                "messages": state.get("messages", []) + [response]}

    return reviewer_node
