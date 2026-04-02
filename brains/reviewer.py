"""
Reviewer Agent - Ollama (llama3.2:1b)
"""
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage
from pathlib import Path

OLLAMA_MODEL    = "llama3.2:1b"
OLLAMA_BASE_URL = "http://localhost:11434"
FS_DIR          = Path("virtual_fs")
FS_DIR.mkdir(exist_ok=True)

def _write(filename, content):
    (FS_DIR / filename).write_text(content, encoding="utf-8")

def _read(filename):
    p = FS_DIR / filename
    return p.read_text(encoding="utf-8") if p.exists() else "No report found"

def create_reviewer():
    llm = ChatOllama(model=OLLAMA_MODEL, base_url=OLLAMA_BASE_URL,
                     temperature=0.3, num_predict=200)  # hard cap at 200 tokens

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
