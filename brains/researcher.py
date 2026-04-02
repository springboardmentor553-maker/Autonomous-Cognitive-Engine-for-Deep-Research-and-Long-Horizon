"""
Researcher Agent - Ollama (llama3.2:1b)
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

def create_researcher():
    llm = ChatOllama(model=OLLAMA_MODEL, base_url=OLLAMA_BASE_URL,
                     temperature=0.7, num_predict=300)  # hard cap at 300 tokens

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
