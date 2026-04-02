import os
import json
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langgraph.prebuilt import create_react_agent

from tools import (
    write_todos, write_file, read_file, edit_file, ls, task,
    VFS, VirtualFileSystem
)
import tools as tools_module

load_dotenv()

# ══════════════════════════════════════════════════════════════
# LLM
# ══════════════════════════════════════════════════════════════

llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.1)

# ══════════════════════════════════════════════════════════════
# System Prompt — Full Milestone 4 Integration
# Mentor's system prompt pattern:
#   "When to delegate, when to store, when to retrieve files,
#    integrate all outputs before generating final response"
# ══════════════════════════════════════════════════════════════

SYSTEM_PROMPT = """You are a supervisor agent — an autonomous cognitive engine for deep research and long-horizon tasks.

FULL WORKFLOW (follow exactly):

STEP 1 — PLAN:
   Call write_todos FIRST to break the request into 4-6 structured TODO steps.
   Do nothing else before this.

STEP 2 — EXECUTE each TODO in order. For each step:
   - Research tasks → task(agent_name="research_agent", input_data="AI in healthcare")
   - Summarize tasks → task(agent_name="summarization_agent", input_data="<actual text from read_file>")
   - Simple reasoning → handle directly

STEP 3 — STORE after every task() call:
   write_file("research.txt", <result from task()>)
   write_file("summary.txt", <result from task()>)

STEP 4 — RETRIEVE before summarizing:
   read_file("research.txt") → get content → pass content to summarization_agent

STEP 5 — SYNTHESIZE at the end:
   ls() → read_file() each file → combine into final structured report

CORRECT task() examples:
   task(agent_name="research_agent", input_data="AI in healthcare diagnostics")
   task(agent_name="research_agent", input_data="quantum computing cybersecurity")
   task(agent_name="research_agent", input_data="climate change policies global")
   task(agent_name="summarization_agent", input_data="Key Facts: LLMs improve productivity...")

RULES:
- input_data must be plain text, no brackets, no special characters, no parentheses
- input_data for research_agent: 3-5 plain words describing the topic
- input_data for summarization_agent: actual text content, never a filename
- Always write_file immediately after every task() call
- Final response must combine ALL collected information into a structured report
"""

# ══════════════════════════════════════════════════════════════
# Build the integrated agent (all tools from M1 + M2 + M3)
# ══════════════════════════════════════════════════════════════

def build_agent():
    """Build the fully integrated supervisor agent with all milestone tools."""
    agent = create_react_agent(
        llm,
        tools=[write_todos, write_file, read_file, edit_file, ls, task]
    )
    return agent


# ══════════════════════════════════════════════════════════════
# Execution loop — runs one task through the full pipeline
# Mentor pattern:
#   User → Planning → Execution → Delegation → Storage →
#   Retrieval → Synthesis → Output
# ══════════════════════════════════════════════════════════════

def run_task(agent, user_request: str) -> dict:
    """
    Run one user request through the full integrated workflow.

    Returns:
        dict with result, delegation_log, files, final_output
    """
    # Reset VFS for each run
    tools_module.VFS = VirtualFileSystem()

    inputs = {
        "messages": [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=user_request)
        ]
    }

    result = agent.invoke(inputs)
    messages = result.get("messages", [])

    # ── Extract execution data ────────────────────────────────
    todos          = []
    delegation_log = []
    tool_seq       = []

    for msg in messages:
        if hasattr(msg, "tool_calls") and msg.tool_calls:
            for call in msg.tool_calls:
                tool_seq.append(call["name"])

        if hasattr(msg, "name"):
            if msg.name == "write_todos":
                try:
                    parsed = json.loads(msg.content)
                    todos  = parsed.get("todos", [])
                except Exception:
                    pass

            if msg.name == "task":
                try:
                    parsed = json.loads(msg.content)
                    delegation_log.append({
                        "agent":  parsed.get("agent", ""),
                        "status": parsed.get("status", ""),
                        "result": str(parsed.get("result", ""))[:200]
                    })
                except Exception:
                    pass

    # ── Extract final AI response ─────────────────────────────
    final_output = ""
    for msg in reversed(messages):
        if isinstance(msg, AIMessage):
            content = msg.content if isinstance(msg.content, str) else str(msg.content)
            if len(content) > 100:
                final_output = content
                break

    return {
        "request":        user_request,
        "todos":          todos,
        "delegation_log": delegation_log,
        "tool_sequence":  tool_seq,
        "files":          tools_module.VFS.list_files(),
        "final_output":   final_output,
        "messages":       messages
    }


# ══════════════════════════════════════════════════════════════
# Synthesis check — did agent read all files and combine them?
# Mentor: "read all files, combine into final report"
# ══════════════════════════════════════════════════════════════

def check_synthesis(tool_seq: list, files: dict) -> bool:
    """Check if agent performed the synthesis step correctly."""
    root_files = files.get("root", {})
    if not root_files:
        return False

    # read_file was called after write_file (retrieval happened)
    write_indices = [i for i, n in enumerate(tool_seq) if n == "write_file"]
    read_indices  = [i for i, n in enumerate(tool_seq) if n == "read_file"]

    retrieval_happened = any(
        r > w for w in write_indices for r in read_indices
    ) if write_indices and read_indices else False

    return retrieval_happened