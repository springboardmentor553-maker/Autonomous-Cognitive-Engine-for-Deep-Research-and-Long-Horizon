"""
Supervisor Agent — ReAct reasoning node for the Autonomous Cognitive Engine.
"""

from __future__ import annotations

import time
import random
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.language_models import BaseChatModel

from core.llm import get_llm
from core.state import AgentState
from tools.write_todos import write_todos
from tools.file_system_tools import ls, read_file, write_file, edit_file
from tools.tavily_search import tavily_search
from tools.delegate_task import delegate_task, get_available_agents


# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are an Autonomous Cognitive Engine — a thorough senior research agent.

## Tools
- write_todos: Plan tasks. Call this FIRST and ALONE. Always create 4-6 specific research tasks.
- tavily_search: Quick single-topic web search.
- delegate_task: Hand a research topic to the web_search_agent sub-agent.
- write_file: Save results to the virtual file system.
- read_file: Read a saved file.
- edit_file: Update an existing file.
- ls: List saved files.

## Sub-Agent (used via delegate_task)
- agent_name="web_search_agent"
  task = ONE short research topic, under 60 characters.
  Returns a structured research block with findings and sources.

## MANDATORY WORKFLOW — FOLLOW THIS EXACTLY EVERY TIME

### STEP 1 — ALWAYS call write_todos FIRST
- Create 4 to 6 specific TODO tasks covering DIFFERENT aspects of the topic.
- Example for "Indian stock market":
  ["BSE NSE overview and history",
   "Indian stock market 2024 performance",
   "Major indices Sensex Nifty analysis",
   "Foreign vs domestic investor trends",
   "Future outlook and risks"]
- Never skip this step. Never proceed without a TODO list.

### STEP 2 — Research EACH TODO
For each TODO task:
  a) Call delegate_task(agent_name="web_search_agent", task="<short topic under 60 chars>")
  b) Then immediately call write_file to save that result

### STEP 3 — Read all files back
Call read_file for EACH saved file, one per step.

### STEP 4 — Write comprehensive final answer
Write a detailed report using ALL research gathered. Minimum 800 words.

## STRICT RULES
- ONE tool per step — no exceptions.
- task string in delegate_task must be under 60 characters, noun phrase only.
- Never pass filenames as task. Never call summarization_agent.
- Never skip steps. Never skip any TODO.

## FINAL ANSWER REQUIREMENTS — THIS IS CRITICAL
Your final message MUST:
- Be at minimum 800 words of detailed prose
- Have a proper title (# heading)
- Have a section for EACH TODO task researched (## subheadings)
- Include specific facts, numbers, statistics from the research files
- Have an Introduction and a Conclusion
- Cite sources where available
- NEVER summarize in 3-4 sentences — write a FULL detailed report
- Use ALL the information from every file you read back
"""

# ---------------------------------------------------------------------------
# Tool list
# ---------------------------------------------------------------------------

ALL_TOOLS = [
    write_todos,
    tavily_search,
    delegate_task,
    ls,
    read_file,
    write_file,
    edit_file,
]


def get_agent_runnable() -> BaseChatModel:
    """Return the LLM with all tools bound, one call at a time."""
    llm = get_llm()
    return llm.bind_tools(ALL_TOOLS, parallel_tool_calls=False)


# ---------------------------------------------------------------------------
# Simple message detector
# ---------------------------------------------------------------------------

_SIMPLE_PATTERNS = {
    "hello", "hi", "hey", "hiya", "howdy", "bye", "goodbye",
    "thanks", "thank you", "cheers", "ok", "okay", "cool",
    "yes", "no", "sure", "nope", "yep",
}

_RESEARCH_KEYWORDS = {
    "report", "research", "find", "search", "explain", "compare",
    "analyze", "analyse", "summarize", "summarise", "write", "list",
    "what", "why", "how", "when", "where", "who", "tell", "give",
    "delegate", "specialist", "effect", "impact", "trends", "market",
    "analyse", "overview", "history", "future", "deep",
}

_SIMPLE_PROMPT = "You are a helpful assistant. Reply briefly and conversationally. No tools."


def _is_simple_message(text: str) -> bool:
    """Return True if the message is chitchat with no research intent."""
    cleaned = text.strip().lower().rstrip("!.,?")
    if cleaned in _SIMPLE_PATTERNS:
        return True
    words = cleaned.split()
    if len(words) <= 4 and not any(w in _RESEARCH_KEYWORDS for w in words):
        return True
    return False


# ---------------------------------------------------------------------------
# Retry helper
# ---------------------------------------------------------------------------

def _invoke_with_retry(agent, messages, max_retries: int = 3):
    """
    Invoke the agent with exponential backoff retry on tool_use_failed.

    Groq's llama models occasionally fail to generate valid function call
    XML. Retrying with a short delay usually succeeds on the next attempt.
    On the second retry, message history is trimmed to reduce context pressure.
    """
    last_exc = None
    for attempt in range(max_retries):
        try:
            return agent.invoke(messages)
        except Exception as exc:
            err_str = str(exc)
            if "tool_use_failed" not in err_str and "Failed to call a function" not in err_str:
                raise
            last_exc = exc
            wait = (2 ** attempt) + random.uniform(0, 1)
            print(f"\n  [retry {attempt + 1}/{max_retries}] tool_use_failed — retrying in {wait:.1f}s…")
            time.sleep(wait)
            if attempt == 1 and len(messages) > 8:
                trimmed = messages[:2] + messages[-6:]
                print(f"  [retry] trimmed context from {len(messages)} to {len(trimmed)} messages")
                messages = trimmed
    raise last_exc


# ---------------------------------------------------------------------------
# Supervisor node
# ---------------------------------------------------------------------------

def supervisor_node(state: AgentState) -> dict:
    """
    LangGraph node: one reasoning step for the supervisor agent.

    - Simple chitchat: direct LLM reply, no tools
    - Research tasks: full ReAct loop with planning enforcement
    """
    # --- Simple message fast path ---
    human_messages = [m for m in state["messages"] if hasattr(m, "type") and m.type == "human"]
    if human_messages:
        last_human = str(human_messages[-1].content)
        if _is_simple_message(last_human):
            llm = get_llm()
            response = llm.invoke([
                SystemMessage(content=_SIMPLE_PROMPT),
                HumanMessage(content=last_human),
            ])
            return {"messages": [response]}

    messages = [SystemMessage(content=SYSTEM_PROMPT)] + list(state["messages"])

    # --- Enforce TODO planning if agent skipped it ---
    todos = state.get("todos", [])
    has_tool_calls_yet = any(
        hasattr(m, "tool_calls") and m.tool_calls
        for m in state["messages"]
    )

    if not has_tool_calls_yet and not todos:
        # First step — force the model to plan before anything else
        enforce = SystemMessage(content=(
            "MANDATORY FIRST ACTION: You MUST call write_todos RIGHT NOW. "
            "Create 4 to 6 specific research tasks covering different aspects "
            "of the topic. Do NOT call any other tool. Do NOT search yet. "
            "Call write_todos ONLY."
        ))
        messages = messages + [enforce]

    elif todos and not state.get("final_output"):
        # Mid-execution — check if all todos are done and push for final answer
        pending = [t for t in todos if t["status"] != "done"]
        files = state.get("files", {})
        all_done = len(pending) == 0
        has_files = len(files) > 0

        if all_done and has_files:
            # All research done — push for comprehensive final answer
            file_summary = ", ".join(f'"{k}"' for k in files.keys())
            push = SystemMessage(content=(
                f"All research complete. You have {len(files)} research files: {file_summary}. "
                f"Now write your COMPREHENSIVE FINAL REPORT — minimum 800 words. "
                f"Use ALL information from every file. Include specific facts, numbers, "
                f"statistics. Write proper sections with ## headings for each topic. "
                f"Do NOT call any tools. Write the full detailed report NOW."
            ))
            messages = messages + [push]

        elif not all_done:
            # Still have todos — remind to keep working
            done_count = len([t for t in todos if t["status"] == "done"])
            delegation_count = len(state.get("delegation_history", []))
            pending_tasks = [t["task"] for t in pending]
            remind = SystemMessage(content=(
                f"Progress: {done_count}/{len(todos)} tasks done, "
                f"{delegation_count} delegations completed. "
                f"Next pending tasks: {pending_tasks[:2]}. "
                f"Continue: call delegate_task for the next pending task. "
                f"task must be a short noun phrase under 60 chars."
            ))
            messages = messages + [remind]

    agent = get_agent_runnable()
    response = _invoke_with_retry(agent, messages)
    return {"messages": [response]}
