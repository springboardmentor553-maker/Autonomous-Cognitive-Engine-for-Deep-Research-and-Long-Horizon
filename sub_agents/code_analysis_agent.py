"""
sub_agents/code_analysis_agent.py - Specialized Code Analysis Sub-Agent
Milestone 3: Sub-Agent Delegation

This sub-agent analyzes, reviews, explains, or drafts code snippets.
It runs in its own isolated LangGraph context.

Interface:
    run_code_analysis_agent(task: str, context: str = "") -> str
"""

import os
import time
import json
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from typing import Annotated
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict


# ─────────────────────────────────────────────
# Sub-Agent State
# ─────────────────────────────────────────────

class CodeAnalysisState(TypedDict):
    messages: Annotated[list, add_messages]
    analysis_notes: list


# ─────────────────────────────────────────────
# Sub-Agent System Prompt
# ─────────────────────────────────────────────

CODE_ANALYSIS_SYSTEM_PROMPT = """You are a Code Analysis Specialist Agent.

Your expertise covers: code review, architecture analysis, bug detection, refactoring advice,
performance analysis, security review, and writing/drafting code.

TASK APPROACH:
1. Carefully read the code or coding task description.
2. Use `log_analysis_note` to record individual observations as you analyze.
3. Cover these dimensions (where relevant):
   - **Correctness**: Does it do what it's supposed to?
   - **Architecture**: Is the design clean and maintainable?
   - **Performance**: Are there bottlenecks or inefficiencies?
   - **Security**: Are there vulnerabilities or risks?
   - **Best Practices**: Does it follow language/framework conventions?
4. Review your notes with `get_analysis_notes`.
5. Produce a final code analysis report with:
   - Brief summary of what the code does
   - Detailed findings by category
   - Specific recommendations with example fixes where applicable
   - Overall quality assessment (1-10 scale)

If writing new code, produce clean, well-commented, production-ready code."""


# ─────────────────────────────────────────────
# Internal Tools
# ─────────────────────────────────────────────

_analysis_notes: list[dict] = []
_note_counter = 0


@tool
def log_analysis_note(category: str, observation: str, severity: str = "info") -> str:
    """
    Log an observation or finding during code analysis.

    Args:
        category   : Category such as "correctness", "performance", "security",
                     "architecture", "style", "suggestion".
        observation: The specific observation or finding.
        severity   : "critical", "warning", "info", or "suggestion".

    Returns:
        Confirmation with note ID.
    """
    global _note_counter
    _note_counter += 1
    note = {
        "id": _note_counter,
        "category": category,
        "observation": observation,
        "severity": severity,
    }
    _analysis_notes.append(note)
    return json.dumps({
        "success": True,
        "note_id": _note_counter,
        "total_notes": len(_analysis_notes)
    })


@tool
def get_analysis_notes() -> str:
    """
    Retrieve all logged code analysis notes, grouped by category.

    Returns:
        JSON with all analysis notes.
    """
    if not _analysis_notes:
        return json.dumps({
            "notes": [],
            "message": "No analysis notes yet. Use log_analysis_note first."
        })

    by_category: dict[str, list] = {}
    for n in _analysis_notes:
        cat = n["category"]
        if cat not in by_category:
            by_category[cat] = []
        by_category[cat].append(n)

    return json.dumps({
        "notes": _analysis_notes,
        "by_category": by_category,
        "total": len(_analysis_notes),
    }, indent=2)


@tool
def check_code_pattern(pattern_name: str, description: str) -> str:
    """
    Check whether a specific code pattern or anti-pattern exists in the code.

    Args:
        pattern_name: Name of the pattern (e.g., "singleton", "god_object", "n+1_query").
        description : Brief description of what you're looking for.

    Returns:
        Confirmation to continue analysis with this pattern in mind.
    """
    return json.dumps({
        "pattern": pattern_name,
        "description": description,
        "instruction": (
            f"Look for '{pattern_name}' pattern in the provided code. "
            f"Description: {description}. Record your findings using log_analysis_note."
        )
    })


CODE_ANALYSIS_TOOLS = [log_analysis_note, get_analysis_notes, check_code_pattern]


# ─────────────────────────────────────────────
# Sub-Agent Graph
# ─────────────────────────────────────────────

def _build_code_analysis_graph():
    google_api_key = os.getenv("GOOGLE_API_KEY")
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash",   # switched from flash-lite (daily quota exhausted)
        google_api_key=google_api_key,
        temperature=0,
    )
    llm_with_tools = llm.bind_tools(CODE_ANALYSIS_TOOLS)

    def agent_node(state: CodeAnalysisState) -> CodeAnalysisState:
        messages = state["messages"]
        if not messages or not isinstance(messages[0], SystemMessage):
            messages = [SystemMessage(content=CODE_ANALYSIS_SYSTEM_PROMPT)] + list(messages)
        for attempt in range(5):
            try:
                response = llm_with_tools.invoke(messages)
                time.sleep(2)  # proactive pacing: 30 RPM = 1 call / 2s
                break
            except Exception as e:
                if ("429" in str(e) or "RESOURCE_EXHAUSTED" in str(e)) and attempt < 4:
                    wait = 15 * (2 ** attempt)
                    print(f"  ⏳  [code_analysis_agent] Rate limit — waiting {wait}s...")
                    time.sleep(wait)
                else:
                    raise
        return {"messages": [response], "analysis_notes": state.get("analysis_notes", [])}

    def tool_node_fn(state: CodeAnalysisState) -> CodeAnalysisState:
        tool_node = ToolNode(CODE_ANALYSIS_TOOLS)
        result = tool_node.invoke(state)
        return {**result, "analysis_notes": state.get("analysis_notes", [])}

    def should_continue(state: CodeAnalysisState) -> str:
        last = state["messages"][-1]
        if hasattr(last, "tool_calls") and last.tool_calls:
            return "tools"
        return END

    graph = StateGraph(CodeAnalysisState)
    graph.add_node("agent", agent_node)
    graph.add_node("tools", tool_node_fn)
    graph.set_entry_point("agent")
    graph.add_conditional_edges("agent", should_continue, {"tools": "tools", END: END})
    graph.add_edge("tools", "agent")
    return graph.compile()


# ─────────────────────────────────────────────
# Public Interface
# ─────────────────────────────────────────────

def run_code_analysis_agent(task: str, context: str = "") -> str:
    """
    Run the Code Analysis Sub-Agent on the given task.

    Args:
        task   : The coding task description (review, draft, explain, refactor, etc.).
        context: The code to analyze, or additional context.

    Returns:
        A plain-text code analysis report or generated code string.
    """
    global _analysis_notes, _note_counter
    _analysis_notes = []
    _note_counter = 0

    graph = _build_code_analysis_graph()

    prompt = task
    if context:
        prompt += f"\n\n--- CODE / CONTEXT ---\n{context}"

    initial_state: CodeAnalysisState = {
        "messages": [HumanMessage(content=prompt)],
        "analysis_notes": [],
    }

    final_state = graph.invoke(initial_state)

    for msg in reversed(final_state["messages"]):
        if isinstance(msg, AIMessage) and msg.content:
            return msg.content

    return "Code analysis sub-agent produced no output."
