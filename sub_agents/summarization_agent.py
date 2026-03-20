"""
sub_agents/summarization_agent.py - Specialized Summarization Sub-Agent
Milestone 3: Sub-Agent Delegation

This sub-agent is invoked by the supervisor to summarize large or complex content.
It runs as an isolated LangGraph graph with a minimal, focused context.

Interface:
    run_summarization_agent(task: str, context: str = "") -> str
"""

import os
import time
import json
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from typing import Annotated
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict


# ─────────────────────────────────────────────
# Sub-Agent State
# ─────────────────────────────────────────────

class SummarizationState(TypedDict):
    messages: Annotated[list, add_messages]
    result: str


# ─────────────────────────────────────────────
# Sub-Agent System Prompt
# ─────────────────────────────────────────────

SUMMARIZATION_SYSTEM_PROMPT = """You are a Summarization Specialist Agent.

Your ONLY job is to produce clear, accurate, and well-structured summaries.

INSTRUCTIONS:
1. Read the provided content or context carefully.
2. Identify the key points, themes, and important details.
3. Write a structured summary that includes:
   - A 2-3 sentence overview
   - Bullet-point key findings or themes (at least 3-5 points)
   - A brief conclusion or implication

FORMAT RULES:
- Use clear headings and bullet points
- Be comprehensive but concise
- Preserve important facts, numbers, and names
- Do NOT add opinions or speculations — only summarize what is given

OUTPUT: Produce only the final summary. Do not explain your process."""


# ─────────────────────────────────────────────
# Internal Tools
# ─────────────────────────────────────────────

_summary_scratch: dict[str, str] = {}


@tool
def save_summary_note(note: str) -> str:
    """
    Save an intermediate summarization note for later consolidation.
    
    Args:
        note: The partial summary or key point to save.
    
    Returns:
        Confirmation of the save.
    """
    key = f"note_{len(_summary_scratch)}"
    _summary_scratch[key] = note
    return json.dumps({
        "success": True,
        "saved_key": key,
        "total_notes": len(_summary_scratch)
    })


@tool
def get_all_notes() -> str:
    """
    Retrieve all previously saved summarization notes.
    
    Returns:
        JSON with all saved notes concatenated.
    """
    if not _summary_scratch:
        return json.dumps({"notes": {}, "message": "No notes saved yet."})
    return json.dumps({"notes": _summary_scratch, "count": len(_summary_scratch)})


SUMMARIZATION_TOOLS = [save_summary_note, get_all_notes]


# ─────────────────────────────────────────────
# Sub-Agent Graph
# ─────────────────────────────────────────────

def _build_summarization_graph():
    google_api_key = os.getenv("GOOGLE_API_KEY")
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash",   # switched from flash-lite (daily quota exhausted)
        google_api_key=google_api_key,
        temperature=0,
    )
    llm_with_tools = llm.bind_tools(SUMMARIZATION_TOOLS)

    def agent_node(state: SummarizationState) -> SummarizationState:
        messages = state["messages"]
        if not messages or not isinstance(messages[0], SystemMessage):
            messages = [SystemMessage(content=SUMMARIZATION_SYSTEM_PROMPT)] + list(messages)
        for attempt in range(5):
            try:
                response = llm_with_tools.invoke(messages)
                time.sleep(2)  # proactive pacing: 30 RPM = 1 call / 2s
                break
            except Exception as e:
                if ("429" in str(e) or "RESOURCE_EXHAUSTED" in str(e)) and attempt < 4:
                    wait = 15 * (2 ** attempt)
                    print(f"  ⏳  [summarization_agent] Rate limit — waiting {wait}s...")
                    time.sleep(wait)
                else:
                    raise
        return {"messages": [response], "result": state.get("result", "")}

    def tool_node_fn(state: SummarizationState) -> SummarizationState:
        tool_node = ToolNode(SUMMARIZATION_TOOLS)
        result = tool_node.invoke(state)
        return {**result, "result": state.get("result", "")}

    def should_continue(state: SummarizationState) -> str:
        last = state["messages"][-1]
        if hasattr(last, "tool_calls") and last.tool_calls:
            return "tools"
        return END

    graph = StateGraph(SummarizationState)
    graph.add_node("agent", agent_node)
    graph.add_node("tools", tool_node_fn)
    graph.set_entry_point("agent")
    graph.add_conditional_edges("agent", should_continue, {"tools": "tools", END: END})
    graph.add_edge("tools", "agent")
    return graph.compile()


# ─────────────────────────────────────────────
# Public Interface
# ─────────────────────────────────────────────

def run_summarization_agent(task: str, context: str = "") -> str:
    """
    Run the Summarization Sub-Agent on the given task.

    Args:
        task   : Description of what to summarize.
        context: The actual content to summarize (optional — agent uses its knowledge if empty).

    Returns:
        A plain-text summary string.
    """
    global _summary_scratch
    _summary_scratch = {}  # reset scratch for each run

    graph = _build_summarization_graph()

    prompt = task
    if context:
        prompt += f"\n\n--- CONTENT TO SUMMARIZE ---\n{context}"

    initial_state: SummarizationState = {
        "messages": [HumanMessage(content=prompt)],
        "result": "",
    }

    final_state = graph.invoke(initial_state)

    # Extract the last AI message as the result
    for msg in reversed(final_state["messages"]):
        if isinstance(msg, AIMessage) and msg.content:
            return msg.content

    return "Summarization sub-agent produced no output."
