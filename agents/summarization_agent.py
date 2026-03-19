"""
Summarization Sub-Agent — Milestone 3.

A specialized LangGraph runnable that accepts long text and returns
a concise, structured summary.  It is registered in the sub-agent
registry and invoked exclusively through the delegate_task tool.

The agent uses a single LLM call (no tool loop needed) because
summarization is a direct text-in / text-out task.
"""

from __future__ import annotations

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import StateGraph, START, END
from typing_extensions import TypedDict

from core.llm import get_llm


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------

class SummarizationState(TypedDict):
    """Minimal state for the summarization sub-agent."""

    input_text: str   # text to summarise (set before invocation)
    summary: str      # output summary (set by the agent node)


# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

SUMMARIZATION_SYSTEM_PROMPT = """You are an expert summarization assistant.

Your job is to read the provided text and produce a concise, well-structured summary.

Rules:
- Keep the summary to 20–30% of the original length.
- Preserve all key facts, figures, and conclusions.
- Use clear section headings if the original has multiple topics.
- Write in plain, professional prose.
- Do NOT add information that is not in the original text.
- End with a one-sentence "Key Takeaway".
"""


# ---------------------------------------------------------------------------
# Node
# ---------------------------------------------------------------------------

def summarize_node(state: SummarizationState) -> dict:
    """
    Invoke the LLM to summarise ``state["input_text"]``.

    Parameters
    ----------
    state : SummarizationState

    Returns
    -------
    dict
        Partial state update setting ``summary``.
    """
    llm = get_llm()

    messages = [
        SystemMessage(content=SUMMARIZATION_SYSTEM_PROMPT),
        HumanMessage(content=f"Please summarise the following text:\n\n{state['input_text']}"),
    ]

    response = llm.invoke(messages)
    return {"summary": response.content}


# ---------------------------------------------------------------------------
# Graph
# ---------------------------------------------------------------------------

def build_summarization_graph():
    """
    Build and compile the summarization sub-agent graph.

    Returns
    -------
    CompiledGraph
        A single-node LangGraph that reads ``input_text`` and writes ``summary``.
    """
    builder = StateGraph(SummarizationState)
    builder.add_node("summarize", summarize_node)
    builder.add_edge(START, "summarize")
    builder.add_edge("summarize", END)
    return builder.compile()


# Module-level compiled graph — imported by the registry
summarization_graph = build_summarization_graph()


def run_summarization_agent(task: str) -> str:
    """
    Public entry point used by the delegate_task tool.

    Parameters
    ----------
    task : str
        The text to summarise (passed as the full task description).

    Returns
    -------
    str
        The generated summary.
    """
    result = summarization_graph.invoke({"input_text": task, "summary": ""})
    return result.get("summary", "ERROR: Summarization agent returned no output.")
