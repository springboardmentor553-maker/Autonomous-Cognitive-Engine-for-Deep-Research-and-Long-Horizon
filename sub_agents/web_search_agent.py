"""
sub_agents/web_search_agent.py - Specialized Web/Research Search Sub-Agent
Milestone 3: Sub-Agent Delegation

This sub-agent performs research and knowledge-gathering tasks.
It uses the LLM's built-in knowledge and structured research tools.
(Tavily can be swapped in by updating the search tool if the API key is available.)

Interface:
    run_web_search_agent(task: str, context: str = "") -> str
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

class WebSearchState(TypedDict):
    messages: Annotated[list, add_messages]
    findings: list


# ─────────────────────────────────────────────
# Sub-Agent System Prompt
# ─────────────────────────────────────────────

WEB_SEARCH_SYSTEM_PROMPT = """You are a Research Specialist Agent.

Your ONLY job is to perform thorough, factual research on the given topic and return structured findings.

RESEARCH PROCESS:
1. Analyze the research task carefully.
2. Use `record_finding` to save each distinct finding as you research.
3. Cover multiple angles: background, current state, key facts, statistics, expert opinions, trends.
4. After recording all findings, call `get_all_findings` to review them.
5. Produce a final research report organized as:
   - **Topic Overview**: 2-3 sentence introduction
   - **Key Findings**: numbered list of your most important discoveries
   - **Supporting Details**: relevant facts, figures, and context
   - **Conclusion**: what the findings mean or imply

QUALITY STANDARDS:
- Be factual and specific — cite specific names, numbers, dates when relevant
- Cover at least 5 distinct findings
- Do NOT fabricate statistics — only state what you know with confidence
- Clearly separate established facts from emerging trends"""


# ─────────────────────────────────────────────
# Internal Tools
# ─────────────────────────────────────────────

_findings_store: list[dict] = []
_finding_counter = 0


@tool
def record_finding(topic: str, finding: str, confidence: str = "high") -> str:
    """
    Record a research finding for the current task.

    Args:
        topic     : The sub-topic or category this finding belongs to.
        finding   : The actual research finding or fact.
        confidence: "high", "medium", or "low" — how confident you are.

    Returns:
        Confirmation with the finding ID.
    """
    global _finding_counter
    _finding_counter += 1
    entry = {
        "id": _finding_counter,
        "topic": topic,
        "finding": finding,
        "confidence": confidence,
    }
    _findings_store.append(entry)
    return json.dumps({
        "success": True,
        "finding_id": _finding_counter,
        "total_findings": len(_findings_store)
    })


@tool
def get_all_findings() -> str:
    """
    Retrieve all recorded research findings.

    Returns:
        JSON with all findings grouped by topic.
    """
    if not _findings_store:
        return json.dumps({
            "findings": [],
            "message": "No findings recorded yet. Use record_finding first."
        })

    # Group by topic
    by_topic: dict[str, list] = {}
    for f in _findings_store:
        topic = f["topic"]
        if topic not in by_topic:
            by_topic[topic] = []
        by_topic[topic].append(f)

    return json.dumps({
        "findings": _findings_store,
        "by_topic": by_topic,
        "total": len(_findings_store),
    }, indent=2)


@tool
def web_lookup(query: str) -> str:
    """
    Perform a knowledge lookup on a specific query.
    Uses the LLM's built-in knowledge to answer research questions.

    Args:
        query: The specific research question or topic to look up.

    Returns:
        A structured answer based on available knowledge.
    """
    # This tool triggers an LLM self-reflection via the agent loop.
    # In production, swap this for a Tavily/DuckDuckGo API call.
    return json.dumps({
        "query": query,
        "instruction": (
            "Use your built-in knowledge to answer this query. "
            "Be specific with facts, dates, and numbers where possible. "
            f"Query: {query}"
        )
    })


WEB_SEARCH_TOOLS = [record_finding, get_all_findings, web_lookup]


# ─────────────────────────────────────────────
# Sub-Agent Graph
# ─────────────────────────────────────────────

def _build_web_search_graph():
    google_api_key = os.getenv("GOOGLE_API_KEY")
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash",   # switched from flash-lite (daily quota exhausted)
        google_api_key=google_api_key,
        temperature=0,
    )
    llm_with_tools = llm.bind_tools(WEB_SEARCH_TOOLS)

    def agent_node(state: WebSearchState) -> WebSearchState:
        messages = state["messages"]
        if not messages or not isinstance(messages[0], SystemMessage):
            messages = [SystemMessage(content=WEB_SEARCH_SYSTEM_PROMPT)] + list(messages)
        for attempt in range(5):
            try:
                response = llm_with_tools.invoke(messages)
                time.sleep(2)  # proactive pacing: 30 RPM = 1 call / 2s
                break
            except Exception as e:
                if ("429" in str(e) or "RESOURCE_EXHAUSTED" in str(e)) and attempt < 4:
                    wait = 15 * (2 ** attempt)
                    print(f"  ⏳  [web_search_agent] Rate limit — waiting {wait}s...")
                    time.sleep(wait)
                else:
                    raise
        return {"messages": [response], "findings": state.get("findings", [])}

    def tool_node_fn(state: WebSearchState) -> WebSearchState:
        tool_node = ToolNode(WEB_SEARCH_TOOLS)
        result = tool_node.invoke(state)
        return {**result, "findings": state.get("findings", [])}

    def should_continue(state: WebSearchState) -> str:
        last = state["messages"][-1]
        if hasattr(last, "tool_calls") and last.tool_calls:
            return "tools"
        return END

    graph = StateGraph(WebSearchState)
    graph.add_node("agent", agent_node)
    graph.add_node("tools", tool_node_fn)
    graph.set_entry_point("agent")
    graph.add_conditional_edges("agent", should_continue, {"tools": "tools", END: END})
    graph.add_edge("tools", "agent")
    return graph.compile()


# ─────────────────────────────────────────────
# Public Interface
# ─────────────────────────────────────────────

def run_web_search_agent(task: str, context: str = "") -> str:
    """
    Run the Web Search / Research Sub-Agent on the given task.

    Args:
        task   : The research question or topic to investigate.
        context: Optional additional context or constraints.

    Returns:
        A plain-text research report string.
    """
    global _findings_store, _finding_counter
    _findings_store = []
    _finding_counter = 0

    graph = _build_web_search_graph()

    prompt = task
    if context:
        prompt += f"\n\n--- ADDITIONAL CONTEXT ---\n{context}"

    initial_state: WebSearchState = {
        "messages": [HumanMessage(content=prompt)],
        "findings": [],
    }

    final_state = graph.invoke(initial_state)

    for msg in reversed(final_state["messages"]):
        if isinstance(msg, AIMessage) and msg.content:
            return msg.content

    return "Web search sub-agent produced no output."
