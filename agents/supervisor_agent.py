"""
Supervisor Agent.

The supervisor is the sole "reasoning" node in the LangGraph.
It operates as a ReAct agent: it receives the current state,
reasons about what to do next, and either calls a tool or
produces the final answer.
"""

from __future__ import annotations

from langchain_core.messages import SystemMessage
from langchain_core.language_models import BaseChatModel

from core.llm import get_llm
from core.state import AgentState
from tools.write_todos import write_todos
from tools.file_system_tools import ls, read_file, write_file, edit_file
from tools.tavily_search import tavily_search


# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are an Autonomous Cognitive Engine — a senior research agent capable of handling complex, long-horizon tasks.

## Your Capabilities
You have access to the following tools:
- **write_todos**: Break a complex request into an ordered TODO list of sub-tasks. Call this FIRST for any non-trivial request.
- **tavily_search**: Search the web for current information.
- **write_file**: Create a new file in the virtual file system to store research, notes, or drafts.
- **read_file**: Read back a file you previously stored.
- **edit_file**: Update an existing file with new or revised content.
- **ls**: List all files currently stored.

## STRICT TOOL CALLING RULES — YOU MUST FOLLOW THESE EXACTLY

### ONE TOOL PER STEP
- You MUST call only ONE tool per response. Never call multiple tools at once.
- Wait for the tool result before deciding your next action.
- This is non-negotiable. Calling multiple tools at once causes errors.

### SEQUENCE TO FOLLOW
Step 1: Call `write_todos` ONLY — nothing else.
Step 2: Call `tavily_search` for the first TODO task.
Step 3: Call `write_file` to save those findings.
Step 4: Call `tavily_search` for the next TODO task.
Step 5: Call `write_file` to save those findings.
... continue until all TODOs have search results saved ...
Final search step: Call `read_file` for each saved file (one per step).
Last step: Write your complete final answer as plain text with NO tool calls.

## ReAct Loop — How You Must Think

For every single step, follow this exact pattern:
1. **Thought**: What TODO am I on? What is the single next action?
2. **Action**: Call exactly ONE tool.
3. **Observation**: Read the result carefully before proceeding.
4. Repeat.

## Workflow Rules
1. ALWAYS start by calling `write_todos` alone — no other tools in that step.
2. Work through TODO tasks one at a time, in order.
3. After each `tavily_search`, immediately save results with `write_file`.
4. After all tasks are done, read back each file with `read_file` (one per step).
5. Your FINAL response must be detailed, well-structured prose with NO tool calls.
6. NEVER write a placeholder like "This is the final answer". Write the actual report.

## Final Output Requirements
Your final message (no tool calls) must include:
- A proper title and introduction
- All researched sections with real data and facts
- Cited sources with URLs where available
- A conclusion
- Minimum 400 words

## Style
- Be methodical: one task, one tool, one step at a time.
- Cite sources when reporting web search findings.
- Write files with descriptive names (e.g. "topic_research.txt", "final_summary.md").
- Keep intermediate files focused on one topic each.
"""


# ---------------------------------------------------------------------------
# Tool binding
# ---------------------------------------------------------------------------

ALL_TOOLS = [
    write_todos,
    tavily_search,
    ls,
    read_file,
    write_file,
    edit_file,
]


def get_agent_runnable() -> BaseChatModel:
    """
    Return the LLM with all tools bound.

    parallel_tool_calls=False enforces one tool per step at the API level,
    preventing the model from batching all actions into a single response
    which causes loss of intermediate reasoning and state corruption.

    Returns
    -------
    BaseChatModel
        A ChatGroq instance augmented with tool schemas.
    """
    llm = get_llm()
    return llm.bind_tools(ALL_TOOLS, parallel_tool_calls=False)


# ---------------------------------------------------------------------------
# Node function
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Simple request detector — avoids burning tokens on greetings/chitchat
# ---------------------------------------------------------------------------

SIMPLE_PATTERNS = {
    "hello", "hi", "hey", "hiya", "howdy",
    "bye", "goodbye", "exit", "quit",
    "thanks", "thank you", "cheers", "ok", "okay", "cool",
    "yes", "no", "sure", "nope", "yep",
}

def _is_simple_message(text: str) -> bool:
    """Return True if the message is a short greeting or chitchat."""
    cleaned = text.strip().lower().rstrip("!.,?")
    # Single word match
    if cleaned in SIMPLE_PATTERNS:
        return True
    # Very short messages (≤4 words) with no research keywords
    words = cleaned.split()
    research_keywords = {
        "report", "research", "find", "search", "explain", "compare",
        "analyze", "analyse", "summarize", "summarise", "write", "list",
        "what", "why", "how", "when", "where", "who", "tell", "give",
    }
    if len(words) <= 4 and not any(w in research_keywords for w in words):
        return True
    return False


SIMPLE_SYSTEM_PROMPT = """You are a helpful AI assistant. 
Respond naturally and conversationally. 
Do NOT use any tools for simple greetings or short messages.
Keep your response brief and friendly."""


def supervisor_node(state: AgentState) -> dict:
    """
    LangGraph node: run the supervisor agent for one reasoning step.

    For simple greetings and chitchat, responds directly without tools
    to avoid wasting API tokens. For research requests, uses the full
    ReAct system prompt with all tools available.

    Parameters
    ----------
    state : AgentState
        Current graph state.

    Returns
    -------
    dict
        Partial state update containing the new ``messages`` entry.
    """
    # Check if this is the first human message and it's simple chitchat
    human_messages = [m for m in state["messages"] if hasattr(m, "type") and m.type == "human"]
    if human_messages:
        last_human = human_messages[-1].content
        if _is_simple_message(str(last_human)):
            # Use a lightweight call with no tools
            from langchain_core.messages import HumanMessage as HM
            llm = get_llm()
            response = llm.invoke(
                [SystemMessage(content=SIMPLE_SYSTEM_PROMPT)]
                + [HM(content=str(last_human))]
            )
            return {"messages": [response]}

    # Full ReAct agent for research/complex tasks
    agent = get_agent_runnable()
    messages = [SystemMessage(content=SYSTEM_PROMPT)] + list(state["messages"])
    response = agent.invoke(messages)

    return {"messages": [response]}
