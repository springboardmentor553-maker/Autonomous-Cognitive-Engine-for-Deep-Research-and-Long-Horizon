"""
Supervisor Agent - Orchestrates multi-agent workflow with delegation tracking
"""
from langchain_groq import ChatGroq
from langchain_core.tools import tool
from brains.mainagent import write_todos
from brains.filetools import read_file, write_file, get_fs_stats
import json


# Global tracking for delegation statistics
DELEGATION_STATS = {
    "researcher_calls": 0,
    "writer_calls": 0,
    "reviewer_calls": 0,
    "supervisor_direct_actions": 0
}


@tool
def delegate_to_researcher(task: str) -> str:
    """
    Delegate research task to Researcher agent.
    
    Args:
        task: Description of research needed
    
    Returns:
        Status message with delegation info
    """
    DELEGATION_STATS["researcher_calls"] += 1
    
    total_delegations = sum([
        DELEGATION_STATS['researcher_calls'],
        DELEGATION_STATS['writer_calls'],
        DELEGATION_STATS['reviewer_calls']
    ])
    
    print(f"\n{'='*80}")
    print(f"🔄 DELEGATION EVENT #{total_delegations}")
    print(f"{'='*80}")
    print(f"FROM: Supervisor")
    print(f"TO: Researcher Agent")
    print(f"TASK: {task}")
    print(f"REASON: Requires web search and data gathering expertise")
    print(f"DELEGATION TYPE: Research task requiring external information")
    print(f"{'='*80}\n")
    
    return json.dumps({
        "status": "delegated",
        "agent": "researcher",
        "task": task,
        "delegation_number": DELEGATION_STATS["researcher_calls"]
    })


@tool
def delegate_to_writer(task: str) -> str:
    """
    Delegate writing task to Writer agent.
    
    Args:
        task: Description of writing needed
    
    Returns:
        Status message with delegation info
    """
    DELEGATION_STATS["writer_calls"] += 1
    
    total_delegations = sum([
        DELEGATION_STATS['researcher_calls'],
        DELEGATION_STATS['writer_calls'],
        DELEGATION_STATS['reviewer_calls']
    ])
    
    print(f"\n{'='*80}")
    print(f"🔄 DELEGATION EVENT #{total_delegations}")
    print(f"{'='*80}")
    print(f"FROM: Supervisor")
    print(f"TO: Writer Agent")
    print(f"TASK: {task}")
    print(f"REASON: Requires professional content creation and document formatting")
    print(f"DELEGATION TYPE: Document creation task")
    print(f"{'='*80}\n")
    
    return json.dumps({
        "status": "delegated",
        "agent": "writer",
        "task": task,
        "delegation_number": DELEGATION_STATS["writer_calls"]
    })


@tool
def delegate_to_reviewer(task: str) -> str:
    """
    Delegate review task to Reviewer agent.
    
    Args:
        task: Description of review needed
    
    Returns:
        Status message with delegation info
    """
    DELEGATION_STATS["reviewer_calls"] += 1
    
    total_delegations = sum([
        DELEGATION_STATS['researcher_calls'],
        DELEGATION_STATS['writer_calls'],
        DELEGATION_STATS['reviewer_calls']
    ])
    
    print(f"\n{'='*80}")
    print(f"🔄 DELEGATION EVENT #{total_delegations}")
    print(f"{'='*80}")
    print(f"FROM: Supervisor")
    print(f"TO: Reviewer Agent")
    print(f"TASK: {task}")
    print(f"REASON: Requires quality assurance and final review")
    print(f"DELEGATION TYPE: Quality control task")
    print(f"{'='*80}\n")
    
    return json.dumps({
        "status": "delegated",
        "agent": "reviewer",
        "task": task,
        "delegation_number": DELEGATION_STATS["reviewer_calls"]
    })


@tool
def list_files_tool() -> str:
    """
    List all files in the virtual file system.
    Returns JSON with file information.
    """
    stats = get_fs_stats()
    return json.dumps(stats, indent=2)


@tool
def simple_text_operation(operation: str, text: str) -> str:
    """
    Perform simple text operations directly (no delegation needed).
    
    Args:
        operation: Type of operation (uppercase, lowercase, reverse, count_words)
        text: Text to operate on
    
    Returns:
        Result of operation
    """
    DELEGATION_STATS["supervisor_direct_actions"] += 1
    
    print(f"\n{'='*80}")
    print(f"✅ DIRECT ACTION BY SUPERVISOR (No Delegation Needed)")
    print(f"{'='*80}")
    print(f"OPERATION: {operation}")
    print(f"INPUT: {text[:50]}...")
    print(f"REASON: Simple operation, no specialized agent required")
    print(f"ACTION TYPE: Direct supervisor execution")
    print(f"{'='*80}\n")
    
    if operation == "uppercase":
        result = text.upper()
    elif operation == "lowercase":
        result = text.lower()
    elif operation == "reverse":
        result = text[::-1]
    elif operation == "count_words":
        result = f"Word count: {len(text.split())}"
    else:
        result = f"Unknown operation: {operation}"
    
    return json.dumps({
        "operation": operation,
        "result": result,
        "handled_by": "supervisor_directly"
    })


def create_supervisor_agent():
    """Create supervisor agent with delegation and coordination tools."""
    
    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=0.0
    )
    
    # Supervisor's tools - NO web_search (delegates to researcher instead)
    tools = [
        write_todos,
        delegate_to_researcher,
        delegate_to_writer,
        delegate_to_reviewer,
        simple_text_operation,
        read_file,
        write_file,
        list_files_tool
    ]
    
    llm_with_tools = llm.bind_tools(tools)
    
    system_prompt = """You are a Supervisor Agent managing a team of specialist agents.

YOUR TEAM:
- RESEARCHER: Web search specialist, gathers external information, stores in files
- WRITER: Content creation specialist, creates professional documents from research
- REVIEWER: Quality assurance specialist, ensures accuracy and completeness

DELEGATION RULES (FOLLOW STRICTLY):

✓ DELEGATE TO RESEARCHER when task requires:
  - Web search or external information gathering
  - Market research, technology research, data collection
  - Any information not in your knowledge base

✓ DELEGATE TO WRITER when task requires:
  - Creating documents, reports, or structured content
  - Professional formatting and organization
  - Compiling multiple sources into cohesive output

✓ DELEGATE TO REVIEWER when task requires:
  - Quality checking and verification
  - Accuracy validation against sources
  - Final review before delivery

✓ HANDLE DIRECTLY (use simple_text_operation) when task is:
  - Simple text transformation (uppercase, lowercase, reverse)
  - Basic calculations or word counts
  - File listing (use list_files_tool)

COORDINATION WORKFLOW:
1. Analyze the pre-existing plan (5 TODO steps)
2. For each step, determine: delegate or handle directly?
3. Use delegation tools to assign work to appropriate agents
4. Monitor file system to track progress
5. Continue until all steps completed

CRITICAL NOTES:
- Plan already exists (5 steps) - DO NOT create new plan
- Each TODO step requires ONE delegation (researcher/writer/reviewer)
- Agents will use their own tools (web_search, write_file, read_file)
- You coordinate; agents execute
- Check files to verify agents completed their work

EXAMPLE COORDINATION:
Step 1: "Research solar energy" → delegate_to_researcher(task="Research solar energy technology and market trends")
Step 4: "Write report" → delegate_to_writer(task="Write comprehensive report from all research findings")
Step 5: "Review report" → delegate_to_reviewer(task="Review and finalize the report for quality")

You are the orchestrator - delegate wisely and coordinate effectively!"""
    
    return llm_with_tools, system_prompt


def get_delegation_stats():
    """Get current delegation statistics."""
    return DELEGATION_STATS.copy()


def reset_delegation_stats():
    """Reset delegation statistics."""
    global DELEGATION_STATS
    DELEGATION_STATS = {
        "researcher_calls": 0,
        "writer_calls": 0,
        "reviewer_calls": 0,
        "supervisor_direct_actions": 0
    }
