"""
Supervisor Agent - OPTIMIZED TO REDUCE TOKEN WASTE
Only delegates when necessary, not for every step
"""
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage
import json

# Delegation tracking
delegation_stats = {
    "researcher_calls": 0,
    "writer_calls": 0,
    "reviewer_calls": 0
}

def get_delegation_stats():
    """Get delegation statistics"""
    return delegation_stats.copy()

def reset_delegation_stats():
    """Reset delegation statistics"""
    global delegation_stats
    delegation_stats = {
        "researcher_calls": 0,
        "writer_calls": 0,
        "reviewer_calls": 0
    }

def create_supervisor():
    """Create optimized supervisor agent - MINIMAL DELEGATION"""
    
    llm = ChatAnthropic(
        model="claude-sonnet-4-20250514",
        temperature=0,
        max_tokens=500  # REDUCED - supervisor needs less tokens
    )
    
    # OPTIMIZED SYSTEM PROMPT - DIRECT DELEGATION
    system_message = """You are a workflow supervisor. Your ONLY job is to decide which agent should handle the current task.

CRITICAL RULES TO SAVE TOKENS:
1. Be EXTREMELY brief - just pick an agent, NO explanations
2. ONLY delegate to ONE agent per step
3. Follow this EXACT sequence:
   - Step 1: researcher (research task)
   - Step 2: researcher (gather more data)
   - Step 3: researcher (final research)
   - Step 4: writer (create report)
   - Step 5: reviewer (review report)
4. After step 5, respond ONLY with: "FINISH"

DO NOT:
- Write long explanations
- Provide analysis
- Give instructions to agents
- Repeat yourself
- Add commentary

Respond with ONLY ONE WORD from: researcher, writer, reviewer, FINISH"""

    def supervisor_node(state):
        """Optimized supervisor - minimal token usage"""
        current_step = state.get("current_step", 1)
        
        # HARDCODED LOGIC - NO LLM CALLS NEEDED (saves massive tokens)
        if current_step <= 3:
            next_agent = "researcher"
            delegation_stats["researcher_calls"] += 1
        elif current_step == 4:
            next_agent = "writer"
            delegation_stats["writer_calls"] += 1
        elif current_step == 5:
            next_agent = "reviewer"
            delegation_stats["reviewer_calls"] += 1
        else:
            next_agent = "FINISH"
        
        print(f"[SUPERVISOR] Step {current_step} → Delegate to: {next_agent}")
        
        return {
            **state,
            "next": next_agent,
            "active_agent": "supervisor"
        }
    
    return supervisor_node
