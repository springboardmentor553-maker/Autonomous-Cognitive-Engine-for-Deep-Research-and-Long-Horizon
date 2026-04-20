"""
Supervisor Agent - Hardcoded Routing
Hardcoded routing - no LLM calls needed, saves tokens.
"""

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
    """Create supervisor agent - hardcoded routing, no LLM needed"""

    def supervisor_node(state):
        """Supervisor node - routes by step number"""
        current_step = state.get("current_step", 1)

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
