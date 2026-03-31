"""
Shared Delegation Layer Architecture (Milestone 3 / 4)
This file introduces a central Registry Pattern.
Any developer can create an agent elsewhere, import `agent_registry`, and register their agent dynamically.
"""

import sys
sys.stdout.reconfigure(encoding='utf-8')  # type: ignore
from typing import Callable, Dict, Any

class SubAgentRegistry:
    def __init__(self):
        # Maps an agent identifier (e.g., 'research_agent') to a tuple of (description, function/runnable)
        self._agents: Dict[str, dict] = {}

    def register_agent(self, name: str, description: str, execute_func: Callable):
        """
        Allows any developer across the codebase to register their custom agent.
        """
        print(f"[REGISTRY] \u2795 Registered new sub-agent: '{name}'")
        self._agents[name] = {
            "description": description,
            "execute": execute_func
        }

    def get_agent_info(self):
        """Returns the list of available agents and their descriptions for the Supervisor LLM."""
        return [{"name": name, "description": data["description"]} for name, data in self._agents.items()]

    def execute_agent(self, name: str, task_state: Any):
        """Dynamically invokes the registered agent if it exists."""
        if name not in self._agents:
            raise ValueError(f"Determinism Error: Requested agent '{name}' is not in the registry!")
        
        print(f"[DELEGATOR] \u2192 Routing task dynamically to '{name}'")
        return self._agents[name]["execute"](task_state)

# The shared Global Registry instance
agent_registry = SubAgentRegistry()


# =========================================================================================
# EXAMPLE USAGE MOCKUP: How someone else's external agents securely hook into the system
# =========================================================================================
if __name__ == "__main__":
    print("======================================================")
    print("SHARED DELEGATION LAYER & REGISTRY DEMONSTRATION")
    print("======================================================\n")
    
    # 1. Someone else creates a complex agent entirely separately
    def alexs_code_agent(state):
        print("   \u21b3 [ALEX'S CODE AGENT] \u27a4 Writing Python code... Done.")
        return {"data": "print('Hello World')"}

    def sallys_research_agent(state):
        print("   \u21b3 [SALLY'S RESEARCH AGENT] \u27a4 Searching Tavily... Found.")
        return {"data": "Artificial Intelligence statistics found."}

    # 2. They register their agents into our central layer during app startup
    agent_registry.register_agent(
        name="coding",
        description="Generates compiled python code. Use for any programming tasks.",
        execute_func=alexs_code_agent
    )
    agent_registry.register_agent(
        name="research",
        description="Scrapes the internet using Tavily. Use for gathering facts.",
        execute_func=sallys_research_agent
    )
    
    print("\n[SUPERVISOR VIEW] The Supervisor LLM now automatically sees:")
    for info in agent_registry.get_agent_info():
        print(f"  - {info['name']}: {info['description']}")

    print("\n--- INITIATING DETERMINISTIC DELEGATION ---\n")
    
    # 3. How do we deterministically decide which sub-agent is right?
    # -> During the 'Write Todos' phase, the Supervisor LLM uses Structured Output (Pydantic / Enums).
    # Instead of just outputting "1. Find statistics", it outputs:
    # {"task_id": 1, "desc": "Find statistics", "designated_agent": "research"}
    
    # Here is what a perfectly deterministic AI-generated task list looks like from the Supervisor:
    deterministic_tasks = [
        {"task_id": 1, "desc": "Write a hello world script", "designated_agent": "coding"},
        {"task_id": 2, "desc": "Find recent AI stats", "designated_agent": "research"},
    ]
    
    # 4. The Shared Delegation Layer loop simply blindly trusts the designated agent constraint!
    for task in deterministic_tasks:
        print(f"Task {task['task_id']}: '{task['desc']}'")
        agent_to_call = task["designated_agent"]
        
        # Look up dynamically without any hardcoded "if/else" logic!
        try:
            result = agent_registry.execute_agent(agent_to_call, task)
            print(f"   [DELEGATION SUCCESS] Result -> {result.get('data')}\n")
        except ValueError as e:
            print(f"   [DELEGATION FAILED] {e}")
