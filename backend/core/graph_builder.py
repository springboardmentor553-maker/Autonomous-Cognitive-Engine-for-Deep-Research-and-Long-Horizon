from backend.core.state import AgentState
from backend.tools.planning_tool import plan_tasks
from backend.tools.delegation_tool import delegate_task
from backend.tools.search_tool import web_search
from backend.sub_agents.summarizer_agent import summarize
from backend.sub_agents.research_agent import research
from backend.sub_agents.code_agent import generate_code
def run_agent(goal: str):
    state = AgentState(goal=goal)
    state.todos = plan_tasks(goal)
    while state.todos:
        task = state.todos.pop(0)
        if "research" in task:
            result = research(task)
        elif "code" in task:
            result = generate_code(task)
        elif "summary" in task:
            result = summarize(task)
        else:
            result = web_search(task)

        state.completed.append(task)
        state.memory.append(result)
    return state