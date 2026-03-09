from backend.sub_agents.research_agent import research_agent


def execute_plan(state):

    trace = []

    for i, task in enumerate(state["todos"], start=1):

        trace.append(f"THINK → Analyzing Task {i}")

        filepath = research_agent(i, task)

        trace.append(f"WRITE FILE → memory/{filepath}")
        trace.append(f"OBSERVE → Stored research in memory/{filepath}")

    state["trace"] = trace

    return state
