def evaluate_output(state):

    score = {}

    score["tasks_completed"] = len(state["completed_tasks"])

    score["research_articles"] = len(state["research_data"])

    score["summaries_generated"] = len(state["summaries"])

    state["evaluation"] = score

    return state