from agents.summarizer import summarizer

def task(agent_name, task_text):

    if agent_name == "summarizer":
        return summarizer(task_text)

    return "No agent found"