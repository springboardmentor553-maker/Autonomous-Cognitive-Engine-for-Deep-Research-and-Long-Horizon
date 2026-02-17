def delegate_task(task: str):
    if "code" in task:
        return "code_agent"
    if "research" in task:
        return "research_agent"
    return "summarizer_agent"