from utils.llm import llm

def write_todos(task_description):

    prompt = f"""
    Break this into SHORT, CLEAR tasks.

    Rules:
    - Max 6 tasks
    - Each task = ONE simple action
    - No numbering
    - No explanations
    - No headings

    Assign type:
    - summarize
    - research
    - general

    STRICT FORMAT (VERY IMPORTANT):
    task | type

    Example:
    Collect healthcare data | research
    Summarize AI benefits | summarize

    Task:
    {task_description}
    """

    response = llm.invoke(prompt)
    lines = response.content.strip().split("\n")

    todos = []

    for line in lines:
        line = line.strip()

        # skip empty lines
        if not line:
            continue

        # ensure correct format
        if "|" in line:
            parts = line.split("|")

            # safety check
            if len(parts) == 2:
                task_text = parts[0].strip()
                task_type = parts[1].strip().lower()

                # normalize type
                if task_type not in ["summarize", "research", "general"]:
                    task_type = "general"

                todos.append({
                    "task": task_text,
                    "type": task_type,
                    "status": "pending"
                })

    return todos