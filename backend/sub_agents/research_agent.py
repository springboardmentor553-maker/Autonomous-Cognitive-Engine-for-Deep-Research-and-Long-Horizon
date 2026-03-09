from backend.tools.file_system_tools import write_file


def research_agent(task_id, task_name):

    content = f"""
Research Analysis for: {task_name}

Artificial intelligence is widely used in healthcare to improve
diagnosis, treatment, patient engagement, and predictive analytics.
This research explores the key application and its impact.
"""

    path = f"research/task_{task_id}.txt"

    write_file(path, content)

    return path