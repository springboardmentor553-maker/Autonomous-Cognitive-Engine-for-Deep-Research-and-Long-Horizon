import json
from backend.tools.file_system_tools import write_file


def write_todos(objective):

    if "three" in objective.lower():
        todos = [
            {"task": "Medical Imaging Analysis"},
            {"task": "Clinical Decision Support Systems"},
            {"task": "AI Chatbots in Healthcare"},
            {"task": "Final Summary"}
        ]
    else:
        todos = [
            {"task": "Medical Imaging Analysis"},
            {"task": "Clinical Decision Support Systems"},
            {"task": "AI Chatbots in Healthcare"},
            {"task": "Predictive Analytics"},
            {"task": "Telemedicine Platforms"},
            {"task": "Final Summary"}
        ]

    print("\n[PLANNING] TODO list created\n")

    return todos