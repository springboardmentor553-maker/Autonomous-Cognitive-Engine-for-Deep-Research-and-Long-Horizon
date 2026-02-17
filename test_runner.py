from main import run_agent

test_tasks = [
    "Create a market analysis for electric vehicles",
    "Prepare a technical blog on LangGraph",
    "Generate a research summary on AI fairness"
]

for task in test_tasks:
    print("=" * 50)
    print("Testing Task:", task)
    result = run_agent(task)
    print("TODOs Generated:", result["todos_created"])