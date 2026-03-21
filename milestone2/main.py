from state import state
from tools.task import task

text = """
Artificial Intelligence in healthcare helps in diagnosis,
robotic surgery, and personalized medicine.
"""

# Step 1: Supervisor reasoning
state["trace"].append("Supervisor: Identified summarization task")

# Step 2: Delegation
state["trace"].append("Supervisor: Delegating to summarizer agent")
summary = task("summarizer", text)

# Step 3: Integration of result
state["memory"]["summary.txt"] = summary
state["trace"].append("Supervisor: Stored summary in memory")

# Step 4: Output
print("\nSUMMARY:\n")
print(summary)

print("\nTRACE:\n")
for step in state["trace"]:
    print(step)