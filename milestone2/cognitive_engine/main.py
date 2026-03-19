from state import state
from tools.file_tools import write_file, read_file, edit_file, ls
from selective_reader import load_required_files


# STEP 1 — Summarize Framework A
write_file(state, "A_summary.txt", "Framework A focuses on transparency.")

# STEP 2 — Summarize Framework B
write_file(state, "B_summary.txt", "Framework B focuses on fairness.")

# STEP 3 — Summarize Framework C
write_file(state, "C_summary.txt", "Framework C focuses on accountability.")

# STEP 4 — Summarize Framework D
write_file(state, "D_summary.txt", "Framework D focuses on safety.")


# STEP 5 — Compare frameworks (Selective Retrieval)
files_needed = [
    "A_summary.txt",
    "B_summary.txt",
    "C_summary.txt",
    "D_summary.txt"
]

content = load_required_files(state, files_needed)

comparison = f"""
Framework Comparison

{content}

Key Differences:
A → transparency
B → fairness
C → accountability
D → safety
"""

write_file(state, "comparison.txt", comparison)


# STEP 6 — Create unified model
comp_data = read_file(state, "comparison.txt")

unified_model = f"""
Unified AI Ethics Model

{comp_data}

Combined Principles:
Transparency
Fairness
Accountability
Safety
"""

write_file(state, "unified_model.txt", unified_model)


# STEP 7 — Refine unified model (edit operation)
model_data = read_file(state, "unified_model.txt")

refined_model = model_data + "\nAdded sustainability considerations."

edit_file(state, "unified_model.txt", refined_model)


# FINAL OUTPUT

print("\nFILES IN MEMORY:\n")
files = ls(state)
for filename in files:
    print(filename)

print("\nTRACE:\n")
for step in state["trace"]:
    print(step)