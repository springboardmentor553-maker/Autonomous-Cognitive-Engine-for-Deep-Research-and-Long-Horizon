import os
import warnings
from typing import Dict, List

warnings.filterwarnings("ignore", category=DeprecationWarning)

from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.tools import tool
from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.prebuilt import create_react_agent

# --------------------------------------------------
# 1. Load Environment Variables
# --------------------------------------------------

load_dotenv()

os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY", "")
os.environ["LANGCHAIN_PROJECT"] = "milestone_2_vfs_validation"

# --------------------------------------------------
# 2. Initialize LLM
# --------------------------------------------------

llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0
)

# --------------------------------------------------
# 3. Virtual File System
# --------------------------------------------------

class VirtualFileSystem:

    def __init__(self):
        self.fs = {"root": {}}

    def write_file(self, path: str, content: str):
        parts = path.split("/")
        current = self.fs["root"]

        for p in parts[:-1]:
            if p not in current:
                current[p] = {}
            current = current[p]

        current[parts[-1]] = content
        return f"File '{path}' written successfully."

    def read_file(self, path: str):
        parts = path.split("/")
        current = self.fs["root"]

        for p in parts:
            if p not in current:
                return f"Error: {path} not found."
            current = current[p]

        if isinstance(current, dict):
            return "Error: Path is a directory."

        return current

    def edit_file(self, path: str, new_content: str):
        parts = path.split("/")
        current = self.fs["root"]

        for p in parts[:-1]:
            if p not in current:
                return f"Error: {path} not found."
            current = current[p]

        if parts[-1] not in current:
            return f"Error: {path} not found."

        current[parts[-1]] = new_content
        return f"File '{path}' updated successfully."

    def list_files(self):
        # Updated to return the full filesystem state to match desired output
        return self.fs
    
 
VFS = VirtualFileSystem()

# --------------------------------------------------
# 4. Tools for Agent
# --------------------------------------------------

@tool
def write_file(filename: str, content: str) -> str:
    """Store intermediate results into a virtual file."""
    return VFS.write_file(filename, content)


@tool
def read_file(filename: str) -> str:
    """Retrieve stored content from a virtual file."""
    return VFS.read_file(filename)


@tool
def edit_file(filename: str, new_content: str) -> str:
    """Modify an existing virtual file."""
    return VFS.edit_file(filename, new_content)


@tool
def ls() -> str:
    """List available files in the virtual file system."""
    return str(VFS.list_files())


# --------------------------------------------------
# 5. System Prompt (CRITICAL UPDATE)
# --------------------------------------------------


system_prompt = """
You are a precise architecture agent. You operate under strict constraints.

**CRITICAL CONSTRAINT**: You have NO memory of previous steps or the user prompt. 
1. Even if you just wrote a file, or the user prompt contains data, you cannot remember the content. 
2. You must use `read_file` to retrieve content before you can use it in any comparison or editing step.
3. Do not assume you know what is in a file. Always read it first.

Workflow Rules:
1. Save results into files immediately using write_file.
2. When comparing information, read the relevant files first (you cannot use the prompt's input directly).
3. When refining data, read the file before editing it.
4. Use ls to see available stored files if needed.
"""

# --------------------------------------------------
# 6. Create ReAct Agent
# --------------------------------------------------

agent = create_react_agent(
    llm,
    tools=[write_file, read_file, edit_file, ls]
)

# --------------------------------------------------
# 7. Architecture Validation
# --------------------------------------------------

def validate_architecture(trace: List[Dict]):

    print("\n--- ARCHITECTURE VALIDATION REPORT ---")

    sequence = []

    for call in trace:
        name = call['name']
        args = call['args']
        fname = args.get('filename', '')
        sequence.append((name, fname))

    # Dependency Chain Check

    comparison_idx = next(
        (i for i, (name, fname) in enumerate(sequence)
         if "comparison" in str(fname)),
        None
    )

    if comparison_idx:

        steps_before = sequence[:comparison_idx]

        reads_before = [
            s for s in steps_before
            if s[0] == 'read_file'
        ]

        if len(reads_before) >= 2:
            print("✅ DEPENDENCY CHAIN: Agent read inputs before comparison.")
        else:
            print("❌ DEPENDENCY CHAIN: Comparison created without reading inputs.")
    
    # Refinement Logic Check

    edit_calls = [
        i for i, (name, _) in enumerate(sequence)
        if name == 'edit_file'
    ]

    if edit_calls:

        last_edit = edit_calls[-1]

        pre_edit_steps = sequence[max(0, last_edit - 3):last_edit]

        read_before_edit = any(
            s[0] == 'read_file'
            for s in pre_edit_steps
        )

        if read_before_edit:
            print("✅ REFINEMENT LOGIC: Agent read file before editing.")
        else:
            print("❌ REFINEMENT LOGIC: Agent edited without reading.")

    else:
        print("⚠️ No edit_file call detected.")

    # Selective Retrieval

    total_reads = len([s for s in sequence if s[0] == "read_file"])
    total_writes = len([s for s in sequence if s[0] == "write_file"])

    print(f"📊 Total Reads: {total_reads}")
    print(f"📊 Total Writes: {total_writes}")

    print("--------------------------------------")


# --------------------------------------------------
# 8. Execution Function
# --------------------------------------------------

def run_final_validation():

    global VFS
    VFS = VirtualFileSystem()

    task = """
    Process the following inputs:

    Input A: "Project Alpha focuses on speed."
    Input B: "Project Beta focuses on stability."

    Steps:

    1. Write 'A.txt' with Input A summary.
    2. Write 'B.txt' with Input B summary.
    3. Compare A and B.
       Read both files and write 'comparison.txt'.
    4. Refine the comparison.
       Read 'comparison.txt' and edit it to add a conclusion.
    """

    print("🚀 Running Final Architectural Test...")

    result = agent.invoke({
        "messages": [
            SystemMessage(content=system_prompt),
            HumanMessage(content=task)
        ]
    })

    # Extract tool call trace

    full_trace = []

    for msg in result["messages"]:
        if hasattr(msg, "tool_calls") and msg.tool_calls:
            for call in msg.tool_calls:
                full_trace.append(call)

    # Validate architecture

    validate_architecture(full_trace)

    # Print Virtual File System State

    print("\n📁 Virtual File System State:")
    print(VFS.list_files())


# --------------------------------------------------
# 9. Main
# --------------------------------------------------

if __name__ == "__main__":
    run_final_validation()

