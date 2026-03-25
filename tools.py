import json
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.tools import tool, StructuredTool

from sub_agents import sub_agents

llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.1)

# ══════════════════════════════════════════════
# FROM MILESTONE 1 — Planning Tool
# ══════════════════════════════════════════════

planning_prompt = PromptTemplate(
    input_variables=["task"],
    template="""You are a precise planning assistant. Break the following task into exactly 4 clear, actionable TODO steps.
Each step must start with one of these verbs: Research, Analyze, Summarize, Save.

Task: {task}

Return ONLY a JSON array of exactly 4 strings. No markdown, no explanation.
Example: ["Research X", "Analyze the findings", "Summarize results", "Save final report"]"""
)

def write_todos_logic(task: str) -> str:
    try:
        content = llm.invoke(planning_prompt.format(task=task)).content.strip()
        if content.startswith("```"):
            content = content.split("\n", 1)[1]
            if content.endswith("```"):
                content = content.rsplit("\n", 1)[0]
            content = content.strip()
        steps = json.loads(content)
        if not isinstance(steps, list):
            raise ValueError("Output must be a list")
        todos = [{"task": step, "status": "pending"} for step in steps]
        return json.dumps({"todos": todos, "count": len(todos)}, indent=2)
    except Exception as e:
        return json.dumps({"error": f"Failed to generate plan: {str(e)}"})

write_todos = StructuredTool.from_function(
    func=write_todos_logic,
    name="write_todos",
    description="Generate a structured task breakdown. Call this FIRST before anything else."
)


# ══════════════════════════════════════════════
# FROM MILESTONE 2 — Virtual File System
# ══════════════════════════════════════════════

class VirtualFileSystem:

    def __init__(self):
        self.fs = {"root": {}}

    def write_file(self, path: str, content: str) -> str:
        parts = path.split("/")
        current = self.fs["root"]
        for p in parts[:-1]:
            if p not in current:
                current[p] = {}
            current = current[p]
        current[parts[-1]] = content
        return f"File '{path}' written successfully."

    def read_file(self, path: str) -> str:
        parts = path.split("/")
        current = self.fs["root"]
        for p in parts:
            if p not in current:
                return f"Error: '{path}' not found."
            current = current[p]
        if isinstance(current, dict):
            return "Error: Path is a directory."
        return current

    def edit_file(self, path: str, new_content: str) -> str:
        parts = path.split("/")
        current = self.fs["root"]
        for p in parts[:-1]:
            if p not in current:
                return f"Error: '{path}' not found."
            current = current[p]
        if parts[-1] not in current:
            return f"Error: '{path}' not found."
        current[parts[-1]] = new_content
        return f"File '{path}' updated successfully."

    def list_files(self) -> dict:
        return self.fs


VFS = VirtualFileSystem()


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
    """List all files in the virtual file system."""
    return str(VFS.list_files())


# ══════════════════════════════════════════════
# MILESTONE 3 NEW — Task Delegation Tool
# ══════════════════════════════════════════════

def task_logic(agent_name: str, input_data: str) -> str:
    """
    Delegate a sub-task to a specialized sub-agent.

    Available agents:
      - research_agent      : pass a short topic (5 words max)
      - summarization_agent : pass the actual text content to summarize

    input_data: one short sentence only, max 15 words, no special characters.
    """
    agent = sub_agents.get(agent_name)

    if not agent:
        return json.dumps({
            "status": "error",
            "result": f"Agent '{agent_name}' not found. Available: {list(sub_agents.keys())}"
        })

    # Fix 1: truncate runaway input_data
    if len(input_data) > 200:
        input_data = input_data[:200]

    # Fix 2: if input_data is a filename, auto-read from VFS
    if input_data.strip().endswith(".txt"):
        file_content = VFS.read_file(input_data.strip())
        if not file_content.startswith("Error:"):
            input_data = file_content

    try:
        result = agent.invoke(input_data)
        return json.dumps({
            "status": "success",
            "agent":  agent_name,
            "result": result
        })
    except Exception as e:
        return json.dumps({
            "status": "error",
            "agent":  agent_name,
            "result": f"Agent error: {str(e)}"
        })


task = StructuredTool.from_function(
    func=task_logic,
    name="task",
    description=(
        "Delegate a sub-task to a specialist sub-agent. "
        "agent_name: research_agent OR summarization_agent. "
        "input_data: one short sentence only, max 15 words, no special characters."
    )
)