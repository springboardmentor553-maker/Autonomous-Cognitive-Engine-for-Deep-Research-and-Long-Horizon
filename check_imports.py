import traceback, sys

steps = [
    ("langchain_groq",       "from langchain_groq import ChatGroq"),
    ("state",                "from state import AgentState, TodoItem, DelegationEntry"),
    ("filesystem_tools",     "from filesystem_tools import get_virtual_fs, set_virtual_fs, FILESYSTEM_TOOLS"),
    ("sub_agents.registry",  "from sub_agents.registry import SUB_AGENT_REGISTRY, run_sub_agent"),
    ("delegation_tool",      "from delegation_tool import DELEGATION_TOOLS"),
    ("tools",                "from tools import ALL_TOOLS, PLANNING_TOOLS"),
    ("streamlit",            "import streamlit"),
]

for name, stmt in steps:
    try:
        exec(stmt, {})
        print(f"PASS: {name}")
    except Exception as e:
        print(f"FAIL: {name}")
        traceback.print_exc(file=sys.stdout)
        sys.exit(1)

print("ALL OK")
