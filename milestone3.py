"""
Milestone 3: Multi-Agent Collaboration Test — Ollama (llama3)
"""
import os
import sys
from langchain_core.messages import HumanMessage
from workflow.multi_agent_flow import create_multi_agent_workflow, get_tool_call_stats, reset_tool_call_stats
from brains.filetools import clear_virtual_fs, FILE_SYSTEM_DIR, get_fs_stats
from brains.supervisor import get_delegation_stats, reset_delegation_stats

os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"]    = "milestone3"

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL    = os.getenv("OLLAMA_MODEL",    "llama3")

print(f"✓ Ollama URL:  {OLLAMA_BASE_URL}")
print(f"✓ Model:       {OLLAMA_MODEL}")
print(f"✓ LangSmith tracing ENABLED — project: milestone3")
print(f"✓ Storage: {FILE_SYSTEM_DIR.absolute()}\n")


def run_milestone3():
    print("=" * 100)
    print("MILESTONE 3: MULTI-AGENT COLLABORATION")
    print("=" * 100)

    reset_delegation_stats()
    reset_tool_call_stats()

    FILE_SYSTEM_DIR.mkdir(exist_ok=True)
    print("Clearing previous files...", end=" ")
    clear_virtual_fs()
    print("✓\n")

    print("Initializing multi-agent workflow...", end=" ")
    workflow = create_multi_agent_workflow()
    print("✓\n")

    task = """
    Create a comprehensive analysis of renewable energy solutions.

    Focus on:
    1. Solar energy technology and market
    2. Wind energy technology and market
    3. Comparison and future outlook

    Deliver a professional report with research, analysis, and recommendations.
    """

    print("Task: Renewable energy analysis")
    print("Starting workflow...\n" + "=" * 100 + "\n")

    initial_state = {
        "messages": [HumanMessage(content=task)],
        "todos": [
            {"id": 1, "description": "Research solar energy technology and market trends",  "status": "pending"},
            {"id": 2, "description": "Research wind energy technology and market trends",   "status": "pending"},
            {"id": 3, "description": "Research renewable energy future outlook",            "status": "pending"},
            {"id": 4, "description": "Write comprehensive report from all research findings","status": "pending"},
            {"id": 5, "description": "Review and finalize the report for quality",          "status": "pending"},
        ],
        "current_step":     1,
        "completed_steps":  [],
        "active_agent":     "supervisor",
        "created_files":    [],
        "pending_files":    [],
        "researcher_status":"idle",
        "writer_status":    "idle",
        "reviewer_status":  "idle",
        "user_task":        task,
        "final_output":     ""
    }

    try:
        result = workflow.invoke(initial_state, {"recursion_limit": 50})

        print("\n" + "=" * 100)
        print("WORKFLOW COMPLETE")
        print("=" * 100)

        fs_stats = get_fs_stats()
        files    = fs_stats.get("files", [])
        print(f"\n✅ Steps completed:  {result.get('completed_steps', [])}")
        print(f"✅ Files created:    {len(files)}")

        if files:
            print(f"\n📁 Files in {FILE_SYSTEM_DIR}:")
            for fname in files:
                fpath = FILE_SYSTEM_DIR / fname
                if fpath.exists():
                    print(f"  • {fname} ({fpath.stat().st_size} bytes)")

        ds = get_delegation_stats()
        ts = get_tool_call_stats()
        total_delegations = ds['researcher_calls'] + ds['writer_calls'] + ds['reviewer_calls']

        print("\n📊 DELEGATION EVENTS:")
        print(f"  • Researcher: {ds['researcher_calls']}")
        print(f"  • Writer:     {ds['writer_calls']}")
        print(f"  • Reviewer:   {ds['reviewer_calls']}")
        print(f"  • Total:      {total_delegations}")

        print("\n🔧 TOOL USAGE:")
        for k, v in ts.items():
            print(f"  • {k}: {v}")
        print(f"  • Total: {sum(ts.values())}")

        checks = {
            "todos_created":       len(result.get('todos', [])) == 5,
            "all_steps_completed": len(result.get('completed_steps', [])) >= 5,
            "delegation_occurred": total_delegations > 0,
            "researcher_used":     ds['researcher_calls'] > 0,
            "writer_used":         ds['writer_calls'] > 0,
            "reviewer_used":       ds['reviewer_calls'] > 0,
            "files_created":       len(files) > 0,
            "tools_used":          sum(ts.values()) > 0,
        }

        print("\n" + "=" * 100)
        print("MILESTONE 3 VALIDATION")
        print("=" * 100)
        for name, passed in checks.items():
            print(f"  {'✓' if passed else '✗'} {name.replace('_',' ').title()}")

        all_passed = all(checks.values())
        print("\n" + "=" * 100)
        print("🎉 MILESTONE 3: PASSED ✓" if all_passed else "⚠️  MILESTONE 3: PARTIAL PASS")
        print("=" * 100)
        return all_passed

    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback; traceback.print_exc()
        return False


if __name__ == "__main__":
    success = run_milestone3()
    print(f"\nResult: {'✅ PASSED' if success else '❌ FAILED'}")
    sys.exit(0 if success else 1)
