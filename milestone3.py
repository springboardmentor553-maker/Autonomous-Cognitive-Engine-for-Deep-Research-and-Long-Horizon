"""
Milestone 3: Multi-Agent Collaboration Test
Demonstrates Supervisor coordinating Researcher, Writer, and Reviewer
"""
import os
import sys
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from workflow.multi_agent_flow import create_multi_agent_workflow, get_tool_call_stats, reset_tool_call_stats
from brains.filetools import clear_virtual_fs, FILE_SYSTEM_DIR, get_fs_stats
from brains.supervisor import get_delegation_stats, reset_delegation_stats

# Configure tracing
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "milestone3"

load_dotenv()

# Verify API key
if not os.getenv("GROQ_API_KEY"):
    print("❌ ERROR: GROQ_API_KEY not set!")
    sys.exit(1)

print("✓ Groq API key detected")
print("✓ LangSmith tracing ENABLED")
print(f"✓ Project: milestone3")
print(f"✓ Storage: {FILE_SYSTEM_DIR.absolute()}\n")


def run_milestone3():
    """Run multi-agent workflow test."""
    
    print("=" * 100)
    print("MILESTONE 3: MULTI-AGENT COLLABORATION")
    print("=" * 100)
    print("\nDemonstrates:")
    print("  - Supervisor orchestrating workflow")
    print("  - Researcher gathering information")
    print("  - Writer creating content")
    print("  - Reviewer ensuring quality")
    print("\n" + "=" * 100 + "\n")
    
    # Reset statistics
    reset_delegation_stats()
    reset_tool_call_stats()
    
    # Setup
    FILE_SYSTEM_DIR.mkdir(exist_ok=True)
    print("Clearing previous files...", end=" ")
    clear_virtual_fs()
    print("✓\n")
    
    # Create workflow
    print("Initializing multi-agent workflow...", end=" ")
    workflow = create_multi_agent_workflow()
    print("✓\n")
    
    # Define task
    task = """
    Create a comprehensive analysis of renewable energy solutions.
    
    Focus on:
    1. Solar energy technology and market
    2. Wind energy technology and market
    3. Comparison and future outlook
    
    Deliver a professional report with research, analysis, and recommendations.
    """
    
    print("Task: Renewable energy analysis")
    print("Expected workflow:")
    print("  1. Supervisor creates 5-step plan")
    print("  2. Researcher gathers solar energy data")
    print("  3. Researcher gathers wind energy data")
    print("  4. Writer creates comprehensive report")
    print("  5. Reviewer finalizes and polishes")
    print("\nStarting workflow (this may take 2-3 minutes)...\n")
    print("=" * 100 + "\n")
    
    # Initial state with pre-made plan
    initial_state = {
        "messages": [HumanMessage(content=task)],
        "todos": [
            {"id": 1, "description": "Research solar energy technology and market trends", "status": "pending"},
            {"id": 2, "description": "Research wind energy technology and market trends", "status": "pending"},
            {"id": 3, "description": "Research renewable energy future outlook", "status": "pending"},
            {"id": 4, "description": "Write comprehensive report from all research findings", "status": "pending"},
            {"id": 5, "description": "Review and finalize the report for quality", "status": "pending"}
        ],
        "current_step": 1,
        "completed_steps": [],
        "active_agent": "supervisor",
        "created_files": [],
        "pending_files": [],
        "researcher_status": "idle",
        "writer_status": "idle",
        "reviewer_status": "idle",
        "user_task": task,
        "final_output": ""
    }
    
    try:
        # Execute workflow
        result = workflow.invoke(
            initial_state,
            {"recursion_limit": 50}
        )
        
        print("\n" + "=" * 100)
        print("WORKFLOW COMPLETE")
        print("=" * 100)
        
        # Display results
        print(f"\n✅ Steps completed: {result.get('completed_steps', [])}")
        print(f"✅ Final step reached: {result.get('current_step', 0)}")
        
        # Get final file stats
        fs_stats = get_fs_stats()
        files = fs_stats.get("files", [])
        
        print(f"✅ Files created: {len(files)}")
        
        # Show files
        if files:
            print(f"\n📁 Files in {FILE_SYSTEM_DIR}:")
            for fname in files:
                fpath = FILE_SYSTEM_DIR / fname
                if fpath.exists():
                    size = fpath.stat().st_size
                    print(f"  • {fname} ({size} bytes)")
        else:
            print(f"\n⚠️  No files found in {FILE_SYSTEM_DIR}")
        
        # Get statistics
        delegation_stats = get_delegation_stats()
        tool_stats = get_tool_call_stats()
        
        print("\n" + "=" * 100)
        print("DELEGATION & TOOL USAGE STATISTICS")
        print("=" * 100)
        
        print("\n📊 DELEGATION EVENTS:")
        print(f"  • Researcher delegations: {delegation_stats['researcher_calls']}")
        print(f"  • Writer delegations: {delegation_stats['writer_calls']}")
        print(f"  • Reviewer delegations: {delegation_stats['reviewer_calls']}")
        print(f"  • Supervisor direct actions: {delegation_stats['supervisor_direct_actions']}")
        total_delegations = delegation_stats['researcher_calls'] + delegation_stats['writer_calls'] + delegation_stats['reviewer_calls']
        print(f"  • Total delegations: {total_delegations}")
        
        print("\n🔧 TOOL USAGE:")
        print(f"  • web_search calls: {tool_stats.get('web_search', 0)}")
        print(f"  • write_file calls: {tool_stats.get('write_file', 0)}")
        print(f"  • read_file calls: {tool_stats.get('read_file', 0)}")
        print(f"  • write_todos calls: {tool_stats.get('write_todos', 0)}")
        print(f"  • edit_file calls: {tool_stats.get('edit_file', 0)}")
        
        total_operations = sum(tool_stats.values())
        print(f"  • Total tool operations: {total_operations}")
        
        print("\n💡 WORKFLOW EFFICIENCY:")
        todos_count = len(result.get('todos', []))
        if todos_count > 0:
            delegation_ratio = total_delegations / todos_count * 100
            print(f"  • Delegation ratio: {delegation_ratio:.0f}% (delegations vs total steps)")
        print(f"  • Supervisor involvement: {delegation_stats['supervisor_direct_actions']} direct actions")
        print(f"  • Multi-agent collaboration: {'✓ Active' if total_delegations > 0 else '✗ No delegation'}")
        
        # Validation
        print("\n" + "=" * 100)
        print("MILESTONE 3 VALIDATION")
        print("=" * 100)
        
        checks = {
            "todos_created": todos_count == 5,
            "all_steps_completed": len(result.get('completed_steps', [])) >= 5,
            "delegation_occurred": total_delegations > 0,
            "researcher_used": delegation_stats['researcher_calls'] > 0,
            "writer_used": delegation_stats['writer_calls'] > 0,
            "reviewer_used": delegation_stats['reviewer_calls'] > 0,
            "files_created": len(files) > 0,
            "tools_used": total_operations > 0
        }
        
        for check_name, passed in checks.items():
            icon = "✓" if passed else "✗"
            display_name = check_name.replace('_', ' ').title()
            print(f"  {icon} {display_name}: {passed}")
        
        all_passed = all(checks.values())
        
        print("\n" + "=" * 100)
        if all_passed:
            print("🎉 MILESTONE 3: PASSED ✓")
            print("Multi-agent collaboration successful!")
        else:
            print("⚠️  MILESTONE 3: PARTIAL PASS")
            print("Some features not working as expected")
        print("=" * 100)
        
        print(f"\n📊 LangSmith Trace:")
        print("   https://smith.langchain.com/ → Project: milestone3")
        print("=" * 100)
        
        return all_passed
        
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    print("\n" + "=" * 100)
    print("  MILESTONE 3: MULTI-AGENT COLLABORATION TEST")
    print("  Supervisor + Researcher + Writer + Reviewer")
    print("=" * 100 + "\n")
    
    success = run_milestone3()
    
    print(f"\n{'='*100}")
    print(f"Result: {'✅ PASSED' if success else '❌ FAILED'}")
    print(f"{'='*100}\n")
    
    sys.exit(0 if success else 1)