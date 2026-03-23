import sys
import os
import time

# Add the project root to the python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app import create_planning_agent, run_agent
from tools.filesystem.storage import vfs

def analyze_tool_sequence(messages):
    """Analyzes the order of tool calls to verify 'Planning First' and 'Selective Retrieval'."""
    sequence = []
    for msg in messages:
        # Check for tool calls in AI messages
        if hasattr(msg, 'tool_calls') and msg.tool_calls:
            for tc in msg.tool_calls:
                sequence.append(tc.get('name'))
    return sequence

def test_milestone2_flow():
    print("="*80)
    print("MILESTONE 2 EVALUATION: AUTONOMOUS CONTEXT OFFLOADING")
    print("="*80)

    agent = create_planning_agent()

    # MENTOR-ALIGNED TASK: We provide the data but DO NOT tell it to save.
    # We want to see if the agent autonomously chooses to use write_file.
    task = """
    I need an architecture comparison for high-end AI chips. 
    
    RESEARCH DATA:
    NVIDIA H100: Uses Hopper architecture, 80GB HBM3 memory, 4th Gen Tensor Cores.
    AMD MI300X: Uses CDNA 3 architecture, 192GB HBM3 memory, 5.3TB/s bandwidth.
    
    Based on this research data, please provide a comparison. 
    Manage your context window efficiently by offloading this technical data 
    into separate summaries before giving the final answer.
    """

    thread_id = f"m2-eval-{int(time.time())}"
    result = run_agent(agent, task, thread_id=thread_id)

    # --- 1. TOOL SEQUENCE ANALYSIS ---
    sequence = analyze_tool_sequence(result["messages"])
    
    print("\n" + "="*20 + " MENTOR CRITERIA EVALUATION " + "="*20)
    
    # Criterion A: Planning First
    if sequence and sequence[0] == "write_todos":
        print("✅ PLANNING FIRST: Agent successfully called write_todos before execution.")
    else:
        print("❌ PLANNING ERROR: Agent failed to plan first.")

    # Criterion B: Autonomous Offloading
    all_files = vfs.ls()
    if any(name in ["write_file", "write_file_tool"] for name in sequence):
        print(f"✅ CONTEXT OFFLOADING: Agent autonomously used write_file. Created: {all_files}")
    else:
        print("❌ OFFLOADING ERROR: Agent did not save summaries to the VFS.")

    # Criterion C: Meaningful Naming
    generic_names = ["file", "data", "test", "output"]
    meaningful = not any(any(g in f.lower() for g in generic_names) for f in all_files)
    if all_files and meaningful:
        print("✅ MEANINGFUL NAMING: Filenames look descriptive and logical.")
    elif all_files:
        print("⚠️  NAMING WARNING: Filenames appear generic (e.g., 'data.txt').")

    # --- 2. FINAL ANSWER VERIFICATION ---
    final_ans = ""
    for msg in reversed(result["messages"]):
        if hasattr(msg, 'content') and msg.type == "ai" and not msg.tool_calls:
            final_ans = str(msg.content)
            break
            
    if final_ans:
        print(f"\n✅ FINAL SYNTHESIS: {final_ans[:100]}...")
    else:
        print("\n❌ SYNTHESIS ERROR: Could not find final AI response.")

    print("="*80)

if __name__ == "__main__":
    test_milestone2_flow()