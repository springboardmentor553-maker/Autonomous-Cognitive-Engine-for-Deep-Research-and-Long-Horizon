import sys
import os
import time

# 1. Path Fix: Ensure the script can find app.py
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app import create_planning_agent, run_agent

def test_milestone3_chain_of_command():
    print("="*80)
    print("MILESTONE 3: SPECIALIZED MULTI-AGENT COLLABORATION")
    print("="*80)

    agent = create_planning_agent()
    
    # COMPLEX TASK: Forces the use of Researcher, Summarizer, Comparator, and Refiner.
    # Note: We do NOT tell it which agents to use. It must decide autonomously.
    task = """
    Perform a professional technical comparison between NVIDIA's 'Blackwell' GPU 
    architecture and AMD's 'Instinct MI325X'. 
    
    Requirements:
    1. Research the core technical specs for both.
    2. Provide a summarized comparison of their performance targets.
    3. Generate a final, refined professional report saved as 'gpu_battle_2025.txt'.
    """
    
    print("🚀 Initializing Chain of Command...")
    print("Expected Flow: Planning -> Researching -> Summarizing -> Comparing -> Refining -> Storing")
    
    thread_id = f"m3-specialist-{int(time.time())}"
    run_agent(agent, task, thread_id=thread_id)
    
    print("\n" + "="*30)
    print("✅ TEST COMPLETE")
    print("Check LangSmith for: 'task_delegate' calls to all 4 specialists.")
    print("Check Virtual FS for: 'gpu_battle_2025.txt'")
    print("="*80)

if __name__ == "__main__":
    test_milestone3_chain_of_command()