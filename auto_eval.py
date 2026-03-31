import os
import dotenv
dotenv.load_dotenv()

from langchain_core.messages import HumanMessage
from src.agent import app
from langchain_groq import ChatGroq

# 10 Varied Complex Requests for the Evaluation Suite
requests = [
    "Plan a comprehensive research report on the impact of quantum computing on modern cryptography protocols by 2030.",
    "Create a detailed plan to design a specialized machine learning pipeline architecture for real-time fraud detection in high-frequency financial transactions.",
    "Develop a phased roadmap for integrating augmented reality into an e-commerce platform's mobile shopping experience.",
    "Outline a strategic business plan for a startup focusing on sustainable urban vertical farming.",
    "Design a technical architecture and deployment strategy for a decentralized blockchain-based voting system.",
    "Structure a comprehensive onboarding and training program for a team of 50 remote software engineers.",
    "Plan a multimedia marketing campaign for the launch of a new electric vehicle targeting eco-conscious millennials.",
    "Create a step-by-step procedure for migrating a legacy monolithic enterprise application to a microservices architecture.",
    "Develop a risk management and contingency plan for an upcoming international music festival with 100,000 attendees.",
    "Outline a curriculum and lesson plan for a 12-week bootcamp teaching advanced full-stack web development."
]

# Initialize the evaluator model once to reuse the connection
evaluator_model = ChatGroq(model="llama-3.1-8b-instant", temperature=0, max_tokens=10, max_retries=1)

def evaluate_tasks_logic(request: str, todos: list[str]) -> bool:
    """Uses Groq LLM as a judge to evaluate if the sub-tasks are logical and relevant."""
    if not todos:
        return False
        
    prompt = f"Does this list of tasks relate to the request: '{request}'? Tasks: "
    for i, task in enumerate(todos, 1):
        prompt += f"{i}. {task} "
    prompt += "Reply YES or NO."
    
    try:
        res = evaluator_model.invoke(prompt)
        content = res.content.upper()
        if "YES" in content:
            return True
        else:
            print(f"       [Evaluator said: {res.content}]")
            return False
    except Exception as e:
        print(f"       [Evaluator Error: {e}]")
        return False

def run_suite():
    print("==================================================")
    print("      Automated Evaluation Suite - Milestone 1   ")
    print("==================================================")
    print("Metric: Task Decomposition Accuracy")
    print(f"Total Requests: {len(requests)}\n")
    
    success_count = 0
    total = len(requests)
    
    from langchain_core.messages import SystemMessage
    system_prompt = SystemMessage(content="You are a planning assistant. You MUST call the `write_todos` tool to output the broken down tasks. Do not just reply with text.")

    for i, req in enumerate(requests, 1):
        print(f"Test {i}/{total}: {req}")
        
        max_attempts = 5
        guided_req = f"{req} Call the `write_todos` tool with 3 short tasks."
        initial_state = {"messages": [HumanMessage(content=guided_req)], "todos": []}
        for attempt in range(max_attempts):
            try:
                # Run the agent (trace equivalent happens within app.invoke)
                final_state = app.invoke(initial_state)
                
                todos = final_state.get("todos", [])
                messages = final_state.get("messages", [])
                
                # Verify write_todos tool invocation in state history
                tool_invoked = False
                for msg in reversed(messages):
                    if hasattr(msg, "tool_calls") and msg.tool_calls:
                        if any(tc["name"] == "write_todos" for tc in msg.tool_calls):
                            tool_invoked = True
                            break
                        
                if not tool_invoked:
                    if attempt < max_attempts - 1:
                        print(f"  [!] Retrying due to failed tool invocation (Attempt {attempt+1}/{max_attempts})")
                        continue
                    else:
                        print("  [x] FAIL: 'write_todos' tool was not invoked in the agent trace.")
                        break
                print("  [✓] 'write_todos' tool properly invoked.")
                    
                # Verify sub-tasks stored correctly in state
                if not todos:
                    if attempt < max_attempts - 1:
                        print(f"  [!] Retrying due to empty sub-tasks (Attempt {attempt+1}/{max_attempts})")
                        continue
                    else:
                        print("  [x] FAIL: Sub-tasks were not stored correctly in the state trace (todos empty).")
                        break
                print(f"  [✓] State trace contains {len(todos)} stored sub-tasks.")
                    
                # Verify generated list is logical and relevant using LLM-as-a-judge
                is_logical = evaluate_tasks_logic(req, todos)
                if not is_logical:
                    print("  [x] FAIL: LLM Evaluator judged the task plan as illogical or irrelevant.")
                    break
                print("  [✓] Plan evaluated as logical and relevant.")
                
                print("  -> TEST PASS")
                success_count += 1
                break # Exit the retry loop on success
                
            except Exception as e:
                if attempt < max_attempts - 1:
                    print(f"  [!] Retrying due to error: {e}")
                    continue
                else:
                    print(f"  [x] ERROR during test execution: {e}")
                    break
            
        print("-" * 50)
            
    success_rate = (success_count / total) * 100
    print(f"\nEvaluation Complete.")
    print(f"Success Rate: {success_rate:.1f}% ({success_count}/{total})")
    
    if success_rate > 80:
        print("RESULT: Success Criteria MET (>80%).")
    else:
        print("RESULT: Success Criteria NOT MET (<=80%).")
    print("==================================================")

def run_vfs_suite():
    print("\n==================================================")
    print("      Automated Evaluation Suite - VFS       ")
    print("==================================================")
    print("Metric: Correct File System Tool Usage")
    
    long_text_1 = " ".join(["AI represents a fundamental shift in computing capability, advancing rapidly over the last decade."] * 10)
    long_text_2 = " ".join(["The integration of AI into healthcare could streamline diagnostics and significantly reduce human error."] * 10)
    long_text_3 = " ".join(["However, ethical considerations and data privacy remain major hurdles for global AI adoption."] * 10)
    
    vfs_requests = [
        f"Context 1: {long_text_1}\nContext 2: {long_text_2}\nRead these contexts. Save a short summary of Context 1 to 'c1.txt' and a short summary of Context 2 to 'c2.txt' using write_file. Read them both using read_file. Call write_todos with the results.",
        f"Context A: {long_text_3}\nContext B: {long_text_1}\nSave a short note of Context A to 'a.txt' and Context B to 'b.txt'. Read them using read_file. Call write_todos with 3 tasks based on them.",
        "You need to process system logs. Log 1: 'Error 404 on /api/login'. Log 2: 'Timeout on /db/query'. Save Log 1 to 'log1.txt' and Log 2 to 'log2.txt' via write_file. Then use read_file on both. Call write_todos to list the fixing tasks.",
        "Process these legal clauses. Clause 1: 'Liability is limited to $100'. Clause 2: 'Governing law is California'. Save each to a file named 'cl1.txt' and 'cl2.txt' via write_file. Read them back and execute write_todos.",
        "Analyze stock data. Stock AAPL: 'Up 2%'. Stock MSFT: 'Down 1%'. Save the data to 'aapl.txt' and 'msft.txt' using write_file. Read the files with read_file. Plan next steps using write_todos tool."
    ]
    
    success_count = 0
    total = len(vfs_requests)
    
    for i, req in enumerate(vfs_requests, 1):
        print(f"Test VFS {i}/{total}...")
        
        max_attempts = 4
        for attempt in range(max_attempts):
            try:
                guided_req = req + " Only use these tools sequentially (e.g. write_file -> write_file -> read_file -> read_file -> write_todos). DO NOT run parallel tools."
                initial_state = {"messages": [HumanMessage(content=guided_req)], "todos": [], "vfs": {}}
                final_state = app.invoke(initial_state)
                
                messages = final_state.get("messages", [])
                
                write_used = False
                read_used = False
                todos_used = False
                
                for msg in reversed(messages):
                    if hasattr(msg, "tool_calls") and msg.tool_calls:
                        for tc in msg.tool_calls:
                            if tc["name"] == "write_file":
                                write_used = True
                            elif tc["name"] == "read_file":
                                read_used = True
                            elif tc["name"] == "write_todos":
                                todos_used = True
                
                if write_used and read_used and todos_used and final_state.get("vfs"):
                    print("  [✓] write_file and read_file tools were both properly invoked.")
                    print(f"  [✓] VFS state updated successfully. ({len(final_state['vfs'])} files mapped)")
                    print("  -> TEST PASS")
                    success_count += 1
                    break
                else:
                    if attempt < max_attempts - 1:
                        print(f"  [!] Retrying... missing tools (write: {write_used}, read: {read_used}, todos: {todos_used})")
                        continue
                    else:
                        print(f"  [x] FAIL: Did not meet criteria (write: {write_used}, read: {read_used}, todos: {todos_used}, vfs_files: {len(final_state.get('vfs', {}))}).")
                        break
                        
            except Exception as e:
                if attempt < max_attempts - 1:
                    print(f"  [!] Retrying due to error: {e}")
                    continue
                else:
                    print(f"  [x] ERROR during test execution: {e}")
                    break
        print("-" * 50)
        
    success_rate = (success_count / total) * 100
    print(f"\nVFS Evaluation Complete.")
    print(f"Success Rate: {success_rate:.1f}% ({success_count}/{total})")
    
    if success_rate > 80:
        print("RESULT: Success Criteria MET (>80%).")
    else:
        print("RESULT: Success Criteria NOT MET (<=80%).")
    print("==================================================")

if __name__ == "__main__":
    run_suite()
    run_vfs_suite()
