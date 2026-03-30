from langchain_core.runnables import RunnableConfig
from langchain_core.messages import AIMessage
from brains.sub_agents import SUB_AGENTS
from graphs.state import AgentState
from tools.execution.file_tools import write_file
import time

def execute_researcher(state: AgentState, config: RunnableConfig):
    return _execute_worker("researcher", state, config)

def execute_summarizer(state: AgentState, config: RunnableConfig):
    return _execute_worker("summarizer", state, config)
    
def execute_comparator(state: AgentState, config: RunnableConfig):
    return _execute_worker("comparator", state, config)

def execute_refiner(state: AgentState, config: RunnableConfig):
    return _execute_worker("refiner", state, config)

def _execute_worker(role: str, state: AgentState, config: RunnableConfig):
    todos = state.get("todos", [])
    
    for todo in todos:
        if todo.get("status") == "pending":
            task_text = todo["task"]
            
            # 1. Invoke the LCEL chain (Restores the perfect tool tracing)
            chain_response = SUB_AGENTS[role].invoke(
                {"input": task_text}, 
                config={"run_name": f"{role}_specialist", **config}
            )
            result = chain_response.content
            
            # 2. Save the File (Will now properly attach to the trace)
            if "gpu_battle_2025.txt" in task_text.lower():
                filename = "gpu_battle_2025.txt"
            else:
                filename = f"{role}_output.txt"
                
            write_file.invoke(
                {"filename": filename, "content": result},
                config={"run_name": f"{role}_file_save", **config}
            )
            
            # 3. Update status
            todo["status"] = "done"
            
            # 4. Inject into State (Restores the "Final Answer" output in the graph)
            new_message = AIMessage(content=f"[{role.upper()} FINAL OUTPUT]:\n{result}")
            
            print(f"⏳ Sleeping for 4s to respect API rate limits...")
            time.sleep(4)

            return {
                "todos": todos, 
                "messages": [new_message] 
            } 
            
    return {"todos": todos}