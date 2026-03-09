import dotenv
import os
# Load environment variables before importing modules that depend on them
dotenv.load_dotenv()

from src.agent import app
from langchain_core.messages import HumanMessage

def test_milestone1_planning():
    if not os.getenv("GROQ_API_KEY"):
        print("GROQ_API_KEY not found, skipping test")
        return

    initial_message = "Create a detailed study plan for learning Quantum Computing in 4 weeks."
    initial_state = {"messages": [HumanMessage(content=initial_message)], "todos": []}
    
    print("Running agent with input:", initial_message)
    
    # Run the graph
    final_state = app.invoke(initial_state)
    
    # Verify that todos were generated and stored in state
    if not final_state.get("todos"):
        raise AssertionError("No TODOs found in final state")
    if len(final_state["todos"]) == 0:
        raise AssertionError("TODO list is empty")
    if len(final_state["messages"]) <= 1:
        raise AssertionError("Agent did not respond")
    
    print("\nTest passed: Plan generated successfully.")
    print("Generated TODOs:", final_state["todos"])

if __name__ == "__main__":
    try:
        test_milestone1_planning()
    except Exception as e:
        print(f"Test failed: {e}")
        import traceback
        traceback.print_exc()
        import traceback
        traceback.print_exc()
