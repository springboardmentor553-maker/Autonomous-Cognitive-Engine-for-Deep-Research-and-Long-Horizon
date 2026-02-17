import os
from dotenv import load_dotenv
import google.generativeai as genai

from tools import write_todos
from state import AgentState

# Load environment variables
load_dotenv()

# Configure Gemini API
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Load Gemini model
model = genai.GenerativeModel("gemini-pro")


def run_agent(user_request: str) -> AgentState:
    """
    Runs the foundational planning agent.
    Takes a complex user request and generates a structured TODO plan.
    """

    # Initialize agent state
    state = AgentState(user_task=user_request)

    # Planning prompt
    planning_prompt = f"""
You are an AI planning agent.
Your task is to break the given complex objective into
clear, logical, and actionable TODO steps.

Objective:
{user_request}

Return only the plan description.
"""

    # Call LLM (Gemini)
    response = model.generate_content(planning_prompt)

    # Create TODOs using planning tool
    todos = write_todos(user_request)

    # Update state
    state.todos = todos
    state.status = "planning_completed"

    # Print LLM reasoning (for transparency / evaluation)
    print("\n🧠 LLM Planning Output:\n")
    print(response.text)

    return state


if __name__ == "__main__":
    print("=== Autonomous Cognitive Agent (Milestone 1) ===\n")

    user_input = input("Enter a complex task:\n> ")

    final_state = run_agent(user_input)

    print("\n📌 Final Agent State:")
    print(final_state)
    print("\n✅ TODOs saved to generated_todos.json")