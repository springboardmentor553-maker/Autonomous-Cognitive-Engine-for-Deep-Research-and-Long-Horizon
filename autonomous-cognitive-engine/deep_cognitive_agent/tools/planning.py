from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

def write_todos(goal: str):
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    prompt = f"""
    You are an intelligent planning agent.
    Break the following goal into clear step-by-step TODOs.

    Goal: {goal}
    """

    response = llm.invoke(prompt)
    return response.content

