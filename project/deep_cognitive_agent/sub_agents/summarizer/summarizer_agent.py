from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.runnables import RunnableLambda
import os


llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0,
    google_api_key=os.getenv("GOOGLE_API_KEY"),
)


def summarization_logic(text: str) -> str:
    """
    Core logic of the summarization worker agent.
    """

    print("[WORKER AGENT] Summarization Agent Running")

    response = llm.invoke(
        f"""
You are a specialized summarization agent.

Summarize the following content clearly.

CONTENT:
{text}

Return a concise summary.
"""
    )

    return response.content


# Wrap as runnable so LangSmith traces it as a worker agent
summarization_agent = RunnableLambda(summarization_logic)