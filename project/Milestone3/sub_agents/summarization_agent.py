import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda

load_dotenv()

llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    google_api_key=os.getenv("GOOGLE_API_KEY")
)

summary_prompt = PromptTemplate(
    input_variables=["text"],
    template="""
You are a specialized summarization agent.
Summarize the following text clearly and concisely in bullet points.

Text:
{text}

Provide a structured summary.
"""
)

def summarization_agent(text: str) -> str:
    prompt = summary_prompt.format(text=text)
    response = llm.invoke(prompt)
    return response.content

summarizer = RunnableLambda(summarization_agent)