from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)

PROMPTS = {
    "researcher": "You are a research specialist. Gather technical details on: {input}",
    "summarizer": "You are a summarization specialist. Condense this text professionally: {input}",
    "comparator": "Compare these findings and identify key differences/similarities: {input}",
    "unifier": "Integrate these disparate points into one cohesive, unified report: {input}",
    "refiner": "Review the following report and improve its clarity, tone, and structure: {input}"
}

def get_worker_chain(role: str):
    """Returns the native LCEL chain so LangSmith can track it perfectly."""
    prompt = ChatPromptTemplate.from_template(PROMPTS[role])
    return prompt | llm