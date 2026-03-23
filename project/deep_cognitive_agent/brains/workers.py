from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)

# Specialized Prompts for each Worker
PROMPTS = {
    "researcher": "You are a research specialist. Gather technical details on: {input}",
    "summarizer": "You are a summarization specialist. Condense this text professionally: {input}",
    "comparator": "Compare these findings and identify key differences/similarities: {input}",
    "unifier": "Integrate these disparate points into one cohesive, unified report: {input}",
    "refiner": "Review the following report and improve its clarity, tone, and structure: {input}"
}

def worker_logic(role: str, data: str):
    """The generic engine for all specialized sub-agents."""
    prompt = ChatPromptTemplate.from_template(PROMPTS[role])
    chain = prompt | llm
    return chain.invoke({"input": data}).content