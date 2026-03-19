import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate

load_dotenv(override=True)

# Initialize Groq
llm = ChatGroq(
    model="llama-3.1-8b-instant",
    groq_api_key=os.getenv("GROQ_API_KEY"),
    temperature=0.1
)

summary_prompt = PromptTemplate(
    input_variables=["text"],
    template="""
You are an expert AI Ethics Researcher.
Summarize the following raw framework data into a single, powerful sentence.

RAW DATA:
{text}

CONCISE SUMMARY:"""
)

def summarization_agent(text):
    chain = summary_prompt | llm
    response = chain.invoke({"text": text})
    return response.content.strip()