import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda

# 1. Load the environment variables from your .env file
load_dotenv(override=True)

# 2. Initialize the Groq LLM correctly
# Note: 'gpt-4o-mini' is an OpenAI model. For Groq, use 'llama-3.1-8b-instant'
llm = ChatGroq(
    model="llama-3.1-8b-instant", 
    temperature=0,
    groq_api_key=os.getenv("GROQ_API_KEY")
)

# 3. Define the Agentic Prompt
summary_prompt = PromptTemplate(
    input_variables=["text"],
    template="""
You are a specialized AI Ethics Summarizer.
Your goal is to extract the core principle from the text below.

TEXT TO SUMMARIZE:
{text}

FORMAT:
- Provide a 1-sentence headline.
- Provide 2-3 bullet points of detail.
"""
)

def summarization_agent(text):
    # Standard LCEL Chain: Prompt | LLM
    chain = summary_prompt | llm
    
    # Execute the chain
    response = chain.invoke({"text": text})
    
    # Return the string content
    return response.content

# 4. Wrap for the Registry
summarizer = RunnableLambda(summarization_agent)