from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.schema.runnable import RunnableLambda

llm = ChatOpenAI(model="gpt-4")#u can use any here

summary_prompt = PromptTemplate(
    input_variables=["text"],
    template="""
    You are a specialized summarization agent.
    Your task is to summarize the following text clearly and concisely.

    Text:
    {text}

    Provide a structured summary.
    """
)

def summarization_agent(text):
    prompt = summary_prompt.format(text=text)
    return llm.predict(prompt)

summarizer = RunnableLambda(summarization_agent)
