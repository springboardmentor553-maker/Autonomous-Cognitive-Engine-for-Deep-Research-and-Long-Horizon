import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda

load_dotenv()

llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)

# ──────────────────────────────────────────────
# Sub-Agent 1 : Research Agent
# Purpose      : Gather key facts on a topic
# Toolset      : LLM only
# ──────────────────────────────────────────────

research_prompt = PromptTemplate(
    input_variables=["query"],
    template="""You are a specialized research agent.
Your only job is to find and organize relevant information on the given topic.

Query: {query}

Provide:
- Key Facts (3-5 bullet points)
- Recent Developments (2-3 points)
- Relevant Context (1-2 sentences)

Be concise. Do not repeat yourself."""
)

def research_agent_fn(query: str) -> str:
    response = llm.invoke(research_prompt.format(query=query))
    return response.content


# ──────────────────────────────────────────────
# Sub-Agent 2 : Summarization Agent
# Purpose      : Summarize text clearly
# Toolset      : LLM only
# ──────────────────────────────────────────────

summary_prompt = PromptTemplate(
    input_variables=["text"],
    template="""You are a specialized summarization agent.
Your only job is to summarize the following text clearly and concisely.

Text: {text}

Provide:
- Key Points (bullet list)
- Main Conclusion (1-2 sentences)"""
)

def summarization_agent_fn(text: str) -> str:
    response = llm.invoke(summary_prompt.format(text=text))
    return response.content


# ──────────────────────────────────────────────
# Registry — plain dict (mentor pattern)
# ──────────────────────────────────────────────

sub_agents = {
    "research_agent":      RunnableLambda(research_agent_fn),
    "summarization_agent": RunnableLambda(summarization_agent_fn),
}