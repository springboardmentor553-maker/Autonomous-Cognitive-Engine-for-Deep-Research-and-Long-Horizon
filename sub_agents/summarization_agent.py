"""
sub_agents/summarization_agent.py
====================================
Summarization Sub-Agent

Mentor spec:
  - Specific purpose    : summarize text clearly and concisely
  - Small focused prompt: one PromptTemplate, one job only
  - Limited toolset     : LLM only — nothing else
  - Clear responsibility: receive text -> return summary -> nothing else

Mentor example (we use Groq instead of OpenAI):
    from langchain.chat_models import ChatOpenAI
    from langchain.prompts import PromptTemplate
    from langchain.schema.runnable import RunnableLambda

    llm = ChatOpenAI(model="gpt-4")  # we use ChatGroq

    summary_prompt = PromptTemplate(...)
    def summarization_agent(text): ...
    summarizer = RunnableLambda(summarization_agent)
"""

from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda
from langchain_groq import ChatGroq

import config

# ── LLM setup (Groq instead of OpenAI — mentor said "you can use any") ────────
llm = ChatGroq(
    model=config.MODEL_NAME,     # llama-3.3-70b-versatile
    api_key=config.GROQ_API_KEY,
    temperature=0.1,
    max_tokens=600,
)

# ── Small focused prompt (mentor spec: one job only) ──────────────────────────
summary_prompt = PromptTemplate(
    input_variables=["text"],
    template="""
You are a specialized summarization agent.
Your task is to summarize the following text clearly and concisely.

Text:
{text}

Provide a structured summary with:
- Overview (2-3 sentences)
- Key Points (3 bullet points)
- Conclusion (1 sentence)
""",
)


# ── Agent function (mentor exact pattern) ────────────────────────────────────
def summarization_agent(text: str) -> str:
    """
    Single responsibility: receive text, return structured summary.
    This agent does nothing else — no file writes, no planning, no state.
    """
    prompt = summary_prompt.format(text=text)
    response = llm.invoke(prompt)
    return response.content


# ── Wrap as RunnableLambda (mentor exact pattern) ─────────────────────────────
summarizer = RunnableLambda(summarization_agent)