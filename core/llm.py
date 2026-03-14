"""
Centralised LLM factory for the Autonomous Cognitive Engine.

All agents must import their LLM from this module so that the
model and its configuration live in exactly one place.

Model priority (all free on Groq):
  1. llama-3.3-70b-versatile  – best reasoning, 100k TPD limit
  2. llama-3.1-70b-versatile  – separate token pool, good fallback
  3. mixtral-8x7b-32768        – high free limit, capable fallback

Set GROQ_MODEL in .env to override.
"""

import os
from functools import lru_cache

from langchain_groq import ChatGroq


@lru_cache(maxsize=1)
def get_llm() -> ChatGroq:
    """
    Return a cached ChatGroq instance configured for the project.

    The model can be overridden via the GROQ_MODEL environment variable.
    Default is llama-3.3-70b-versatile.

    Returns
    -------
    ChatGroq
        Ready-to-use LangChain chat model backed by Groq.

    Raises
    ------
    ValueError
        If GROQ_API_KEY is not set in the environment.
    """
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError(
            "GROQ_API_KEY environment variable is not set. "
            "Please add it to your .env file."
        )

    # Allow runtime override via env var — useful for switching when
    # rate limits are hit on one model
    model = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")

    return ChatGroq(
        api_key=api_key,
        model=model,
        temperature=0,
    )
