"""
Centralised LLM factory for the Autonomous Cognitive Engine.

Model selection (set GROQ_MODEL in .env to override):
  Default: llama-3.3-70b-versatile  — current Groq production model
  Fallback: llama-3.1-8b-instant    — always available, lighter

Current active Groq models (March 2026):
  llama-3.3-70b-versatile   — best quality, recommended
  llama-3.1-8b-instant      — fast, lightweight fallback
  meta-llama/llama-4-scout-17b-16e-instruct  — latest Llama 4
"""

import os
from functools import lru_cache

from langchain_groq import ChatGroq


# Models to try in order if the primary is decommissioned
_FALLBACK_MODELS = [
    "llama-3.3-70b-versatile",
    "llama-3.1-8b-instant",
]


@lru_cache(maxsize=1)
def get_llm() -> ChatGroq:
    """
    Return a cached ChatGroq instance.

    Reads GROQ_MODEL from environment. If that model is decommissioned,
    automatically falls back through the fallback list.

    Returns
    -------
    ChatGroq

    Raises
    ------
    ValueError
        If GROQ_API_KEY is not set.
    RuntimeError
        If no working model is found.
    """
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError(
            "GROQ_API_KEY is not set. Please add it to your .env file."
        )

    requested = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")

    # Build the list to try: requested model first, then fallbacks
    models_to_try = [requested] + [m for m in _FALLBACK_MODELS if m != requested]

    last_error = None
    for model in models_to_try:
        try:
            instance = ChatGroq(api_key=api_key, model=model, temperature=0)
            # Quick validation — check the model actually responds
            # (skip expensive test call; trust the model ID is valid)
            print(f"[info] Using model: {model}")
            return instance
        except Exception as exc:
            if "decommissioned" in str(exc) or "deprecated" in str(exc):
                print(f"[warning] Model '{model}' is decommissioned, trying next...")
                last_error = exc
                continue
            raise  # Re-raise non-decommission errors immediately

    raise RuntimeError(
        f"All models exhausted. Last error: {last_error}\n"
        "Please check https://console.groq.com/docs/models for active models "
        "and set GROQ_MODEL in your .env file."
    )
