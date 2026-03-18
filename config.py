"""
config.py — Model and environment configuration.
All settings are loaded from .env (or environment variables).
"""

import os
from dotenv import load_dotenv

load_dotenv()


# ─── LLM ──────────────────────────────────────────────────────────────────────

# Available Groq models (free tier friendly):
#   llama-3.3-70b-versatile   ← best for complex reasoning (recommended)
#   llama3-8b-8192            ← fastest / lightest
#   mixtral-8x7b-32768        ← long context (32k tokens)
#   gemma2-9b-it              ← Google Gemma 2
MODEL_NAME: str = os.getenv("MODEL_NAME", "llama-3.3-70b-versatile")

# Maximum number of agent iterations before forced stop
MAX_ITERATIONS: int = int(os.getenv("MAX_ITERATIONS", "50"))

# ─── API Keys ─────────────────────────────────────────────────────────────────

GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")
TAVILY_API_KEY: str = os.getenv("TAVILY_API_KEY", "")

# ─── LangSmith ────────────────────────────────────────────────────────────────

LANGCHAIN_TRACING_V2: str = os.getenv("LANGCHAIN_TRACING_V2", "false")
LANGCHAIN_API_KEY: str = os.getenv("LANGCHAIN_API_KEY", "")
LANGCHAIN_PROJECT: str = os.getenv("LANGCHAIN_PROJECT", "autonomous-cognitive-engine")

# ─── Validation ───────────────────────────────────────────────────────────────

def validate_config() -> None:
    """Raise early if required keys are missing."""
    missing = []
    if not GROQ_API_KEY:
        missing.append("GROQ_API_KEY")
    if not TAVILY_API_KEY:
        missing.append("TAVILY_API_KEY")
    if missing:
        raise EnvironmentError(
            f"Missing required environment variables: {', '.join(missing)}\n"
            "Please fill in your API keys in the .env file.\n"
            "Get a free Groq key at: https://console.groq.com"
        )