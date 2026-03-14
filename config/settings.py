"""
Application settings loaded from environment variables.

All configuration is centralised here so nothing is hardcoded
elsewhere in the codebase.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field


@dataclass(frozen=True)
class Settings:
    """
    Immutable application settings derived from environment variables.

    Attributes
    ----------
    groq_api_key : str
        API key for Groq LLM provider.
    tavily_api_key : str
        API key for Tavily web search.
    langchain_tracing : bool
        Whether LangSmith tracing is enabled.
    langchain_project : str
        LangSmith project name for trace grouping.
    langchain_api_key : str
        LangSmith API key (optional, only needed if tracing is on).
    groq_model : str
        Groq model identifier to use.
    """

    groq_api_key: str = field(default_factory=lambda: os.getenv("GROQ_API_KEY", ""))
    tavily_api_key: str = field(default_factory=lambda: os.getenv("TAVILY_API_KEY", ""))
    langchain_tracing: bool = field(
        default_factory=lambda: os.getenv("LANGCHAIN_TRACING_V2", "false").lower() == "true"
    )
    langchain_project: str = field(
        default_factory=lambda: os.getenv("LANGCHAIN_PROJECT", "deep-cognitive-agent")
    )
    langchain_api_key: str = field(
        default_factory=lambda: os.getenv("LANGCHAIN_API_KEY", "")
    )
    groq_model: str = field(
        default_factory=lambda: os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
    )

    def validate(self) -> None:
        """
        Raise ValueError if any mandatory API key is missing.

        Raises
        ------
        ValueError
        """
        missing: list[str] = []
        if not self.groq_api_key:
            missing.append("GROQ_API_KEY")
        if not self.tavily_api_key:
            missing.append("TAVILY_API_KEY")
        if self.langchain_tracing and not self.langchain_api_key:
            missing.append("LANGCHAIN_API_KEY (required when LANGCHAIN_TRACING_V2=true)")

        if missing:
            raise ValueError(
                "The following required environment variables are not set:\n"
                + "\n".join(f"  - {k}" for k in missing)
                + "\n\nPlease copy .env.example to .env and fill in the values."
            )


def get_settings() -> Settings:
    """Return a validated Settings instance."""
    s = Settings()
    s.validate()
    return s
