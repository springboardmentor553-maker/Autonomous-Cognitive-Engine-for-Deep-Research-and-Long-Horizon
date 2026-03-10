

"""
tools/research/summarize.py — In-agent text summarisation helper tool.
"""

from __future__ import annotations

import json

from langchain_core.tools import tool

from utils.logger import get_logger

logger = get_logger(__name__)


@tool
def summarize_text(text: str, max_words: int = 200, focus: str = "") -> str:
    """
    Produce a concise summary of the provided text.

    This is a lightweight extraction tool — for heavy summarisation the agent
    should reason directly.  Primarily used to compress search snippets before
    saving them to the virtual file system.

    Args:
        text:      The source text to summarise.
        max_words: Approximate maximum word count for the summary.
        focus:     Optional topic/question to focus the summary around.

    Returns:
        JSON with the summary string.
    """
    # Simple extractive approach: take first N sentences proportional to max_words
    sentences = [s.strip() for s in text.replace("\n", " ").split(". ") if s.strip()]
    budget = max_words * 6  # ~6 chars per word on average
    result_parts = []
    used = 0

    for sentence in sentences:
        if used + len(sentence) > budget:
            break
        result_parts.append(sentence)
        used += len(sentence)

    summary = ". ".join(result_parts)
    if result_parts and not summary.endswith("."):
        summary += "."

    logger.info(f"summarize_text → {len(summary.split())} words")
    return json.dumps({"summary": summary, "focus": focus})
