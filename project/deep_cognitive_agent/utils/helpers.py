"""
Utility helper functions for the Deep Cognitive Agent.
"""

import re
import time
from typing import Optional


_THINK_BLOCK_RE = re.compile(r"<think>.*?</think>", re.IGNORECASE | re.DOTALL)


def parse_retry_after(err_str: str, default: int = 30) -> int:
    """Extract recommended wait seconds from a Groq rate-limit error message.

    Args:
        err_str: The error message string from the API.
        default: Default wait time (seconds) if parsing fails.

    Returns:
        Number of seconds to wait before retrying.
    """
    match = re.search(r"try again in (?:(\d+)m)?(\d+(?:\.\d+)?)s", err_str)
    if match:
        minutes = int(match.group(1) or 0)
        seconds = float(match.group(2))
        return int(minutes * 60 + seconds) + 2  # small safety margin
    return default


def is_rate_limit_error(err_str: str) -> bool:
    """Check if an error string indicates a rate limit."""
    return "429" in err_str or "rate_limit" in err_str.lower()


def is_server_overload_error(err_str: str) -> bool:
    """Check if an error string indicates a transient server overload (503)."""
    return "503" in err_str or "over capacity" in err_str.lower() or "overloaded" in err_str.lower()


def invoke_with_retry(llm, prompt: str, max_retries: int = 3) -> str:
    """Invoke an LLM with automatic rate-limit retry.

    Args:
        llm: The LLM instance to invoke.
        prompt: The prompt string to send.
        max_retries: Maximum number of retry attempts.

    Returns:
        The text content of the LLM response.
    """
    for attempt in range(max_retries):
        try:
            response = llm.invoke(prompt)
            return response.content
        except Exception as e:
            err_str = str(e)
            if attempt < max_retries - 1:
                if is_rate_limit_error(err_str):
                    wait = parse_retry_after(err_str)
                    print(f"  ⏳ Rate limited. Waiting {wait}s before retry "
                          f"{attempt + 2}/{max_retries}...")
                    time.sleep(wait)
                    continue
                if is_server_overload_error(err_str):
                    wait = min(2 ** attempt * 10, 60)
                    print(f"  ⏳ Server overloaded (503). Waiting {wait}s before retry "
                          f"{attempt + 2}/{max_retries}...")
                    time.sleep(wait)
                    continue
            raise


def truncate(text: str, max_length: int = 200, suffix: str = "...") -> str:
    """Truncate a string to max_length, adding suffix if truncated."""
    if len(text) <= max_length:
        return text
    return text[:max_length - len(suffix)] + suffix


def strip_think_blocks(text: str) -> str:
    """Remove any <think>...</think> blocks from model output."""
    if not isinstance(text, str) or not text:
        return ""
    return _THINK_BLOCK_RE.sub("", text).strip()


def sanitize_llm_output(text: str) -> str:
    """Normalize model output before storing or presenting it."""
    cleaned = strip_think_blocks(text)
    # Collapse excessive blank lines while preserving paragraph breaks.
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip()
