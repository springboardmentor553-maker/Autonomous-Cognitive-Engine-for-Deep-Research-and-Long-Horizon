

"""
tools/research/extract_entities.py — Named entity extraction helper.
"""

from __future__ import annotations

import json
import re

from langchain_core.tools import tool

from utils.logger import get_logger

logger = get_logger(__name__)


@tool
def extract_entities(text: str, entity_types: list[str] | None = None) -> str:
    """
    Extract named entities (people, organisations, dates, etc.) from text.

    Uses simple heuristic pattern matching.  For high-quality extraction,
    combine this with an LLM reasoning step.

    Args:
        text:         Source text.
        entity_types: Optional filter. Choices: "DATE", "URL", "EMAIL".
                      If omitted, all types are extracted.

    Returns:
        JSON with extracted entities grouped by type.
    """
    types = set(entity_types or ["DATE", "URL", "EMAIL"])
    entities: dict[str, list[str]] = {}

    if "URL" in types:
        urls = re.findall(r"https?://\S+", text)
        entities["URL"] = list(set(urls))

    if "EMAIL" in types:
        emails = re.findall(r"[\w.+-]+@[\w-]+\.\w+", text)
        entities["EMAIL"] = list(set(emails))

    if "DATE" in types:
        dates = re.findall(
            r"\b(?:\d{4}[-/]\d{2}[-/]\d{2}|(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?\s+\d{1,2},?\s+\d{4}|\d{1,2}\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?\s+\d{4})\b",
            text,
            re.IGNORECASE,
        )
        entities["DATE"] = list(set(dates))

    logger.info(f"extract_entities → {sum(len(v) for v in entities.values())} found")
    return json.dumps({"entities": entities})
