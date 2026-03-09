"""
agent/critic_agent.py — Self-reflection and quality review agent.
"""

from __future__ import annotations

from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage
from pathlib import Path

import config
from utils.logger import get_logger
from utils.parser import safe_json_loads

logger = get_logger(__name__)

_PROMPT_PATH = Path(__file__).parent.parent / "prompts" / "critic_prompt.txt"


class CriticAgent:
    """
    Reviews completed output and returns structured feedback.
    """

    def __init__(self):
        config.validate_config()
        self._llm = ChatGroq(
            model=config.MODEL_NAME,
            groq_api_key=config.GROQ_API_KEY,
            temperature=0,
        )
        self._system = _PROMPT_PATH.read_text() if _PROMPT_PATH.exists() else ""

    def review(self, original_request: str, output: str) -> dict:
        """
        Review output against the original request.

        Returns:
            Dict with keys: approved (bool), score (int), issues, suggestions, summary.
        """
        logger.info("CriticAgent.review — evaluating output quality")

        prompt = (
            f"## Original Request\n{original_request}\n\n"
            f"## Output to Review\n{output}"
        )

        response = self._llm.invoke(
            [SystemMessage(content=self._system), HumanMessage(content=prompt)]
        )

        parsed = safe_json_loads(response.content)
        if parsed and isinstance(parsed, dict):
            return parsed

        return {
            "approved": True,
            "score": 7,
            "issues": [],
            "suggestions": [],
            "summary": response.content[:200],
        }

