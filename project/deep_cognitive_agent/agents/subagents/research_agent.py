"""
Research Agent - Generates detailed research paragraphs.

Sub-agent used by the execution node to produce long-form
research content for individual topics.
"""

from typing import Optional


class ResearchAgent:
    """Generates detailed research content using an LLM."""

    def __init__(self, llm):
        self.llm = llm

    def research(self, topic: str) -> str:
        """Generate a detailed paragraph about the given topic.

        Args:
            topic: Research topic description.

        Returns:
            A long paragraph with facts and analysis.
        """
        prompt = (
            f"Write one long, detailed paragraph (at least 150 words) about "
            f"the following topic. Include specific facts, data points, and "
            f"expert analysis where appropriate.\n\n"
            f"Topic: {topic}\n\n"
            f"Write ONLY the paragraph — no titles, headings, or extra formatting."
        )
        response = self.llm.invoke(prompt)
        return response.content
