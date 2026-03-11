"""
Research Agent - Generates detailed research paragraphs.

Sub-agent used by the supervisor to produce long-form
research content for individual topics.
"""


class ResearchAgent:
    """Generates detailed research content using an LLM."""

    PROMPT_TEMPLATE = (
        "You are a specialized Research Agent.\n"
        "Your ONLY job is to research and write detailed content.\n\n"
        "Write one long, detailed paragraph (at least 150 words) about "
        "the following topic. Include specific facts, data points, and "
        "expert analysis where appropriate.\n\n"
        "Topic: {topic}\n\n"
        "Write ONLY the paragraph — no titles, headings, or extra formatting."
    )

    def __init__(self, llm):
        self.llm = llm

    def invoke(self, input_data: str) -> str:
        """Generate a detailed paragraph about the given topic.

        Args:
            input_data: Research topic description string.

        Returns:
            A long paragraph with facts and analysis.
        """
        prompt = self.PROMPT_TEMPLATE.format(topic=input_data)
        response = self.llm.invoke(prompt)
        return response.content

    def research(self, topic: str) -> str:
        """Alias for invoke() — backward compatibility."""
        return self.invoke(topic)
