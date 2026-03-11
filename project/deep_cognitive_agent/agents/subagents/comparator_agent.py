"""
Comparator Agent - Compares and contrasts multiple research findings.

Sub-agent used by the supervisor to analyze differences and similarities
across multiple research summaries.
"""


class ComparatorAgent:
    """Compares multiple research findings using an LLM."""

    PROMPT_TEMPLATE = (
        "You are a specialized Comparison Agent.\n"
        "Your ONLY job is to compare and contrast the following research summaries.\n\n"
        "Write a thorough comparison analysis that:\n"
        "1. Identifies key differences between the topics\n"
        "2. Highlights surprising similarities\n"
        "3. Analyzes relative strengths and weaknesses\n"
        "4. Notes complementary aspects\n\n"
        "Task context: {task}\n\n"
        "Research summaries:\n\n{sources}\n\n"
        "Write the comparison analysis now (at least 200 words):"
    )

    def __init__(self, llm):
        self.llm = llm

    def invoke(self, input_data: str) -> str:
        """Compare research findings.

        Args:
            input_data: Formatted string containing task context and sources
                        separated by '|||'. Format: 'task|||sources'

        Returns:
            Comparison analysis text.
        """
        parts = input_data.split("|||", 1)
        task = parts[0].strip()
        sources = parts[1].strip() if len(parts) > 1 else ""

        prompt = self.PROMPT_TEMPLATE.format(task=task, sources=sources)
        response = self.llm.invoke(prompt)
        return response.content
