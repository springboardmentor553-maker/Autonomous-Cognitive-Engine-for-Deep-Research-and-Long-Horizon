"""
Summarizer Agent - Combines multiple summaries into one structured output.

Sub-agent used by the synthesis node to merge individual
research summaries into a final cohesive document.
"""


class SummarizerAgent:
    """Combines multiple summaries into a structured final output."""

    def __init__(self, llm):
        self.llm = llm

    def summarize(self, summaries: dict) -> str:
        """Create a structured summary from multiple file contents.

        Args:
            summaries: Dict mapping filename → content.

        Returns:
            A structured summary string.
        """
        combined = "\n\n".join(
            f"--- {fname} ---\n{content}"
            for fname, content in sorted(summaries.items())
        )
        prompt = (
            "You are given individual summaries about a topic. "
            "Create ONE final structured summary that combines all key points.\n\n"
            "Use this structure:\n"
            "1. **Overview**: Brief introduction\n"
            "2. **Key Findings**: Bullet points of main facts\n"
            "3. **Analysis**: Deeper analysis connecting themes\n"
            "4. **Conclusion**: Final takeaway\n\n"
            f"Summaries:\n\n{combined}\n\n"
            "Write the structured summary now:"
        )
        response = self.llm.invoke(prompt)
        return response.content
