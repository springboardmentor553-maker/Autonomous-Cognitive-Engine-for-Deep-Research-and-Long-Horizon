"""
Summarizer Agent - Combines multiple summaries into one structured output.

Sub-agent used by the supervisor to merge individual
research summaries into a final cohesive document.
"""


class SummarizerAgent:
    """Combines multiple summaries into a structured final output."""

    PROMPT_TEMPLATE = (
        "You are a specialized Summarization Agent.\n"
        "Your ONLY job is to summarize the given content clearly and concisely.\n\n"
        "Create ONE final structured summary that combines all key points.\n\n"
        "Use this structure:\n"
        "1. **Overview**: Brief introduction\n"
        "2. **Key Findings**: Bullet points of main facts\n"
        "3. **Analysis**: Deeper analysis connecting themes\n"
        "4. **Conclusion**: Final takeaway\n\n"
        "Content to summarize:\n\n{text}\n\n"
        "Write the structured summary now:"
    )

    def __init__(self, llm):
        self.llm = llm

    def invoke(self, input_data: str) -> str:
        """Create a structured summary from the given text.

        Args:
            input_data: Text content to summarize (may be multiple sections).

        Returns:
            A structured summary string.
        """
        prompt = self.PROMPT_TEMPLATE.format(text=input_data)
        response = self.llm.invoke(prompt)
        return response.content

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
        return self.invoke(combined)
