"""
Refiner Agent - Refines and enhances existing content.

Sub-agent used by the supervisor to improve documents with additional
considerations, depth, and practical guidance.
"""


class RefinerAgent:
    """Refines existing content using an LLM."""

    PROMPT_TEMPLATE = (
        "You are a specialized Refinement Agent.\n"
        "Your ONLY job is to refine and enhance the existing document "
        "based on the given instruction.\n\n"
        "Refinement instruction: {task}\n\n"
        "Current content:\n{existing_content}\n\n"
        "Improve the content by:\n"
        "1. Adding depth and nuance based on the refinement instruction\n"
        "2. Strengthening weak arguments\n"
        "3. Adding practical considerations\n"
        "4. Ensuring coherence and completeness\n\n"
        "Provide the COMPLETE improved version (not just changes):"
    )

    def __init__(self, llm):
        self.llm = llm

    def invoke(self, input_data: str) -> str:
        """Refine existing content.

        Args:
            input_data: Formatted string containing task context and existing content
                        separated by '|||'. Format: 'task|||existing_content'

        Returns:
            Refined content text.
        """
        parts = input_data.split("|||", 1)
        task = parts[0].strip()
        existing_content = parts[1].strip() if len(parts) > 1 else ""

        prompt = self.PROMPT_TEMPLATE.format(
            task=task, existing_content=existing_content
        )
        response = self.llm.invoke(prompt)
        return response.content
