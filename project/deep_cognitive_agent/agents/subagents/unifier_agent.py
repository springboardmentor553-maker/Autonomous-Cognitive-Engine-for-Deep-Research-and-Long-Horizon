"""
Unifier Agent - Proposes a unified model/framework from analysis results.

Sub-agent used by the supervisor to integrate findings from comparison
analysis into a comprehensive unified proposal.
"""


class UnifierAgent:
    """Proposes unified models by integrating analysis results using an LLM."""

    PROMPT_TEMPLATE = (
        "You are a specialized Unification Agent.\n"
        "Your ONLY job is to propose a comprehensive unified model or framework "
        "that integrates the best elements from the provided analysis.\n\n"
        "Task: {task}\n\n"
        "Source analysis:\n{sources}\n\n"
        "Create a well-structured proposal with:\n"
        "1. Core principles of the unified model\n"
        "2. Key components and how they integrate\n"
        "3. Implementation approach\n"
        "4. Expected benefits\n\n"
        "Write the unified model proposal now (at least 200 words):"
    )

    def __init__(self, llm):
        self.llm = llm

    def invoke(self, input_data: str) -> str:
        """Propose a unified model.

        Args:
            input_data: Formatted string containing task context and sources
                        separated by '|||'. Format: 'task|||sources'

        Returns:
            Unified model proposal text.
        """
        parts = input_data.split("|||", 1)
        task = parts[0].strip()
        sources = parts[1].strip() if len(parts) > 1 else ""

        prompt = self.PROMPT_TEMPLATE.format(task=task, sources=sources)
        response = self.llm.invoke(prompt)
        return response.content
