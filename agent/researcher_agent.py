"""
agent/researcher_agent.py — Specialised research sub-agent.

Focuses purely on information gathering and synthesis. Can be invoked as a
standalone sub-agent by the main graph for deep-dive research tasks.
"""

from __future__ import annotations

from langchain_groq import ChatGroq
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from pathlib import Path

import config
from tools.research.web_search import web_search
from tools.research.summarize import summarize_text
from tools.research.extract_entities import extract_entities
from tools.filesystem.write_file import write_file
from tools.filesystem.read_file import read_file
from tools.filesystem.ls import ls
from utils.logger import get_logger

logger = get_logger(__name__)

_PROMPT_PATH = Path(__file__).parent.parent / "prompts" / "researcher_prompt.txt"

RESEARCH_TOOLS = [web_search, summarize_text, extract_entities, write_file, read_file, ls]


class ResearcherAgent:
    """
    Deep-research specialist. Searches, summarises, and saves findings.
    """

    def __init__(self, vfs: dict | None = None):
        config.validate_config()
        base = ChatGroq(
            model=config.MODEL_NAME,
            groq_api_key=config.GROQ_API_KEY,
            temperature=0,
        )
        self._llm = base.bind_tools(RESEARCH_TOOLS)
        self._tool_map = {t.name: t for t in RESEARCH_TOOLS}
        self._system = _PROMPT_PATH.read_text() if _PROMPT_PATH.exists() else ""
        self.vfs: dict = vfs or {}

    def research(self, topic: str, save_path: str | None = None, max_steps: int = 8) -> str:
        """
        Research a topic and optionally save the final summary to a VFS path.

        Args:
            topic:     The research question or topic.
            save_path: If provided, saves the final output to this VFS path.
            max_steps: Maximum iterations.

        Returns:
            Research findings as a string.
        """
        logger.info(f"ResearcherAgent.research: {topic[:80]}")

        prompt = f"Research the following topic thoroughly:\n\n{topic}"
        if save_path:
            prompt += f"\n\nSave your final findings to: {save_path}"

        messages = [
            SystemMessage(content=self._system),
            HumanMessage(content=prompt),
        ]

        for step in range(max_steps):
            response: AIMessage = self._llm.invoke(messages)
            messages.append(response)

            if not response.tool_calls:
                return response.content

            for tc in response.tool_calls:
                tool = self._tool_map.get(tc["name"])
                result = tool.invoke(tc["args"]) if tool else f'{{"error": "unknown tool {tc["name"]}"}}'
                messages.append(
                    ToolMessage(content=result, tool_call_id=tc["id"], name=tc["name"])
                )

        return "Research step limit reached."

