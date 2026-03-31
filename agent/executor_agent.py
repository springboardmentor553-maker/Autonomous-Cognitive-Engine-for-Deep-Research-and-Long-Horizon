"""
agent/executor_agent.py — Single-task executor interface.

Executes one TODO task using the full tool suite with ReAct reasoning.
The graph uses this logic via nodes.py; this class is for standalone testing.
"""

from __future__ import annotations

from langchain_groq import ChatGroq
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from pathlib import Path

import config
from tools import ALL_TOOLS
from backend.utils.logger import get_logger

logger = get_logger(__name__)

_PROMPT_PATH = Path(__file__).parent.parent / "prompts" / "researcher_prompt.txt"


class ExecutorAgent:
    """
    Single-task ReAct executor. Loops until the LLM produces a response
    with no tool calls (i.e., considers the task complete).
    """

    def __init__(self, vfs: dict | None = None):
        config.validate_config()
        base = ChatGroq(
            model=config.MODEL_NAME,
            groq_api_key=config.GROQ_API_KEY,
            temperature=0,
        )
        self._llm = base.bind_tools(ALL_TOOLS)
        self._tool_map = {t.name: t for t in ALL_TOOLS}
        self._system = _PROMPT_PATH.read_text() if _PROMPT_PATH.exists() else ""
        self.vfs: dict = vfs or {}  # shared mutable VFS

    def execute(self, task: str, context: str = "", max_steps: int = 10) -> str:
        """
        Run the ReAct loop for a single task.

        Args:
            task:      Task description.
            context:   Additional context (e.g., prior results).
            max_steps: Maximum tool-calling iterations.

        Returns:
            Final text response from the LLM.
        """
        messages = [
            SystemMessage(content=self._system),
            HumanMessage(content=f"Task: {task}\n\n{context}"),
        ]

        for step in range(max_steps):
            response: AIMessage = self._llm.invoke(messages)
            messages.append(response)

            if not response.tool_calls:
                logger.info(f"ExecutorAgent: task done after {step + 1} steps")
                return response.content

            # Execute all tool calls
            for tc in response.tool_calls:
                tool = self._tool_map.get(tc["name"])
                if tool:
                    try:
                        result = tool.invoke(tc["args"])
                        # Apply VFS side effects
                        self._apply_vfs_effects(result)
                    except Exception as e:
                        result = f'{{"error": "{e}"}}'
                else:
                    result = f'{{"error": "Unknown tool: {tc["name"]}"}}'

                messages.append(
                    ToolMessage(content=result, tool_call_id=tc["id"], name=tc["name"])
                )

        logger.warning("ExecutorAgent: max_steps reached")
        return "Task execution reached step limit."

    def _apply_vfs_effects(self, result: str) -> None:
        """Apply write_file / edit_file side-effects to self.vfs."""
        from backend.utils.parser import safe_json_loads
        from backend.utils.helpers import utc_now

        parsed = safe_json_loads(result) if isinstance(result, str) else None
        if not isinstance(parsed, dict):
            return

        action = parsed.get("action")
        if action == "write_file":
            self.vfs[parsed["path"]] = {
                "content": parsed["content"],
                "created_at": parsed.get("created_at", utc_now()),
                "updated_at": parsed.get("updated_at", utc_now()),
            }
        elif action == "edit_file":
            path = parsed["path"]
            mode = parsed.get("mode", "overwrite")
            new_content = parsed.get("content", "")
            old_text = parsed.get("old_text", "")
            existing = self.vfs.get(path, {})
            current = existing.get("content", "") if isinstance(existing, dict) else ""

            if mode == "append":
                updated = current + "\n" + new_content
            elif mode == "replace" and old_text:
                updated = current.replace(old_text, new_content, 1)
            else:
                updated = new_content

            self.vfs[path] = {
                "content": updated,
                "created_at": existing.get("created_at", utc_now()) if isinstance(existing, dict) else utc_now(),
                "updated_at": parsed.get("updated_at", utc_now()),
            }

