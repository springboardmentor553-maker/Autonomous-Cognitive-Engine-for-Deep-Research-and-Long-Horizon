

"""
memory/working_memory.py — Short-context in-session memory.

Wraps the message list from AgentState with convenience methods.
"""

from __future__ import annotations

from langchain_core.messages import BaseMessage


class WorkingMemory:
    """Manages the active message window for a single agent run."""

    def __init__(self, max_messages: int = 40):
        self._messages: list[BaseMessage] = []
        self.max_messages = max_messages

    def add(self, message: BaseMessage) -> None:
        self._messages.append(message)
        # Trim oldest non-system messages if over limit
        if len(self._messages) > self.max_messages:
            self._messages = self._messages[-self.max_messages:]

    def get_all(self) -> list[BaseMessage]:
        return list(self._messages)

    def clear(self) -> None:
        self._messages.clear()

    def __len__(self) -> int:
        return len(self._messages)
