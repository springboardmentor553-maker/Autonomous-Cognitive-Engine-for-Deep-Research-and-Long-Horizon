from __future__ import annotations

from abc import ABC, abstractmethod

from app.state import GraphState


class BaseAgent(ABC):
    @abstractmethod
    def run(self, task_text: str, shared_state: GraphState) -> str:
        raise NotImplementedError
