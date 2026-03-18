"""
memory/memory_manager.py — Coordinates all memory subsystems.
"""

from __future__ import annotations

from memory.episodic_memory import EpisodicMemory
from memory.vector_store import VectorStore
from memory.working_memory import WorkingMemory
from utils.logger import get_logger

logger = get_logger(__name__)


class MemoryManager:
    """
    Unified interface to all memory layers:
    - working_memory : active message window
    - episodic       : persisted run summaries
    - vector_store   : semantic search over past content
    """

    def __init__(self):
        self.working = WorkingMemory()
        self.episodic = EpisodicMemory()
        self.vector = VectorStore()

    def after_run(self, request: str, output: str, todos: list[dict]) -> None:
        """Call at the end of each run to persist results."""
        run_id = self.episodic.save_run(request, output, todos)
        self.vector.add(run_id, output, metadata={"request": request[:100]})
        self.working.clear()
        logger.info(f"MemoryManager: run {run_id} persisted")

    def recall(self, query: str, top_k: int = 3) -> list[dict]:
        """Semantic search over past run outputs."""
        return self.vector.search(query, top_k=top_k)