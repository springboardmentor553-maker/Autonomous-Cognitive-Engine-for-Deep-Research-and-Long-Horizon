

"""
memory/episodic_memory.py — Past task summaries stored on disk.

Each completed run is saved as a JSON entry so future runs can optionally
retrieve relevant past context.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path

from utils.logger import get_logger

logger = get_logger(__name__)

_MEMORY_DIR = Path.home() / ".ace_memory"


class EpisodicMemory:
    """Persistent log of past agent runs."""

    def __init__(self, storage_dir: Path | None = None):
        self._dir = storage_dir or _MEMORY_DIR
        self._dir.mkdir(parents=True, exist_ok=True)
        self._index_path = self._dir / "index.json"
        self._index: list[dict] = self._load_index()

    def _load_index(self) -> list[dict]:
        if self._index_path.exists():
            try:
                return json.loads(self._index_path.read_text())
            except Exception:
                return []
        return []

    def _save_index(self) -> None:
        self._index_path.write_text(json.dumps(self._index, indent=2))

    def save_run(self, request: str, output: str, todos: list[dict]) -> str:
        """Save a completed run. Returns the run ID."""
        run_id = f"run_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
        entry = {
            "id": run_id,
            "request": request[:200],
            "output_preview": output[:400],
            "task_count": len(todos),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        detail_path = self._dir / f"{run_id}.json"
        detail_path.write_text(
            json.dumps({"request": request, "output": output, "todos": todos}, indent=2)
        )
        self._index.append(entry)
        self._save_index()
        logger.info(f"EpisodicMemory: saved run {run_id}")
        return run_id

    def list_runs(self) -> list[dict]:
        return list(self._index)

    def get_run(self, run_id: str) -> dict | None:
        path = self._dir / f"{run_id}.json"
        if path.exists():
            return json.loads(path.read_text())
        return None
