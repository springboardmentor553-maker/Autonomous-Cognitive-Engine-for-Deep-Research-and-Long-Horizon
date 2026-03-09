

"""
memory/vector_store.py — Embedding-based semantic memory (stub for milestones 3+).

Currently provides an in-memory similarity search using simple TF-IDF-style
keyword overlap as a placeholder until a vector DB is integrated.
"""

from __future__ import annotations

import math
import re
from collections import Counter

from utils.logger import get_logger

logger = get_logger(__name__)


def _tokenise(text: str) -> list[str]:
    return re.findall(r"\w+", text.lower())


def _cosine(a: Counter, b: Counter) -> float:
    dot = sum(a[k] * b[k] for k in a if k in b)
    norm_a = math.sqrt(sum(v ** 2 for v in a.values()))
    norm_b = math.sqrt(sum(v ** 2 for v in b.values()))
    return dot / (norm_a * norm_b + 1e-9)


class VectorStore:
    """Simple keyword-overlap similarity store (no external dependencies)."""

    def __init__(self):
        self._docs: list[dict] = []  # [{id, content, metadata, vector}]

    def add(self, doc_id: str, content: str, metadata: dict | None = None) -> None:
        vector = Counter(_tokenise(content))
        self._docs.append({"id": doc_id, "content": content, "metadata": metadata or {}, "vector": vector})
        logger.debug(f"VectorStore.add: {doc_id}")

    def search(self, query: str, top_k: int = 3) -> list[dict]:
        q_vec = Counter(_tokenise(query))
        scored = [(doc, _cosine(q_vec, doc["vector"])) for doc in self._docs]
        scored.sort(key=lambda x: x[1], reverse=True)
        return [
            {"id": d["id"], "content": d["content"], "metadata": d["metadata"], "score": s}
            for d, s in scored[:top_k]
        ]

    def __len__(self) -> int:
        return len(self._docs)
