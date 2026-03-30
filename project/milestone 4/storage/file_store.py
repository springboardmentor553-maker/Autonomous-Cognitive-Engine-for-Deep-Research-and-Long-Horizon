from __future__ import annotations

from pathlib import Path

from app.config import OUTPUT_DIR
from app.state import GraphState


def write_file(filename: str, content: str, state: GraphState) -> Path:
    path = OUTPUT_DIR / filename
    path.write_text(content, encoding="utf-8")
    state["files"][filename] = str(path)
    return path


def read_file(filename: str, state: GraphState) -> str:
    path = Path(state["files"][filename])
    return path.read_text(encoding="utf-8")


def edit_file(
    filename: str,
    target_text: str,
    replacement_text: str,
    state: GraphState,
) -> str:
    path = Path(state["files"][filename])
    original = path.read_text(encoding="utf-8")
    updated = original.replace(target_text, replacement_text)
    path.write_text(updated, encoding="utf-8")
    return f"Edited {filename}"


def list_files(state: GraphState) -> list[str]:
    return list(state["files"].keys())
