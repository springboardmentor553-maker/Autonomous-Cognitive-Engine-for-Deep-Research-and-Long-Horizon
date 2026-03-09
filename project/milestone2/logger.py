"""
Simple run logger for Milestone 2.
Writes a human-readable trace log for each task run.
"""

import os
import json
from datetime import datetime
from typing import Dict, Any, List


LOG_DIR = "logs"


def log_run(task_id: str, task: str, result: Dict[str, Any]):
    """
    Write a readable trace log for a single task run.

    Args:
        task_id: Short task identifier (e.g., 'task_01')
        task:    Full task description
        result:  Result dict from run_agent()
    """
    os.makedirs(LOG_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filepath   = os.path.join(LOG_DIR, f"{task_id}_{timestamp}.log")

    lines = [
        f"MILESTONE 2 RUN LOG",
        f"Task ID  : {task_id}",
        f"Time     : {timestamp}",
        f"{'='*60}",
        f"\nTASK DESCRIPTION:\n{task[:500]}{'...' if len(task) > 500 else ''}",
        f"\n{'='*60}",
        f"\nTODOs ({len(result.get('todos', []))}):",
    ]
    for i, todo in enumerate(result.get("todos", []), 1):
        lines.append(f"  {i}. [{todo.get('status','?')}] {todo.get('task','')}")

    lines.append(f"\nVIRTUAL FILE SYSTEM ({len(result.get('files', {}))} files):")
    for fname, content in result.get("files", {}).items():
        lines.append(f"\n  ── {fname} ──")
        lines.append(content[:400] + ("…" if len(content) > 400 else ""))

    lines.append(f"\n{'='*60}")
    lines.append(f"Message count: {len(result.get('messages', []))}")

    with open(filepath, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    return filepath
