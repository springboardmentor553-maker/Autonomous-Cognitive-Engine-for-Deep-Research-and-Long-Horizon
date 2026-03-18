"""
tools/__init__.py — All agent tools.

Milestone 1 : write_todos
Milestone 2 : write_file, read_file, edit_file, ls
Milestone 3 : delegate_task
"""

from tools.filesystem import write_file, read_file, edit_file, ls
from tools.planning.write_todos import write_todos
from tools.research.web_search import web_search
from tools.research.summarize import summarize_text
from tools.research.extract_entities import extract_entities
from tools.delegation.delegate_task import delegate_task

ALL_TOOLS = [
    write_todos,       # M1 — task planning
    write_file,        # M2 — VFS write
    read_file,         # M2 — VFS read
    edit_file,         # M2 — VFS edit
    ls,                # M2 — VFS list
    web_search,        # research
    summarize_text,    # research
    extract_entities,  # research
    delegate_task,     # M3 — sub-agent delegation
]

__all__ = ["ALL_TOOLS"]