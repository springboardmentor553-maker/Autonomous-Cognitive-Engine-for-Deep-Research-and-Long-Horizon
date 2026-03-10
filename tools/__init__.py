"""
tools/__init__.py — Aggregates all agent tools into a single list.
"""

from tools.filesystem import write_file, read_file, edit_file, ls
from tools.planning.write_todos import write_todos
from tools.research.web_search import web_search
from tools.research.summarize import summarize_text
from tools.research.extract_entities import extract_entities

ALL_TOOLS = [
    write_todos,
    write_file,
    read_file,
    edit_file,
    ls,
    web_search,
    summarize_text,
    extract_entities,
]

__all__ = ["ALL_TOOLS"]

