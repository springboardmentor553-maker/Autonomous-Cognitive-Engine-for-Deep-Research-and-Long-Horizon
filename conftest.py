"""
conftest.py — Adds the project root to sys.path so pytest can resolve
all local imports (tools, graph, state, etc.) without installation.
"""
import sys
from pathlib import Path

# Insert project root at front of path
sys.path.insert(0, str(Path(__file__).parent))