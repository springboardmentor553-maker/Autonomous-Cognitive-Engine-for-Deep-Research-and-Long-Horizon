from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv


BASE_DIR = Path(__file__).resolve().parent.parent
ENV_FILE = BASE_DIR / ".env"
OUTPUT_DIR = BASE_DIR / "outputs"
REPORTS_DIR = BASE_DIR / "reports"
OUTPUT_DIR.mkdir(exist_ok=True)
REPORTS_DIR.mkdir(exist_ok=True)

load_dotenv(ENV_FILE)

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY", "")

LANGSMITH_API_KEY = os.getenv("LANGSMITH_API_KEY", os.getenv("LANGCHAIN_API_KEY", ""))
LANGSMITH_TRACING = os.getenv("LANGSMITH_TRACING", "true")
LANGCHAIN_PROJECT = os.getenv("LANGCHAIN_PROJECT", "milestone-4-agent")

DEFAULT_THREAD_ID = os.getenv("THREAD_ID", "milestone-4-demo")
MAX_GRAPH_ITERATIONS = int(os.getenv("MAX_GRAPH_ITERATIONS", "8"))
BENCHMARK_RUNS = int(os.getenv("BENCHMARK_RUNS", "10"))
