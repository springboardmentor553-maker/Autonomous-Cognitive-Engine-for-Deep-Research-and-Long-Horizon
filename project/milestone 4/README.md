# Milestone 4 Agentic System

This project is a stronger Milestone 4 implementation built around LangGraph, Groq, checkpointing, dynamic re-planning, file tools, sub-agent graphs, structured evaluation, a batch test harness, and a simple Streamlit UI.

## Main Improvements Over the Earlier Version

- conditional routing after execution
- MemorySaver checkpointing
- richer TODO statuses: `pending`, `in_progress`, `done`, `failed`
- dynamic re-planning when follow-up work is discovered
- `write_file`, `read_file`, `list_files`, and `edit_file` as real tools
- research sub-agent with optional live Tavily web search
- sub-agents implemented as their own LangGraph graphs
- structured evaluation output
- benchmark runner for repeated test runs
- Streamlit UI

## Updated Structure

```text
milestone 4/
├── .env
├── main.py
├── streamlit_app.py
├── README.md
├── requirements.txt
├── app/
│   ├── config.py
│   ├── evaluator.py
│   ├── executor.py
│   ├── models.py
│   ├── parsing.py
│   ├── planner.py
│   ├── state.py
│   ├── supervisor.py
│   └── synthesizer.py
├── agents/
│   ├── base.py
│   ├── summarizer.py
│   └── web_searcher.py
├── tools/
│   ├── search_tools.py
│   └── storage_tools.py
├── storage/
│   └── file_store.py
└── scripts/
    └── batch_runner.py
```

## Run CLI

```bash
pip install -r requirements.txt
python main.py
```

## Run UI

```bash
streamlit run streamlit_app.py
```

## Run Benchmark

```bash
python scripts/batch_runner.py
```

## Environment

```env
GROQ_API_KEY=your_groq_api_key_here
GROQ_MODEL=llama-3.3-70b-versatile
TAVILY_API_KEY=your_tavily_api_key_here
LANGSMITH_API_KEY=your_langsmith_or_langgraph_key_here
LANGSMITH_TRACING=true
LANGCHAIN_PROJECT=milestone-4-agent
THREAD_ID=milestone-4-demo
MAX_GRAPH_ITERATIONS=8
BENCHMARK_RUNS=10
```
