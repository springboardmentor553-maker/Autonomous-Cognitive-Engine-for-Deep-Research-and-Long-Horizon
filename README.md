# Autonomous Cognitive Engine
### Deep Research & Long-Horizon Task Agent

> Powered by **LangGraph** · **Groq** (llama-3.1-8b-instant) · **Tavily Search**

---

## Overview

The Autonomous Cognitive Engine is a stateful AI agent capable of breaking complex, multi-step research tasks into structured TODO lists, executing them sequentially, storing intermediate results in a virtual file system, and synthesising a comprehensive final answer.

### Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    LangGraph Workflow                    │
│                                                         │
│  [START]                                                │
│     │                                                   │
│     ▼                                                   │
│  supervisor ──(has tool calls?)──► process_tool_calls   │
│     ▲                                      │            │
│     └──────────────────────────────────────┘            │
│     │                                                   │
│  (no tool calls → final answer)                         │
│     │                                                   │
│  set_final_output                                       │
│     │                                                   │
│  [END]                                                  │
└─────────────────────────────────────────────────────────┘
```

### Agent State

| Field | Type | Purpose |
|---|---|---|
| `messages` | `list` | Full conversation + tool call history |
| `todos` | `list[TodoItem]` | Structured task list |
| `files` | `dict[str, str]` | Virtual file system |
| `intermediate_results` | `list[str]` | Research snippets accumulated during execution |
| `current_task` | `int` | Pointer to the active TODO |
| `final_output` | `str` | Synthesised answer returned to the user |

---

## Project Structure

```
cognitive-engine/
├── agents/
│   └── supervisor_agent.py     # ReAct supervisor with all tools bound
├── tools/
│   ├── write_todos.py          # Milestone 1 – task planning tool
│   ├── file_system_tools.py    # Milestone 2 – VFS (ls/read/write/edit)
│   └── tavily_search.py        # Web search via Tavily
├── core/
│   ├── state.py                # AgentState TypedDict definition
│   ├── graph.py                # LangGraph workflow assembly
│   └── llm.py                  # Centralised Groq LLM factory
├── config/
│   └── settings.py             # Environment-variable based settings
├── main.py                     # Interactive REPL entry point
├── requirements.txt
├── .env.example
└── README.md
```

---
## Quick Start

### 1. Clone & install

```bash
git clone <your-repo>
cd cognitive-engine
pip install -r requirements.txt
```

### 2. Configure API keys

```bash
cp .env.example .env
# Open .env and fill in GROQ_API_KEY and TAVILY_API_KEY
```

#### Required keys

| Variable | Where to get it |
|---|---|
| `GROQ_API_KEY` | https://console.groq.com/keys |
| `TAVILY_API_KEY` | https://app.tavily.com/ |

#### Optional (LangSmith tracing)

```env
LANGCHAIN_TRACING_V2=true
LANGCHAIN_PROJECT=deep-cognitive-agent
LANGCHAIN_API_KEY=your_langsmith_key
```

### 3. Run

```bash
python main.py
```

You will see an interactive prompt:

```
You: Research the latest developments in quantum computing and write a structured report
```

The agent will:
1. Call `write_todos` to create a task plan
2. Use `tavily_search` to gather information
3. Use `write_file` / `edit_file` to store intermediate notes
4. Synthesise everything into a final answer

---

## Milestones Implemented

### Milestone 1 – Structured Task Planning
- `write_todos` tool accepts a JSON list of task descriptions
- Tasks are stored in `state["todos"]` with `status: "pending"`
- The supervisor always calls this tool first for complex requests

### Milestone 2 – Virtual File System
- `write_file` – create a new file in `state["files"]`
- `read_file` – retrieve stored content
- `edit_file` – update existing file content
- `ls` – list all files in the VFS

---

## Example Session

```
You: Compare the AI strategies of OpenAI, Anthropic, and Google DeepMind

[step 01] AIMessage → tool calls: ['write_todos']
[step 02] ToolMessage
[step 03] AIMessage → tool calls: ['tavily_search']
[step 04] ToolMessage
[step 05] AIMessage → tool calls: ['write_file']
...
[step 12] AIMessage

──────────────────────────────────────────────────────────────────────
FINAL OUTPUT
──────────────────────────────────────────────────────────────────────
# AI Strategy Comparison: OpenAI vs Anthropic vs Google DeepMind
...

TODO STATUS
──────────────────────────────────────────────────────────────────────
  ● [1] Research OpenAI strategy  (done)
  ● [2] Research Anthropic strategy  (done)
  ...

VIRTUAL FILE SYSTEM (4 file(s))
──────────────────────────────────────────────────────────────────────
  📄 openai_research.txt  (892 chars)
  📄 anthropic_research.txt  (743 chars)
  ...
```

---

## Configuration Reference

All settings live in `config/settings.py` and are read from environment variables.

| Variable | Default | Description |
|---|---|---|
| `GROQ_API_KEY` | *(required)* | Groq API key |
| `TAVILY_API_KEY` | *(required)* | Tavily search key |
| `GROQ_MODEL` | `llama-3.1-8b-instant` | Groq model to use |
| `LANGCHAIN_TRACING_V2` | `false` | Enable LangSmith tracing |
| `LANGCHAIN_PROJECT` | `deep-cognitive-agent` | LangSmith project name |
| `LANGCHAIN_API_KEY` | *(optional)* | LangSmith key |

---

## License

MIT
