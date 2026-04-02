#  Autonomous Cognitive Engine for Deep Research and Long-Horizon Tasks

A production-ready autonomous AI agent built with **LangGraph** and **Groq**, capable of executing complex, multi-step research and analysis tasks through structured planning, context offloading, and multi-agent delegation.

---

##  Table of Contents

- [Project Overview](#project-overview)
- [Architecture](#architecture)
- [Project Structure](#project-structure)
- [Milestones](#milestones)
- [Setup & Installation](#setup--installation)
- [Configuration (.env)](#configuration-env)
- [Running the App](#running-the-app)
- [How It Works](#how-it-works)
- [Tools Reference](#tools-reference)
- [Sub-Agents Reference](#sub-agents-reference)
- [LangSmith Tracing](#langsmith-tracing)
- [Rate Limits & Troubleshooting](#rate-limits--troubleshooting)
- [Evaluation](#evaluation)

---

## Project Overview

This framework implements a "Deep Cognitive Task Agent" that mirrors how advanced AI systems handle long-horizon tasks:

1. **Plans** — breaks any request into 3–5 structured TODO items
2. **Executes** — delegates to specialist sub-agents or works directly
3. **Offloads context** — saves findings to a virtual file system to avoid context window limits
4. **Synthesizes** — reads all saved files and produces a final comprehensive report

**Tech Stack:**
- Python 3.11+
- LangGraph (stateful agent graph)
- LangChain (LLM integration & tools)
- Groq API (LLM inference — `llama-3.1-8b-instant` / `llama-3.3-70b-versatile`)
- Streamlit (web UI)
- LangSmith (tracing & observability)
- python-dotenv (environment management)

---

## Architecture

```
User Request
     │
     ▼
┌─────────────────────────────────────────────────┐
│              Supervisor Agent (LangGraph)        │
│                                                  │
│  ┌──────────┐  ┌──────────────┐  ┌───────────┐  │
│  │  PLAN    │  │   EXECUTE    │  │ SYNTHESIZE│  │
│  │write_todos  │ delegate /   │  │ read_file │  │
│  │ (3-5 TODOs) │ do directly  │  │ + report  │  │
│  └──────────┘  └──────────────┘  └───────────┘  │
│                       │                          │
│              ┌────────┼────────┐                 │
│              ▼        ▼        ▼                 │
│         web_search  summar.  code_analysis       │
│          _agent     _agent    _agent             │
│              │        │        │                 │
│              └────────┴────────┘                 │
│                       │                          │
│               Virtual File System                │
│          (dict in AgentState — scratchpad)       │
└─────────────────────────────────────────────────┘
     │
     ▼
Final Report (prose, ≥ 200 words)
```

**LangGraph node flow:**
```
START → [agent] → should_continue? → [tools] → [agent] → ... → END
```

---

## Project Structure

```
project/
│
├── main.py                  # Core LangGraph agent, run_agent(), rate-limit handling
├── app.py                   # Streamlit web UI
├── tools.py                 # ALL_TOOLS registry (planning + VFS + delegation)
├── state.py                 # AgentState, TodoItem, DelegationEntry TypedDicts
├── filesystem_tools.py      # Virtual File System tools (ls, read, write, edit, delete)
├── delegation_tool.py       # task() and list_agents() delegation tools
│
├── sub_agents/
│   ├── __init__.py          # Package marker
│   ├── registry.py          # SUB_AGENT_REGISTRY, run_sub_agent(), list_available_agents()
│   ├── web_search_agent.py  # Deep research specialist
│   ├── summarization_agent.py  # Summarization specialist
│   └── code_analysis_agent.py  # Technical analysis specialist
│
├── test_runner.py           # Milestone 2 evaluation (VFS tool usage)
├── test_runner_m3.py        # Milestone 3 evaluation (delegation)
│
├── .env                     # API keys (never commit this)
├── .gitignore               # Excludes .env, __pycache__, etc.
└── README.md                # This file
```

---

## Milestones

### ✅ Milestone 1 — Foundational Agent & Task Planning
- ReAct agent loop built with LangGraph `StateGraph`
- `write_todos` tool: decomposes any request into 3–5 structured TODOs
- `mark_todo_complete` tool: tracks completion status
- `AgentState` with `todos: list[TodoItem]` persisted across all steps
- **Success criteria:** Agent plans correctly for >80% of requests ✅

### ✅ Milestone 2 — Context Offloading via Virtual File System
- Five VFS tools: `ls`, `read_file`, `write_file`, `edit_file`, `delete_file`
- VFS stored in `AgentState.virtual_files` (dict of filename → content)
- Synced bidirectionally between state and the tools module on every step
- Message trimming (last 6 messages) prevents context window overflow
- **Success criteria:** VFS tools used correctly in >80% of multi-step scenarios ✅

### ✅ Milestone 3 — Sub-Agent Delegation
- `task(agent_name, sub_task, context)` tool delegates to specialist agents
- `list_agents()` tool lets supervisor discover available agents
- Three specialist sub-agents: `web_search_agent`, `summarization_agent`, `code_analysis_agent`
- Each sub-agent has its own `ChatGroq` instance (context isolation)
- `delegation_log` tracked in `AgentState` with agent name, task, duration, preview
- **Success criteria:** Delegation works correctly in >80% of relevant test cases ✅

### ✅ Milestone 4 — Full Integration & Use Case Application
- All components unified: planning + VFS + delegation in single LangGraph workflow
- System prompt guides all three phases: Plan → Execute (with delegation) → Synthesize
- Streamlit web UI with sidebar controls, chat history, live panels
- LangSmith tracing with explicit `LangChainTracer` callback
- Rate-limit handling: RPM throttle (3s gap), TPM backoff, daily quota detection
- **Success criteria:** End-to-end task completion >70% with good output quality ✅

---

## Setup & Installation

### Prerequisites
- Python 3.11 or later
- A [Groq API key](https://console.groq.com) (free)
- A [LangSmith API key](https://smith.langchain.com) (free, optional for tracing)

### Install dependencies

```bash
# Using uv (recommended — faster)
pip install uv
uv pip install langchain langchain-core langchain-groq langgraph langsmith streamlit python-dotenv

# Or using pip directly
pip install langchain langchain-core langchain-groq langgraph langsmith streamlit python-dotenv
```

---

## Configuration (.env)

Create a `.env` file in the project root:

```dotenv
# ── Required ──────────────────────────────────────
GROQ_API_KEY=gsk_your_groq_key_here

# ── Optional: LangSmith Tracing ───────────────────
LANGCHAIN_API_KEY=ls__your_langsmith_key_here
LANGCHAIN_TRACING_V2=true
LANGCHAIN_PROJECT=milestone4-deep-agent

# ── Optional: Model override ──────────────────────
# GROQ_MODEL=llama-3.1-8b-instant   ← default (recommended for free tier)
# GROQ_MODEL=llama-3.3-70b-versatile ← higher quality, uses more tokens
```

>  **Never commit `.env` to git.** It is already listed in `.gitignore`.

---

## Running the App

```bash
# Start the Streamlit web UI
streamlit run app.py

# Or run interactively in the terminal
python main.py
```

Open your browser at `http://localhost:8501`.

---

## How It Works

### Agent Workflow

```
1. PLAN    → call write_todos(3–5 tasks)
              Each task labelled: RESEARCH / ANALYZE / SYNTHESIZE / DRAFT / REVIEW

2. EXECUTE → for each TODO:
               RESEARCH / ANALYZE  → delegate to sub-agent via task()
               SYNTHESIZE / DRAFT  → agent writes directly
               → write_file() to save result
               → mark_todo_complete() to update status

3. SYNTHESIZE → read_file() for each saved file
                → write final prose report (≥200 words, no JSON)
```

### State Management

All state is carried in `AgentState` (LangGraph `TypedDict`):

| Field | Type | Purpose |
|---|---|---|
| `messages` | `list` (with `add_messages` reducer) | Full conversation + tool history |
| `todos` | `list[TodoItem]` | Task plan with status tracking |
| `virtual_files` | `dict[str, str]` | In-memory scratchpad (VFS) |
| `delegation_log` | `list[DelegationEntry]` | Audit trail of sub-agent calls |
| `final_output` | `str` | Extracted prose report |
| `write_todos_invoked` | `bool` | Planning phase guard |

---

## Tools Reference

### Planning Tools
| Tool | Description |
|---|---|
| `write_todos(tasks)` | Create 3–5 structured TODO items. Must be called first. |
| `get_todos()` | Retrieve current TODO list and statuses. |
| `mark_todo_complete(todo_id)` | Mark a TODO as completed by its ID. |

### Virtual File System Tools
| Tool | Description |
|---|---|
| `ls(directory)` | List all files in the virtual file system. |
| `read_file(filename)` | Read the full content of a saved file. |
| `write_file(filename, content)` | Create or overwrite a file. |
| `edit_file(filename, old_text, new_text)` | Find-and-replace within a file. |
| `delete_file(filename)` | Remove a file from the VFS. |

### Delegation Tools
| Tool | Description |
|---|---|
| `task(agent_name, sub_task, context)` | Delegate a sub-task to a specialist agent. |
| `list_agents()` | Discover available sub-agents and their capabilities. |

---

## Sub-Agents Reference

| Agent | Best For | Example Tasks |
|---|---|---|
| `web_search_agent` | Deep research, facts, trends, history | "Research the history of quantum computing" |
| `summarization_agent` | Condensing, comparing, key points | "Summarize pros/cons of microservices vs monolith" |
| `code_analysis_agent` | Code review, tech decisions, architecture | "Analyse Python vs Go for a backend API" |

Each sub-agent:
- Has its own `ChatGroq` instance (context isolation per the spec)
- Has a focused system prompt for its specialisation
- Uses the shared `_throttle()` from `main.py` to respect Groq rate limits
- Has its own retry logic for TPM/RPM errors

---

## LangSmith Tracing

When configured, every agent run is traced at [smith.langchain.com](https://smith.langchain.com) under the project `milestone4-deep-agent`.

**What is traced:**
- Every LLM call (input messages, output, token usage)
- Every tool call (name, input, output)
- Full execution graph with timing per node
- Run name matching the Streamlit run counter

**To enable:**
1. Get a free key at [smith.langchain.com](https://smith.langchain.com)
2. Add to `.env`: `LANGCHAIN_API_KEY=ls__...` and `LANGCHAIN_TRACING_V2=true`
3. Or paste the key in the sidebar and toggle **Enable Tracing** on

**Confirmation:** Look for this line in the terminal when a run starts:
```
🔗 LangSmith tracing ACTIVE → project: milestone4-deep-agent | run: streamlit-run-1
```

---

## Rate Limits & Troubleshooting

### Groq Free Tier Limits
| Limit | Value | Our mitigation |
|---|---|---|
| RPM (requests/min) | 30 | 3s minimum gap between all calls (`_throttle`) |
| TPM (tokens/min) | ~6k–14k depending on model | Trim to last 6 messages per call |
| Daily token quota | Varies | Detect wait >5min → raise error immediately |

### Common Errors

**`⏳ TPM hit — waiting Xs`** — Too many tokens per minute. The agent waits and retries automatically. If it keeps happening, switch to `llama-3.1-8b-instant` in the sidebar.

**`📅 Groq daily token quota exhausted`** — Free tier daily cap hit. Resets at midnight UTC. Switch to `llama-3.1-8b-instant` (uses ~8× fewer tokens) or wait for reset.

**`🔁 Agent hit the step limit`** — Raise the **Max agent steps** slider in the sidebar (default: 40, max: 60).

**`⚠️ LangSmith tracer init failed`** — LangSmith key missing or invalid. Tracing is skipped; the run still completes normally.

**`No module named 'sub_agents'`** — Make sure the `sub_agents/` folder is in the same directory as `main.py` and contains `__init__.py`.

### Recommended Settings (Free Tier)
- **Model:** `llama-3.1-8b-instant`
- **Max agent steps:** 40
- **Tasks per run:** 3 (use 5 only when needed)

---

## Evaluation

### Running Milestone 2 Tests (VFS usage)
```bash
python test_runner.py
```
Success criteria: ≥80% of 10 test cases pass all 6 checks.

### Running Milestone 3 Tests (Delegation)
```bash
python test_runner_m3.py
```
Success criteria: ≥80% of 10 test cases pass all 6 checks.

### Checks per test case
| Check | What it verifies |
|---|---|
| `write_todos_invoked` | Agent planned before executing |
| `write_file_called` | Context was offloaded to VFS |
| `read_file_called` | Context was retrieved before synthesis |
| `files_saved` | ≥2 files in VFS at end of run |
| `final_output_present` | Output is ≥100 chars of prose |
| `keyword_coverage_≥50%` | Final output contains expected keywords |

---

## .gitignore

Make sure your `.gitignore` contains at minimum:

```
.env
__pycache__/
*.pyc
*.pyo
.DS_Store
milestone2_eval_results.json
milestone3_eval_results.json
milestone4_output.json
```