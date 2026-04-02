# Deep ResearchBot — Developer Reference

A multi-agent AI research pipeline built with LangGraph and Ollama.
A supervisor orchestrates three specialist agents — Researcher, Writer, and Reviewer —
to take a plain-text topic and produce a structured research report, all running locally
with no cloud API keys required.

---

## Project Structure

```
Deep_Researchbot/
│
├── brains/                              # All agent logic
│   ├── researcher.py                    # Researcher agent (3 phases, 1 LLM call)
│   ├── writer.py                        # Writer agent (reads research → final report)
│   ├── reviewer.py                      # Reviewer agent (quality verdict)
│   ├── supervisor.py                    # Supervisor (hardcoded routing, no LLM needed)
│   └── filetools.py                     # Virtual file system read/write utilities
│
├── workflow/                            # LangGraph wiring
│   ├── flow.py                          # Planning workflow (write_todos + delegation)
│   ├── multi_agent_flow.py              # Main multi-agent state machine
│   ├── multi_agent.py                   # Agent node definitions
│   ├── multi_agent_state.py             # Shared AgentState TypedDict
│   └── memory_state.py                  # Memory/state persistence helpers
│
├── subagents/                           # Sub-agent registry for delegation
│   └── registry.py                      # Maps agent_type → handler
│
├── cognitive-engine-for-deep-research/  # Experimental cognitive engine module
├── instructions/                        # System prompt text files for agents
├── project/deep_cognitive_agent/        # Early prototype / reference code
│
├── milestone4.py                        # ★ Standalone CLI — full pipeline, no UI
├── milestone3.py                        # Multi-agent collaboration test suite
├── plan_validation.py                   # Milestone 2 test (file context offloading)
├── planning.py                          # Milestone 1 test (TODO planning)
├── context_offloading.py                # Context offloading experiment
├── main.py                              # Alternative entry point
│
├── requirements.txt                     # Python dependencies
├── .env                                 # API keys / env config (not committed)
├── README.md                            # This file
│
└── virtual_fs/                          # Runtime output directory (auto-created)
    ├── healthcare_background.txt
    ├── healthcare_findings.txt
    ├── healthcare_outlook.txt
    ├── healthcare_final_report.txt
    ├── healthcare_review.txt
    └── healthcare_todo_list.txt
```

---

## Requirements

```bash
pip install langchain-ollama langchain-core langgraph flask flask-cors flask-sqlalchemy python-dotenv
```

Ollama must be running locally:

```bash
ollama serve
ollama pull llama3.2:1b
```

---

## Agent Overview

### 🛡️ Supervisor
Routes work to the correct agent at each step using hardcoded logic — no LLM call needed,
which saves significant token overhead and latency.
Steps 1–3 go to Researcher, Step 4 to Writer, Step 5 to Reviewer.

### 🔬 Researcher
Makes a **single LLM call** that produces all three research phases at once
(Background, Key Findings, Trends & Outlook), then splits the output into three
separately named files. This reduces LLM calls from 3 to 1.

### ✍️ Writer
Reads all three research files from the virtual file system,
then composes a structured professional report with Title, Introduction,
Key Findings, Analysis, and Conclusion.

### 🔍 Reviewer
Reads the final report and returns 3 sentences of structured feedback
plus a verdict: `Approved` or `Needs Revision`.

---

## Virtual File System

All agents read and write through `virtual_fs/` — a plain directory on disk.
Files are named after the research topic so they are human-readable:

```
virtual_fs/
  impact_of_ai_background.txt
  impact_of_ai_findings.txt
  impact_of_ai_outlook.txt
  impact_of_ai_final_report.txt
  impact_of_ai_review.txt
  impact_of_ai_todo_list.txt
```

Helper functions in `filetools.py` (and duplicated in each agent for standalone use):

| Function | Description |
|---|---|
| `fs_write(filename, content)` | Write text to virtual_fs/ |
| `fs_read(filename)` | Read text from virtual_fs/ |
| `fs_list()` | List all files |
| `fs_clear()` | Delete all files (called before each run) |
| `fs_stats()` | File count and total bytes |

---

## Running the Pipeline

### Milestone 4 — Standalone CLI (recommended starting point)

Runs the full 5-step pipeline from the terminal with no Flask or UI dependencies.

```bash
# Pass the task as an argument
python milestone4.py "Impact of AI on modern healthcare"

# Or run interactively
python milestone4.py
```

**What you will see:**

```
══════════════════════════════════════════════════════════════
  🚀 MILESTONE 4 — Multi-Agent Research Pipeline
  Model  : llama3.2:1b
  Task   : Impact of AI on modern healthcare
══════════════════════════════════════════════════════════════

  📋 TODO LIST — 5 steps planned:
     [1] Background Research   →  impact_of_ai_background.txt
     [2] Key Findings          →  impact_of_ai_findings.txt
     [3] Trends & Outlook      →  impact_of_ai_outlook.txt
     [4] Report Writing        →  impact_of_ai_final_report.txt
     [5] Quality Review        →  impact_of_ai_review.txt

  ──────────────────────────────────────────────────────────
  🔀 DELEGATION  [06:12:01]
     FROM : SUPERVISOR
     TO   : RESEARCHER
     TASK : Produce background, findings, and outlook files
  ──────────────────────────────────────────────────────────

  ...

  ✅ PIPELINE COMPLETE  (22.4s)

  📁 VIRTUAL FILE SYSTEM  (6 files · 4821 bytes total):
     • impact_of_ai_background.txt           312 bytes
     • impact_of_ai_findings.txt             298 bytes
     ...

  🔀 DELEGATION LOG  (3 delegation events):
     [06:12:01]  Supervisor   →  Researcher   | Research phases 1, 2 & 3
     [06:12:14]  Supervisor   →  Writer       | Compose the final report
     [06:12:19]  Supervisor   →  Reviewer     | Quality verdict

  📄 FINAL REPORT ...
  📝 REVIEW ...
```

---

## Test Suites

### planning.py — Milestone 1
Tests whether the agent reliably creates a 5-step TODO plan with action verbs
and correct structure across 7 different task categories.

```bash
python planning.py
```

Checks: tool call rate, TODO count (4–6), quality score, action verb usage.

---

### plan_validation.py — Milestone 2
Tests whether the agent stores intermediate results in the virtual file system
using meaningful filenames and reads them back correctly before synthesis.
Runs 5 scenarios (cultures, frameworks, climate, languages, history).

```bash
python plan_validation.py
```

Pass criteria: > 80% of scenarios score ≥ 80%.

---

### milestone3.py — Milestone 3
Tests the full multi-agent collaboration pipeline:
Supervisor → Researcher → Writer → Reviewer.
Validates delegation events, file creation, and tool usage statistics.

```bash
python milestone3.py
```

---

## Configuration

Override defaults via environment variables before running:

```bash
set OLLAMA_MODEL=llama3.2:1b        # default
set OLLAMA_BASE_URL=http://localhost:11434   # default
```

### Model selection and RAM requirements

| Model | RAM needed | Speed | Quality |
|---|---|---|---|
| `llama3.2:1b` | ~1.3 GB | Fast | Good |
| `llama3.2:3b` | ~2.0 GB | Medium | Better |
| `qwen2.5:0.5b` | ~0.6 GB | Fastest | Basic |
| `llama3` | ~4.7 GB | Slow | Best |

Pull any model with:

```bash
ollama pull llama3.2:1b
```

---

## LLM Call Budget Per Run

| Agent | Calls | Max tokens out | Purpose |
|---|---|---|---|
| Supervisor | 0 | — | Hardcoded routing |
| Researcher | 1 | 350 | All 3 phases in one pass |
| Writer | 1 | 400 | Final report |
| Reviewer | 1 | 150 | Quality verdict |
| **Total** | **3** | **~900** | |

---

## Common Errors

| Error | Cause | Fix |
|---|---|---|
| `model requires more system memory` | Model too large for available RAM | Switch to `llama3.2:1b` or `qwen2.5:0.5b` |
| `model 'x' not found` | Model not pulled locally | Run `ollama pull <model>` |
| `ModuleNotFoundError: langchain_ollama` | Package not installed | Run `pip install langchain-ollama` |
| `StructuredTool not callable` | `@tool` decorated function called directly | Use `_impl` variants or call `.invoke()` |
| `Working outside application context` | Flask-SQLAlchemy in background thread | Wrap db calls in `with app.app_context():` |
| LangSmith 422 payload error | Tracing sending oversized messages | Set `LANGCHAIN_TRACING_V2=false` |

---

## Architecture Diagram

```
User Input
    │
    ▼
┌─────────────┐
│  Supervisor │  ← Hardcoded routing (no LLM call)
└──────┬──────┘
       │ delegates
   ┌───┴────────────────────────┐
   ▼                            │
┌──────────────┐                │
│  Researcher  │  1 LLM call    │
│  Phase 1–3   │  → 3 files     │
└──────────────┘                │
       │ files ready            │
       ▼                        │
┌──────────────┐                │
│    Writer    │  1 LLM call    │
│  Final report│  → 1 file      │
└──────────────┘                │
       │ report ready           │
       ▼                        │
┌──────────────┐                │
│   Reviewer   │  1 LLM call    │
│    Verdict   │  → 1 file      │
└──────────────┘                │
       │                        │
       └────────────────────────┘
              3 LLM calls total
```
