# Milestone 2 — ReAct Agent with Virtual File System

Extends Milestone 1 (planning agent) with a full virtual file system and
intelligent multi-step execution.

---

## What's New in Milestone 2

| Feature | Milestone 1 | Milestone 2 |
|---------|------------|------------|
| Planning (`write_todos`) | ✅ | ✅ preserved |
| Virtual File System | ❌ | ✅ `write_file`, `read_file`, `ls`, `edit_file` |
| State `files` dict | ❌ | ✅ |
| Selective retrieval | ❌ | ✅ agent reasons which files to read |
| Dependency chains | ❌ | ✅ summarize → compare → unify → refine |
| `edit_file` refinement | ❌ | ✅ |

---

## Project Structure

```
milestone2/
├── app.py                        # Main agent + runner (Milestone 2)
├── main.py                       # Run all 7 tasks
├── tasks.py                      # Task definitions (7 tasks)
├── verify_agent.py               # Consistency + accuracy verification
├── models.py                     # Model availability checker
├── helpers.py                    # Shared utilities
├── logger.py                     # Run trace logger
├── requirements.txt
├── .env.example
├── graphs/
│   └── state.py                  # AgentState with todos + files
├── tools/
│   ├── planning/
│   │   └── write_todos.py        # Milestone 1 tool (preserved)
│   └── filesystem/
│       └── vfs_tools.py          # write_file, read_file, ls, edit_file
└── outputs/                      # JSON outputs per task
```

---

## Setup

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Set up environment variables
cp .env.example .env
# Edit .env and add your GOOGLE_API_KEY and LANGCHAIN_API_KEY

# 3. Run all 7 tasks
python main.py

# 4. Verify the agent
python verify_agent.py
```

---

## Tasks

| # | Label | Key Tools Demonstrated |
|---|-------|----------------------|
| 1 | Climate Change 3-doc summary | write_file × 3, read × 3, write final |
| 2 | 5 Renewable Energy policy docs | write × 5, ls, read × 5, write × 2 |
| 3 | Selective comparison (Germany vs India) | write × 5, ls, read × 2 only |
| 4 | Draft refinement cycle | write, read, edit_file |
| 5 | 4 AI Ethics frameworks → unified model | write × 4, ls, read × 4, write × 2, edit |
| 6 | Scaling test: 4-doc cybersecurity chain | full 10-step chain |
| 7 | Full eval pattern A→B | write → write → ls → read → read → write → read → edit |

---

## Tool Invocation Rules (from Milestone 2 spec)

1. **`write_todos` FIRST** — always plan before executing
2. **Store summaries, not raw text** — never write full documents to VFS
3. **Selective `read_file`** — only read files needed for the current step
4. **`ls` before comparison** — verify what's in VFS before reading
5. **`edit_file` for refinement** — update existing files, don't duplicate
6. **Minimal confirmation** — after write/edit, return brief confirmation only
