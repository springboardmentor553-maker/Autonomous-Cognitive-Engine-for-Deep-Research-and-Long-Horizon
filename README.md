# 🤖 Autonomous Cognitive Engine (ACE)
### Multi-Agent AI Framework for Deep Research and Long-Horizon Task Execution

<div align="center">

![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python)
![LangGraph](https://img.shields.io/badge/LangGraph-0.2.53-green?logo=langchain)
![Groq](https://img.shields.io/badge/Groq-llama--3.3--70b-orange)
![LangSmith](https://img.shields.io/badge/LangSmith-Tracing-purple)
![Tavily](https://img.shields.io/badge/Tavily-Search-red)
![License](https://img.shields.io/badge/License-MIT-yellow)
![Branch](https://img.shields.io/badge/Branch-Namitha-blue)

**Infosys Springboard Capstone Project**  
Built by **Namitha** | Branch: `Namitha`

</div>

---

## 📌 Table of Contents

- [Overview](#-overview)
- [Problem Statement](#-problem-statement)
- [Architecture](#-architecture)
- [Project Structure](#-project-structure)
- [Milestones](#-milestones)
- [Tech Stack](#-tech-stack)
- [Installation](#-installation)
- [Usage](#-usage)
- [Test Results](#-test-results)
- [Future Scope](#-future-scope)
- [Contributing](#-contributing)

---

## 🧠 Overview

The **Autonomous Cognitive Engine (ACE)** is a stateful multi-agent AI framework that can autonomously plan, research, delegate, synthesize, and evaluate complex long-horizon tasks — all without human intervention at each step.

ACE solves a fundamental limitation of single-agent systems: **one agent cannot reliably handle planning, research, summarization, comparison, synthesis, and quality evaluation all at once.** ACE solves this through specialization — a supervisor agent delegates work to specialist sub-agents, stores intermediate results in a virtual file system, and evaluates the final output quality.

### ✨ Key Capabilities

| Capability | Description |
|---|---|
| 🗂️ Task Planning | Converts any user request into a structured TODO list |
| 🤖 Sub-Agent Delegation | Routes specialized tasks to expert sub-agents |
| 📁 Virtual File System | Stores intermediate results in structured JSON files |
| 🔍 Web Research | Searches the web via Tavily for current information |
| 📝 Summarization | Condenses research into structured key points |
| 🔄 Synthesis | Combines all results into a comprehensive final report |
| ⭐ Evaluation | Automatically rates output quality 1-10 with feedback |

---

## ❗ Problem Statement

Single-agent LLM systems face these fundamental challenges:

- **Context window overflow** — cannot hold all research data in memory simultaneously
- **No specialization** — one agent attempts everything, becoming unreliable
- **No memory** — intermediate results are lost between reasoning steps
- **No quality measurement** — no automated way to verify output quality
- **Inefficient delegation** — repetitive tasks handled by the same general agent

**ACE solves all of these** through multi-agent architecture, virtual file system memory, and automated evaluation.

---

## 🏗️ Architecture

```
User Request
      │
      ▼
┌─────────────┐
│  PLANNER    │  ← write_todos(): breaks request into 5-8 structured tasks
└──────┬──────┘
       │
       ▼
┌──────────────────────────────────────────────────────┐
│                  SUPERVISOR (Executor)               │
│                                                      │
│  For each TODO task, supervisor decides:             │
│                                                      │
│  "search/find"    ──► delegate_task("web_searcher")  │
│  "summarize"      ──► delegate_task("summarizer")    │
│  "compare/write"  ──► handle directly                │
│                                                      │
│  After delegation ──► write_file() to store result   │
└──────┬───────────────────┬───────────────────────────┘
       │                   │
       ▼                   ▼
┌─────────────┐    ┌──────────────┐
│ SUMMARIZER  │    │ WEB_SEARCHER │
│  (LLM only) │    │ (Tavily only)│
│             │    │              │
│ /summaries/ │    │ /research/   │
└─────────────┘    └──────────────┘
       │
       ▼
┌─────────────┐
│ SYNTHESISER │  ← reads all VFS files → generates final report
└──────┬──────┘
       │
       ▼
┌─────────────┐
│  EVALUATOR  │  ← rates output 1-10 + 4 quality checks  (Milestone 4)
└──────┬──────┘
       │
       ▼
    OUTPUT
```

### Shared State (AgentState)

```python
state = {
    "messages":       [],   # conversation history
    "todos":          [],   # structured task plan
    "virtual_fs":     {},   # file system: {"/research/ieee.json": {...}}
    "delegation_log": [],   # every sub-agent call recorded
    "evaluation":     None  # quality score + feedback
}
```

---

## 📁 Project Structure

```
ACE/
│
├── agent/                          # Agent implementations
│   ├── planner_agent.py
│   ├── executor_agent.py
│   ├── researcher_agent.py
│   └── critic_agent.py
│
├── graph/                          # LangGraph graph definition
│   ├── build_graph.py              # Wires all nodes + routing
│   ├── nodes.py                    # All node functions
│   └── router.py                   # Conditional edge functions
│
├── state/                          # Shared state schema
│   └── agent_state.py              # AgentState TypedDict
│
├── sub_agents/                     # Milestone 3 — Sub-agents
│   ├── __init__.py
│   ├── summarization_agent.py      # RunnableLambda: text → summary
│   ├── web_search_agent.py         # RunnableLambda: query → findings
│   └── registry.py                 # sub_agents = {"summarizer": ..., "web_searcher": ...}
│
├── tools/                          # LangChain tools
│   ├── planning/
│   │   └── write_todos.py          # M1: create task plan
│   ├── filesystem/
│   │   ├── write_file.py           # M2: store results
│   │   ├── read_file.py            # M2: retrieve results
│   │   ├── edit_file.py            # M2: append/modify
│   │   └── ls.py                   # M2: list VFS files
│   ├── delegation/
│   │   └── delegate_task.py        # M3: route to sub-agent
│   ├── evaluation/
│   │   └── evaluate_output.py      # M4: rate 1-10
│   └── research/
│       ├── web_search.py
│       ├── summarize.py
│       └── extract_entities.py
│
├── prompts/                        # System prompts
│   ├── planner_prompt.txt          # 4-phase planning instructions
│   └── executor_prompt.txt         # Supervisor decision rules
│
├── memory/                         # Memory management
│   ├── episodic_memory.py
│   ├── memory_manager.py
│   ├── vector_store.py
│   └── working_memory.py
│
├── tests/                          # Unit tests
│   ├── test_graph.py
│   └── test_tools.py
│
├── app.py                          # Main entry point
├── config.py                       # Model + API configuration
│
├── test_milestone3_delegation.py   # M3: delegation test (14/14)
├── test_milestone3_improvement.py  # M3: independent decisions (8/8)
├── test_milestone4_evaluation.py   # M4: full pipeline (14/14)
└── test_milestone4_additional_prompts.py  # M4: 5 topics (10/10)
```

---

## 🚀 Milestones

### ✅ Milestone 1 — Task Planning (Weeks 1-2)

Converts any user request into a structured TODO list using the LLM.

```python
def write_todos(task_description):
    steps = llm.predict(
        f"Break this task into structured TODO steps: {task_description}"
    )
    todos = [{"task": step, "status": "pending"}
             for step in steps.split("\n")]
    return todos
```

**Result:** 5-8 ordered tasks with `id`, `description`, `status`, `result`, `delegated_to`

---

### ✅ Milestone 2 — Virtual File System & Memory (Weeks 3-4)

Stores and retrieves intermediate results in a structured virtual file system.

```
VFS Structure:
  /research/<topic>.json    ← raw web search results
  /summaries/<topic>.json   ← condensed key points  (SEPARATE from /research/)
  /compare/<topic>.json     ← analytical comparison
  /drafts/<report>.json     ← final report
```

**Tools:** `write_file` | `read_file` | `edit_file(append)` | `ls`

**Key feature:** Selective reading — synthesiser reads `/drafts/` first, skips raw `/research/` when higher-level files exist.

---

### ✅ Milestone 3 — Sub-Agent Delegation (Weeks 5-6)

Supervisor delegates specialized tasks to expert sub-agents.

```python
# Registry (mentor exact spec)
sub_agents = {
    "summarizer":   RunnableLambda(summarization_agent),
    "web_searcher": RunnableLambda(web_search_agent),
}

# Delegation tool (mentor exact spec)
def task(agent_name: str, input_data: str):
    if agent_name not in sub_agents:
        return "Agent not found."
    agent  = sub_agents[agent_name]
    result = agent.invoke(input_data)
    return result
```

| Sub-Agent | Purpose | Toolset | Trigger Keywords |
|---|---|---|---|
| `summarizer` | Condense text to key points | LLM only | summarize, condense, brief, digest |
| `web_searcher` | Search web for current info | Tavily only | search, find, look up, research |

**Scores:**
- Delegation test: **14/14 (100%)** | Delegation rate: **100%** (required >80%)
- Improvement test: **8/8 (100%)** | Decision accuracy: **100%** (10/10 tasks)

---

### ✅ Milestone 4 — Full Pipeline with Output Evaluation (Weeks 7-8)

Complete end-to-end pipeline with automated quality evaluation.

```
User Request
→ Planning      (write_todos)
→ Execution     (supervisor loop)
→ Delegation    (task tool → sub-agent)
→ File Storage  (write_file — /research/ and /summaries/ SEPARATE)
→ Retrieval     (read_file — explicit reads in trace)
→ Synthesis     (synthesize_results)
→ Evaluation    (evaluate_output — rate 1-10)
→ Output
```

```python
# Synthesis (mentor exact spec)
def synthesize_results():
    files         = ls()
    combined_data = ""
    for file in files:
        combined_data += read_file(file)
    return llm.predict(f"Generate final report from this data: {combined_data}")

# Evaluation (mentor exact spec)
def evaluate_output(report):
    score = llm.predict(f"Rate this report quality from 1 to 10: {report}")
    return score
```

**Evaluation checks:**
1. Did the system complete all tasks?
2. Did delegation happen?
3. Did memory (file storage/retrieval) work?
4. Did output make sense?

**Scores:**
- Pipeline test: **14/14 (100%)** | Output quality: **9/10 (excellent)**
- Additional prompts (5 topics): **10/10 (100%)** | Average score: **9.0/10**

---

## 🛠️ Tech Stack

| Component | Technology | Purpose |
|---|---|---|
| Agent Framework | LangGraph 0.2.53 | Stateful directed graph |
| LLM | Groq llama-3.3-70b-versatile | Fast inference, free tier |
| Web Search | Tavily API | Current information retrieval |
| Tracing | LangSmith | Full pipeline observability |
| Language | Python 3.11 | Core implementation |
| Package Manager | uv | Fast dependency management |
| Version Control | GitHub (Branch: Namitha) | Code management |

---

## ⚙️ Installation

### Prerequisites
- Python 3.11+
- [uv](https://github.com/astral-sh/uv) package manager
- Groq API key (free at [console.groq.com](https://console.groq.com))
- Tavily API key (free at [tavily.com](https://tavily.com))

### Step 1 — Clone the repository

```bash
git clone https://github.com/springboardmentor553-maker/Autonomous-Cognitive-Engine-for-Deep-Research-and-Long-Horizon.git
cd Autonomous-Cognitive-Engine-for-Deep-Research-and-Long-Horizon
git checkout Namitha
```

### Step 2 — Create virtual environment

```bash
uv venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate
```

### Step 3 — Install dependencies

```bash
uv pip install \
    pydantic==2.7.4 \
    pydantic-core==2.18.4 \
    langchain-core==0.3.17 \
    langchain==0.3.7 \
    langchain-groq==0.2.1 \
    langchain-community==0.3.7 \
    langgraph==0.2.53 \
    tavily-python \
    python-dotenv \
    colorlog \
    rich \
    langsmith
```

### Step 4 — Configure API keys

Create a `.env` file in the project root:

```env
# Required
GROQ_API_KEY=gsk_your_groq_key_here
TAVILY_API_KEY=tvly_your_tavily_key_here

# Optional (LangSmith tracing)
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=ls_your_langsmith_key_here
LANGCHAIN_PROJECT=autonomous-cognitive-engine

# Optional (model override)
MODEL_NAME=llama-3.3-70b-versatile
```

---

## 🎯 Usage

### Run the full agent

```bash
python app.py --request "Research IEEE and EU AI ethics frameworks, compare them, write a unified guide"
```

### Run milestone tests (no API keys required)

```bash
# Milestone 3 — Delegation test
python test_milestone3_delegation.py

# Milestone 3 — Independent decision making
python test_milestone3_improvement.py

# Milestone 4 — Full pipeline
python test_milestone4_evaluation.py

# Milestone 4 — Additional prompts (5 topics)
python test_milestone4_additional_prompts.py
```

### Example requests

```bash
# Research and compare
python app.py --request "Research NIST and ISO 27001 cybersecurity frameworks, compare them, write a security guide"

# Analysis
python app.py --request "Research FDA and WHO healthcare AI regulations, compare them, write a compliance guide"

# Topic research
python app.py --request "Research solar and wind energy policies, compare adoption rates, write an investment plan"
```

---

## 📊 Test Results

### All Milestones Summary

| Test | Score | Key Metric |
|---|---|---|
| M1 — Task Planning | ✅ Complete | Structured TODOs from any request |
| M2 — VFS Memory | ✅ Complete | Selective reading works |
| M3 — Delegation | **14/14 (100%)** | Delegation rate: 100% (req >80%) |
| M3 — Improvement | **8/8 (100%)** | Decision accuracy: 100% |
| M4 — Full Pipeline | **14/14 (100%)** | Output quality: 9/10 |
| M4 — 5 Topics | **10/10 (100%)** | Avg score: 9.0/10 |

### Additional Prompt Tests (Milestone 4)

| Topic | Tasks | Delegations | Files | Score |
|---|---|---|---|---|
| Climate Change (IPCC + Paris) | 7/7 | 4 | 6 | 9/10 |
| Cybersecurity (NIST + ISO 27001) | 7/7 | 4 | 6 | 9/10 |
| Healthcare AI (FDA + WHO) | 7/7 | 4 | 6 | 9/10 |
| Renewable Energy (Solar + Wind) | 7/7 | 4 | 6 | 9/10 |
| Space Exploration (NASA + SpaceX) | 7/7 | 4 | 6 | 9/10 |

---

## 🔭 Future Scope

### Milestone 5 — Human-in-the-Loop Checkpoints
- Supervisor pauses before critical decisions and requests human approval
- Human feedback loop improves agent decisions over time
- Configurable checkpoint triggers

### Milestone 6 — Real Vector Database Memory
- Replace in-memory VFS with **Chroma** or **Pinecone** vector store
- Persistent memory across sessions — agent remembers past research
- Semantic search — automatically finds similar past results

### Milestone 7 — Dynamic Agent Creation
- Supervisor creates new specialized sub-agents at runtime based on task needs
- Agent pool grows as new domains are encountered
- Self-improving agent ecosystem

### Milestone 8 — Production Deployment
- Multi-user support with **Redis** shared state
- **REST API** for integration with other systems
- Real-time monitoring dashboard
- Horizontal scaling with load balancer

### Enterprise Vision
ACE evolves into a **general-purpose autonomous research assistant** deployable across:
- Legal research and document analysis
- Medical literature review
- Financial report generation
- Code documentation and review
- Academic research synthesis

---

## 📈 LangSmith Tracing

ACE is fully instrumented for LangSmith observability. Every run traces:

- **delegation_log** — which sub-agents were called, with what input, and what they returned
- **EvaluationResult** — score, quality rating, and four check results
- **VFS state** — which files were created and when
- **Routing decisions** — which path the graph took at each conditional edge

Enable tracing by adding to `.env`:
```env
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=your_key
LANGCHAIN_PROJECT=autonomous-cognitive-engine
```

---

## 🤝 Contributing

1. Fork the repository
2. Create your feature branch: `git checkout -b feature/your-feature`
3. Commit your changes: `git commit -m "Add your feature"`
4. Push to the branch: `git push origin feature/your-feature`
5. Open a Pull Request

---

## 📄 License

This project is licensed under the MIT License.

---

