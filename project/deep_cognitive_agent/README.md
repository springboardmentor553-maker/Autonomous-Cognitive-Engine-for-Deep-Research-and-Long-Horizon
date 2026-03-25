# Autonomous Cognitive Engine for Deep Research & Long-Horizon Tasks

## Overview

This project implements an **Autonomous AI Agent System** capable of handling complex, multi-step research tasks using:

- **Planning** - Structured TODOs with dependencies
- **Memory** - Virtual File System (VFS) for context offloading
- **Execution Loop** - ReAct pattern (Reason → Act → Observe)
- **Multi-Agent Collaboration** - Supervisor + specialized sub-agents
- **Quality Evaluation** - Built-in quality scoring and refinement

Unlike basic LLM applications, this system behaves like a **human researcher** — it plans, executes, stores knowledge, and synthesizes results while maintaining architectural scalability.

---

## Key Features

### 1. Planning System (Milestone 1)

- Uses `write_todos` tool for structured task decomposition
- Enriched TODOs with `step_type`, `output_file`, and `depends_on`
- Ensures controlled, dependency-aware execution

```python
{
    "task": "Research AI ethics framework A",
    "status": "pending",
    "step_type": "research",
    "output_file": "ethics_framework_A_summary.txt",
    "depends_on": []
}
```

---

### 2. Virtual File System (Milestone 2)

- External memory storage (`state["files"]`)
- Prevents context overflow in long-horizon tasks
- Supports selective retrieval — only loads files needed for each step

**Available Operations:**

| Tool | Description |
|------|-------------|
| `write_file(state, filename, content)` | Create/overwrite a virtual file |
| `read_file(state, filename)` | Retrieve file content |
| `edit_file(state, filename, new_content)` | Modify existing file |
| `ls(state)` | List all virtual files |

---

### 3. Execution Engine (ReAct Loop)

```
Reason → Act → Observe → Update State → Repeat
```

- Selects task based on dependencies
- Decides appropriate action/tool
- Executes and stores result in VFS
- Maintains trace log for debugging

---

### 4. Multi-Agent System (Milestone 3)

**Architecture:**

```
User Request
     ↓
Supervisor Agent (planning & coordination)
     ↓
Task Delegation Tool
     ↓
Sub-Agents (specialized execution)
     ↓
Results → Supervisor → Synthesis → Final Output
```

**Available Sub-Agents:**

| Agent | Purpose |
|-------|---------|
| `ResearchAgent` | Information gathering and research |
| `SummarizerAgent` | Content summarization |
| `ComparatorAgent` | Comparative analysis |
| `UnifierAgent` | Synthesizing multiple sources |
| `RefinerAgent` | Quality improvement and refinement |

---

### 5. Quality Evaluation

- LLM-evaluated quality scoring (0-100)
- Configurable quality target threshold
- Detailed feedback for iterative refinement
- Quality gate pass/fail tracking

---

## Project Structure

```
deep_cognitive_agent/
├── agents/
│   ├── supervisor_agent.py          # Main coordinator agent
│   └── subagents/
│       ├── research_agent.py        # Research/information gathering
│       ├── summarizer_agent.py      # Content summarization
│       ├── comparator_agent.py      # Comparative analysis
│       ├── unifier_agent.py         # Multi-source synthesis
│       └── refiner_agent.py         # Quality refinement
│
├── graphs/
│   ├── state.py                     # AgentState TypedDict definition
│   ├── main_graph.py                # Milestone 2 graph builder
│   ├── main_graph_m3.py             # Milestone 3 graph builder
│   ├── supervisor_node.py           # Planning/supervisor node
│   ├── execution_node.py            # Milestone 2 execution node
│   ├── execution_node_m3.py         # Milestone 3 execution with delegation
│   └── synthesis_node.py            # Final output synthesis
│
├── tools/
│   ├── planning/
│   │   └── write_todos.py           # Structured TODO creation
│   ├── vfs/
│   │   ├── write_file.py            # VFS write operation
│   │   ├── read_file.py             # VFS read operation
│   │   ├── edit_file.py             # VFS edit operation
│   │   └── ls.py                    # VFS list files
│   ├── delegation/
│   │   └── task.py                  # Sub-agent task delegation
│   └── external/
│       └── tavily_search.py         # External search integration
│
├── registry/
│   └── subagent_registry.py         # Central sub-agent registry
│
├── prompts/
│   ├── supervisor_prompt.txt        # Supervisor agent prompt
│   ├── supervisor_m3_prompt.txt     # Milestone 3 supervisor prompt
│   ├── research_prompt.txt          # Research agent prompt
│   └── summarizer_prompt.txt        # Summarizer agent prompt
│
├── utils/
│   ├── helpers.py                   # Utility functions
│   └── logger.py                    # Logging utilities
│
├── tests/
│   ├── test_planning.py             # Planning system tests
│   ├── test_milestone.py            # General milestone tests
│   ├── test_milestone2.py           # VFS agent tests
│   └── test_milestone3.py           # Multi-agent tests
│
├── notebooks/
│   └── experiments.ipynb            # Jupyter notebook for experiments
│
├── outputs/                         # Generated output files
│   ├── milestone2_output.json
│   └── milestone3_output.json
│
├── app.py                           # Base application
├── app_milestone2.py                # Milestone 2 entry point
├── app_milestone3.py                # Milestone 3 entry point
├── requirements.txt                 # Python dependencies
├── .env.example                     # Environment template
└── README.md                        # This file
```

---

## Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/springboardmentor553-maker/Autonomous-Cognitive-Engine-for-Deep-Research-and-Long-Horizon.git
cd Autonomous-Cognitive-Engine-for-Deep-Research-and-Long-Horizon/project/deep_cognitive_agent
```

### 2. Create Virtual Environment

```bash
python -m venv .venv
```

### 3. Activate Environment

**Windows (PowerShell):**
```powershell
.\.venv\Scripts\Activate.ps1
```

**Windows (CMD):**
```cmd
.\.venv\Scripts\activate.bat
```

**Linux / macOS:**
```bash
source .venv/bin/activate
```

### 4. Install Dependencies

```bash
pip install -r requirements.txt
```

### 5. Setup Environment Variables

Copy the example environment file and add your API keys:

```bash
cp .env.example .env
```

Edit `.env` with your credentials:

```env
# Groq API Key (Required)
GROQ_API_KEY=gsk_your_groq_api_key_here

# LangSmith Tracing (Optional)
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=your_langsmith_api_key_here
LANGCHAIN_PROJECT=milestone_1_planning
```

**Get your Groq API key:** https://console.groq.com/keys

---

## Running the Project

### Run Milestone 2 (VFS Agent)

```bash
python app_milestone2.py --task "Analyze four AI ethics frameworks and synthesize a unified model"
```

### Run Milestone 3 (Multi-Agent)

```bash
python app_milestone3.py --task "Compare AI ethics frameworks and build unified model"
```

### Interactive Mode

```bash
python app_milestone2.py
# or
python app_milestone3.py
```

---

## Running Tests

### Run All Planning Tests

```bash
python -m pytest tests/test_planning.py -v
```

### Run Milestone 2 Tests

```bash
python -m pytest tests/test_milestone2.py -v
```

### Run Milestone 3 Tests

```bash
python -m pytest tests/test_milestone3.py -v
```

### Run All Tests

```bash
python -m pytest tests/ -v
```

---

## LangSmith Tracing

Enable LangSmith for detailed execution tracing:

1. Set `LANGCHAIN_TRACING_V2=true` in `.env`
2. Add your `LANGCHAIN_API_KEY`
3. Run the agent
4. View traces at: https://smith.langchain.com

**What you can observe:**
- Tool call sequences
- State updates
- Agent delegation flow
- Execution timing

---

## Architecture Flow

```
┌─────────────────────────────────────────────────────────────────┐
│                        User Input                               │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│              Planning Node (write_todos)                        │
│   Creates enriched TODOs with dependencies and output files     │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                    Execution Loop                               │
│   ┌─────────┐    ┌─────────┐    ┌─────────┐    ┌─────────┐     │
│   │ Select  │ →  │ Execute │ →  │ Store   │ →  │ Update  │     │
│   │  Task   │    │  Tool   │    │ in VFS  │    │  State  │     │
│   └─────────┘    └─────────┘    └─────────┘    └─────────┘     │
│                       ↓                                         │
│              ┌────────────────────┐                             │
│              │   Sub-Agents (M3)  │                             │
│              │ researcher/compare │                             │
│              │ unifier/refiner    │                             │
│              └────────────────────┘                             │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                    Synthesis Node                               │
│   Reads relevant files → Generates final structured output      │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                      Final Output                               │
└─────────────────────────────────────────────────────────────────┘
```

---

## Key Concepts

| Concept | Description |
|---------|-------------|
| **AgentState** | TypedDict storing todos, files, messages, progress, and quality metrics |
| **Enriched TODOs** | Structured tasks with step_type, output_file, and depends_on fields |
| **Virtual File System** | In-memory file storage for context offloading |
| **ReAct Pattern** | Reason → Act → Observe → Update execution loop |
| **Task Delegation** | Routing tasks to specialized sub-agents |
| **Context Offloading** | Storing intermediate results in VFS to reduce LLM memory load |
| **Trace Log** | Ordered record of all tool invocations for debugging |

---

## Architectural Principles

- **Selective Retrieval** — Only read files needed for each step
- **Meaningful File Names** — Derived from task content, not numbered
- **Clean Dependency Chain** — Each step builds on prior outputs
- **Memory Offloading** — Content stored in VFS, dropped from context
- **Trace Logging** — Every tool call recorded with purpose
- **No Duplication** — Write to file, return confirmation only
- **Scaling Stability** — Handles 3→20+ files without architecture change

---

## Common Mistakes to Avoid

| Mistake | Why It's Bad |
|---------|--------------|
| Skipping the planning tool | Agent loses structure and direction |
| Using plain text TODOs | No dependency tracking or step typing |
| Not storing data in VFS | Context overflow in long tasks |
| Reading all files blindly | Wastes context, slows execution |
| Over-delegation to sub-agents | Fragmented workflow, coordination overhead |

---

## Success Criteria

- Agent always plans first using `write_todos`
- Uses structured TODOs with dependencies
- Stores all intermediate results in VFS
- Uses selective file retrieval
- Maintains clean dependency chain
- Produces quality-evaluated final output

---

## Technologies Used

- **LangGraph** - Agent orchestration framework
- **LangChain** - LLM integration
- **Groq** - Fast LLM inference (Llama 3.1)
- **LangSmith** - Tracing and debugging
- **Python 3.10+** - Runtime

---

## License

This project is part of the Infosys Springboard Internship Program.

---

## Contributing

1. Create a feature branch from `main`
2. Make your changes
3. Run tests: `python -m pytest tests/ -v`
4. Commit with descriptive message
5. Push and create a Pull Request

---
