# Autonomous Cognitive Engine for Deep Research

An advanced, multi-agent Long-Horizon reasoning framework built using LangGraph. This architecture is designed to handle extremely complex end-to-end research objectives by mimicking a corporate structure: a Supervisor agent dynamically plans and breaks down tasks, routing them concurrently to specialized sub-agents, while intelligently managing memory via a Virtual File System (VFS).

## 🚀 Key Features and Innovations

This project implements a highly structured, scalable autonomous pipeline fulfilling all progressive milestones required for advanced Deep Research:

* **Milestone 1: Dynamic Task Planning**
  * The Supervisor Agent parses hyper-complex, multi-domain objectives and dynamically generates a granular, step-by-step Execution Plan.
* **Milestone 2: Context Offloading via VFS**
  * Instead of infinitely appending bloated strings into the LLM context window (causing "lost-in-the-middle" hallucinations), agents dump raw data to a Virtual File System, keeping the context window pristine while allowing sequential agents to asynchronously read the exact files they need.
* **Milestone 3: Sub-Agent Delegation (Shared Registry Pattern)**
  * Uses a robust `Shared Delegation Layer` allowing the Supervisor to securely map tasks to explicitly defined sub-agent branches (e.g., `research_agent`, `analysis_agent`, `summarizer_agent`) via LangGraph node fan-outs.
* **🆕 Innovation: Autonomous VFS Garbage Collection (Context Pruning)**
  * A crucial leap past standard agents. It runs an asynchronous Garbage Collector (GC) node that evaluates VFS files against the core objective, aggressively compressing highly bloated files and completely purging hallucinated/tangential data to infinitely scale agent memory without hitting token caps.
* **Milestone 4: LLM-as-a-Judge Evaluation Suite**
  * The capstone pipeline. Dynamically generates 10 truly random, hyper-complex research scenarios on execution, streaming the pipeline end-to-end and grading the final output via an LLM judge matrix. Fulfills >70% complex automation success criteria.

## 📁 Repository Structure
* `/src/`: Contains your primary production LangGraph architecture, LLM configurations (`ChatGroq`), state definitions, and functional tool integrations (`Tavily`, etc).
* `test_milestone1.py` - Validates the Supervisor's dynamic planning decomposition trace.
* `test_milestone2.py` - Simulates the VFS context-offloading and memory-saving file pipelines.
* `test_milestone3.py` & `shared_delegation_layer.py` - Demonstrates sub-agent registration and routing.
* `test_milestone4_advanced.py` - Evaluates the LangGraph `Send` API mapping for parallel task execution and LangSmith telemetry hooks.
* `test_vfs_garbage_collector.py` - Evaluates the novel Context Pruning AI GC logic.
* **`test_milestone4_final.py`** - The ultimate evaluation script. Dynamically runs 10 hyper-complex test suites representing the final integrated pipeline and calculates the >70% LLM pass criteria!

## 💻 Getting Started

### Prerequisites
Make sure your Python virtual environment has the core dependencies installed:
```bash
pip install -r requirements.txt
# Ensure you install LangGraph: pip install langgraph langchain
```

### Running the Milestone Validation Scripts
To visually evaluate any milestone's specific trace, execute the corresponding test script. These files generate highly detailed, verbose console outputs to "show the work".

```powershell
# Run the final integrated evaluation loop handling 10 dynamic, complex scenarios:
python test_milestone4_final.py

# Run the Innovation VFS Garbage Collector simulation:
python test_vfs_garbage_collector.py

# Run the Sub-Agent Protocol tests:
python shared_delegation_layer.py
```

### Production Environment Variables
If running the API-driven files (`main.py` & `/src/`), you must furnish API keys securely in your `.env` root path.

*(Rename `.env.example` to `.env` if available)*
```env
# Required for LangGraph Tracing Telemetry
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=your-langsmith-key
LANGCHAIN_PROJECT=Advanced_Deep_Cognitive_Agent

# Required for the LLM Runner inside /src
GROQ_API_KEY=your-groq-key
TAVILY_API_KEY=your-tavily-key
```

## 🛠 Flow Architecture 
The standard end-to-end task flows as follows:

1. `Supervisor` receives objective -> Uses explicit Enums/Structured Logic to define an un-hallucinated `Executor Plan`.
2. Graph routes parallel/sequential branches to targeted `Sub-Agents`.
3. `Sub-Agents` (e.g., `ResearchAgent`) retrieve external insight and offload raw text sizes natively via `VFS WRITE`.
4. `Garbage Collector` trims `VFS READ` payloads by scrubbing irrelevant data.
5. `Synthesizer Agent` compiles the remaining pristine context facts into the `Final Execution Report`.

---
*Built as a highly flexible, multi-agent AI architecture capable of deterministic logic, scale, and cross-framework integrations.*
