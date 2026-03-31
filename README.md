# Autonomous Cognitive Engine for Deep Research 🧠⚙️

An advanced, state-driven multi-agent AI system designed to autonomously plan, execute, and synthesize complex long-horizon research tasks. Built with LangGraph, LangChain, and Google's Gemini 2.5 Flash, this engine breaks down high-level objectives into specialized sub-tasks, delegates them to expert personas, and manages its own memory through a custom virtual file system.

Developed as part of an Infosys Springboard internship project.

## 🚀 Project Evolution & Milestones

This architecture was systematically developed across four distinct milestones, each solving a critical challenge in autonomous AI execution:

### Milestone 1: Context Offloading Strategy
Initial conceptualization to solve the "context window limitation" inherent in LLMs. The focus was on identifying how to extract and preserve high-value information (important points, summaries) from an ongoing process without bloating the active conversational memory, laying the groundwork for long-horizon tasks.

### Milestone 2: Virtual File System (VFS) Integration
Implementation of a robust in-memory storage mechanic to execute the context offloading strategy. 
* **Tools Created:** Custom `write_file`, `read_file`, `ls`, and `edit_file` tool integrations.
* **Capability:** The agent gained the ability to autonomously store intermediate findings into isolated files rather than keeping the full conversation history active, drastically improving token efficiency and operational focus.

### Milestone 3: Delegation & Specialized Agents
Transitioned from a single monolithic agent to a structured, multi-agent delegation system.
* **The Delegation Tool:** A routing mechanism was built to distribute tasks based on required expertise.
* **Specialized Sub-Agents:** Distinct LCEL (LangChain Expression Language) chains were introduced:
  * 🔍 **Researcher:** Gathers raw, highly technical data.
  * 📝 **Summarizer:** Condenses large data streams into concise formats.
  * ⚖️ **Comparator:** Analyzes multiple data points to find correlations and differences.
  * ✨ **Refiner:** Synthesizes final outputs into professional, formatted reports.

### Milestone 4: The Fully Autonomous Agent
The culmination of the project, from sub-task of Complex task (Milestone 1), merging the VFS context management (Milestone 2) with the specialized delegation system (Milestone 3) into a single, cohesive LangGraph `StateGraph`. 


## 📁 Repository Structure

```text
├── brains/                  # Specialist LLM Logic
│   ├── sub_agents.py        # LCEL chain definitions for specialized roles
│   └── workers.py           # Core prompts and Gemini model instantiation
├── graphs/                  # The LangGraph Engine
│   ├── execution_node.py    # Sub-agent execution, VFS tool calling, and state updates
│   ├── main_graph.py        # Graph compilation and edge wiring
│   ├── state.py             # TypedDict definitions for graph memory
│   └── supervisor_node.py   # Conditional routing logic
├── tools/                   # Agent Capabilities
│   ├── execution/           # VFS and file manipulation tools (Milestone 2)
│   ├── filesystem/          # In-memory storage mechanics
│   └── planning/            # Legacy planner tools
├── tests/                   # Execution entry points
│   ├── test_milestone2.py   # VFS context offloading evaluation
│   ├── test_milestone3.py   # Delegation and multi-agent testing
│   └── test_milestone4.py   # Full autonomous deep-research execution
├── outputs/                 # Final synthesized JSON and text reports
└── app.py                   # The Phase 1 Planner and engine launchpad
