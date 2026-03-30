# Autonomous Cognitive Engine for Deep Research 🧠⚙️

An advanced, state-driven multi-agent AI system designed to autonomously plan, execute, and synthesize complex long-horizon research tasks. Built with LangGraph, LangChain, and Google's Gemini 2.5 Flash, this engine breaks down high-level objectives into specialized sub-tasks and manages them through a virtual file system and a robust supervisor-worker architecture.

## 🚀 System Architecture

The engine operates on a Two-Phase Execution Strategy to ensure high accuracy, prevent infinite loops, and gracefully handle API rate limits.

### Phase 1: Strategic Planning
A Lead Planning Agent receives the user's overarching prompt and generates a strict, JSON-formatted execution plan. It breaks the task into sequential `todos` and assigns each to a specialized persona.

### Phase 2: Multi-Agent Execution (LangGraph)
The engine consumes the `todos` and enters a cyclical state graph:
1. **The Supervisor Router:** Evaluates the graph's state and routes pending tasks to the appropriate specialized sub-agent.
2. **Specialized Sub-Agents:** Dedicated LCEL (LangChain Expression Language) chains optimized for specific cognitive tasks:
   * 🔍 **Researcher:** Gathers raw, highly technical data.
   * 📝 **Summarizer:** Condenses large data streams into manageable contexts.
   * ⚖️ **Comparator:** Analyzes multiple data points to find correlations and differences.
   * ✨ **Refiner:** Synthesizes the final output into a professional, formatted report.
3. **Context Offloading (VFS):** To prevent context window overflow, agents autonomously save their intermediate outputs to a Virtual File System using custom tool calling (`write_file`).
4. **State Injection:** Final outputs are injected back into the graph's `AIMessage` history for seamless extraction.

## 📊 Deep Telemetry & Tracing
This project features a "Gold Standard" LangSmith integration. It bypasses opaque lambda wrappers to provide deeply nested, hierarchical waterfall traces:
* **Tool Visibility:** Native LCEL tracking ensures all `write_file` tool payloads are visible in the trace.
* **Custom UI Tabs:** Graph state injections dynamically render custom tabs (e.g., `[REFINER_OUTPUT]`) inside the LangSmith dashboard for rapid debugging.
* **Dynamic Project Naming:** Test files dynamically inject `LANGCHAIN_PROJECT` variables to keep workspaces strictly organized by milestone.

## 📁 Repository Structure

```text
├── brains/                  # The LLM Logic
│   ├── sub_agents.py        # LCEL chain definitions for specialized roles
│   └── workers.py           # Core prompts and Gemini model instantiation
├── graphs/                  # The LangGraph Engine
│   ├── execution_node.py    # Sub-agent execution, tool calling, and state updates
│   ├── main_graph.py        # Graph compilation and edge wiring
│   ├── state.py             # TypedDict definitions for graph memory
│   └── supervisor_node.py   # Conditional routing logic
├── tools/                   # Agent Capabilities
│   ├── execution/           # VFS and file manipulation tools
│   ├── filesystem/          # In-memory storage mechanics
│   └── planning/            # Legacy Milestone 1/2 tools
├── tests/                   # Execution entry points
│   └── test_milestone4.py   # Full deep-research execution script
├── outputs/                 # Final synthesized JSON and text reports
└── app.py                   # The Phase 1 Planner and engine launchpad