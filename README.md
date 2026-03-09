# Autonomous Cognitive Engine

An autonomous planning and execution system that simulates a cognitive workflow for completing complex tasks.

The system plans tasks, executes them using tools, stores artifacts in memory, and evaluates performance.

---

## Milestone Progress

### Milestone 1 — Core Cognitive Engine
Implemented the core architecture of the autonomous system.

Features:
- Task planning module
- Executor for tool-based actions
- Search and delegation tools
- Evaluation framework for task performance

---

### Milestone 2 — Virtual File System

Milestone 2 introduces a **Virtual File System** that allows the agent to store and manage files during execution.

Available tools:

```
ls()          -> list stored files
write_file()  -> create and store file content
read_file()   -> read file contents
edit_file()   -> update existing files
```

This enables the agent to **persist intermediate results and manage artifacts across tasks.**

---

## Configuration

Create a `.env` file in the project root if API configuration is required.

Example:

```
OPENROUTER_API_KEY=sk-...
OPENROUTER_MODEL=mistralai/mistral-7b-instruct
```

---

## Running

Activate the virtual environment and start the backend:

```bash
python -m backend.main
```

The system will execute evaluation tasks and display the results.