# 🧠 Autonomous Cognitive Engine for Deep Research and Long-Horizon Tasks

An advanced AI agent system built using **LangGraph** that autonomously plans, executes, and completes complex multi-step tasks using reasoning, memory, and structured workflows.

---

## 📌 Overview

This project implements a **Deep Cognitive Task Framework** designed to handle **long-horizon tasks** such as research, analysis, and multi-step problem solving.

Unlike traditional AI systems that generate one-shot responses, this engine:
- Breaks complex tasks into structured sub-tasks (TODOs)
- Executes them step-by-step using reasoning
- Stores intermediate results for context retention
- Synthesizes a final structured output

---

## 🏗️ System Architecture
```
User Input
↓
Supervisor Agent (Planner + Decision Maker)
↓
State Graph (LangGraph - Central Memory)
↓
Execution Loop (ReAct: Reason → Act → Observe)
↓
Tools / Sub-Agents (Search, File System, etc.)
↓
Virtual File System (Memory Storage)
↓
Final Synthesis
↓
Final Output (Report / Code / Analysis)

```
---

## 🔁 Workflow

1. **Input**: User provides a complex task  
2. **Planning**: Task is decomposed into TODO steps  
3. **Execution Loop**:
   - Select next task  
   - Reason → Decide action  
   - Execute using tools or sub-agents  
   - Store results  
4. **Memory Management**: Intermediate outputs saved in virtual file system  
5. **Synthesis**: Combine all results  
6. **Output**: Generate structured final response  

---

## 🧩 Key Components

### 🔹 Supervisor Agent
- Acts as the main controller
- Plans tasks and manages execution

### 🔹 State Graph (LangGraph)
- Maintains shared state across the workflow
- Tracks TODOs, memory, and execution status

### 🔹 ReAct Execution Loop
- Implements **Reason → Act → Observe**
- Enables iterative decision-making

### 🔹 Virtual File System
- Stores intermediate outputs (`step_1.txt`, etc.)
- Enables long-horizon task handling

### 🔹 Tools & APIs
- Task planning (`write_todos`)
- File operations (`read_file`, `write_file`)
- External tools (e.g., search APIs)

---

## 🛠️ Tech Stack

- **Python 3.11+**
- **LangGraph**
- **LangChain**
- **LLM API (Claude / similar)**
- **Tavily API (Search)**
- **Jupyter Notebook**
- **Git & GitHub**

---

## ⚙️ Installation & Setup

```bash
# Create virtual environment
uv venv

# Activate environment (Windows)
.venv\Scripts\activate

# Activate environment (Mac/Linux)
source .venv/bin/activate

# Install dependencies
uv pip install langgraph langchain python-dotenv

# ▶️ Running the Project
python -m project.deep_cognitive_agent.app
```
---
### 📊 Sample Execution Output
Task automatically decomposed into multiple steps
Each step executed sequentially
Intermediate results stored as files
Final structured report generated
Quality Score achieved: 9/10

### 🎯 Key Achievements

Built an end-to-end autonomous AI system
Successfully implemented multi-step task execution
Achieved stateful memory management
Generated structured and high-quality outputs
Demonstrated long-horizon reasoning capability

### 🚀 Future Improvements

Add sub-agent delegation system
Integrate real-time APIs for dynamic data
Build user interface (UI)
Optimize execution performance
Enhance evaluation metrics

### 🤝 Contribution
This project is developed as part of the Infosys Springboard AI Internship Program, focusing on real-world AI system design and implementation.

### 📌 Author

Sarvagya Porwal
AI / Data Science Intern
Infosys Springboard

### ⭐ Final Note

This project demonstrates how AI can move beyond single responses to autonomous, intelligent, multi-step problem solving systems, enabling practical real-world applications.