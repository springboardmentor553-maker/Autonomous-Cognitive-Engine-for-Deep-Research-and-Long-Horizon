# 🧠 Autonomous Cognitive Engine for Deep Research and Long-Horizon Planning
An AI-powered cognitive agent designed to perform structured planning, task decomposition, and long-horizon reasoning using LLMs and LangChain.
This project demonstrates a ReAct-style planning architecture where complex problems are converted into structured TODO pipelines before execution.
---
## 🚀 Features
- Structured planning using ReAct methodology
- Long-horizon task decomposition
- LangChain + Groq LLM integration
- Environment-based API configuration
- Modular agent architecture
- Output tracking and structured results
---
## 🧱 Project Structure
deep_cognitive_agent/
agents/        # Core agent logic  
tools/         # Custom tools (write_todos, planning, etc.)  
prompts/       # System prompts & templates  
graphs/        # LangGraph workflows  
registry/      # Agent registration  
outputs/       # Generated outputs  
utils/         # Helper utilities  
app.py         # Entry point  
---
## ⚙️ Installation
### 1. Clone the repository
git clone <repo-url>  
cd deep_cognitive_agent
### 2. Install dependencies
pip install -r requirements.txt
### 3. Setup environment variables
Create a `.env` file:
GROQ_API_KEY=your_api_key_here  
LANGCHAIN_TRACING_V2=true  
LANGCHAIN_PROJECT=milestone_1_planning  
---
## ▶️ Run the Agent
python app.py
Example task:
Build an AI chatbot architecture
The agent will:
1. Decompose the task
2. Generate structured TODO items
3. Save results inside `outputs/`
---
## 🧩 Tech Stack
- Python
- LangChain
- LangGraph
- Groq LLM
- dotenv
---
