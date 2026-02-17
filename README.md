# Deep Research Bot - Autonomous AI Agent

An autonomous AI agent with structured task planning using LangGraph and Google Gemini.

## Milestone 1: Task Planning ✅

### Features
- Structured task decomposition using `write_todos` tool
- LangGraph ReAct loop with state management
- Gemini 2.0 Flash Lite integration
- LangSmith tracing for all runs
- JSON-validated TODO lists

### Results
- Tool Call Rate: 100%
- Quality Score: 100%
- Action Verb Usage: 94%
- All 7 test cases passed

## Setup

### 1. Clone the repo
git clone https://github.com/yourusername/Deep_Researchbot.git
cd Deep_Researchbot

### 2. Create virtual environment
uv venv
.venv\Scripts\activate

### 3. Install dependencies
uv pip install -r requirements.txt

### 4. Set up API keys
cp .env.example .env
# Edit .env and add your actual API keys

### 5. Run demo
python demo.py

### 6. Run tests
python test_refined_milestone1.py

## API Keys Required
- Google AI: https://aistudio.google.com/app/apikey
- LangSmith: https://smith.langchain.com/ (optional, for tracing)

## Project Structure
Deep_Researchbot/
├── workflow/
│   ├── flow.py          # LangGraph workflow
│   └── memory_state.py  # State definitions
├── brains/
│   ├── mainagent.py     # write_todos tool
│   └── researcher.py    # Search tool
├── instructions/
│   └── mainagent.txt    # System prompt
├── demo.py              # Quick demo
├── test_refined_milestone1.py  # Test suite
├── requirements.txt     # Dependencies
├── .env.example         # API key template
└── .gitignore           # Prevents secrets upload
```

---

## 📂 Final GitHub Structure
```
Deep_Researchbot/
├── workflow/
│   ├── flow.py
│   └── memory_state.py
├── brains/
│   ├── mainagent.py
│   └── researcher.py
├── instructions/
│   └── mainagent.txt
├── demo.py
├── test_refined_milestone1.py
├── requirements.txt
├── .env.example          ✅ Safe to upload
├── .gitignore            ✅ Protects secrets
└── README.md             ✅ Documentation
