# 🧠 Autonomous Cognitive Engine

AI-powered research assistant built using Streamlit and an autonomous backend agent.

---

## 🚀 Features

- 🔎 Accepts natural language queries
- 📌 Structured output:
  - Research
  - Analysis
  - Summary
- 🧠 Intelligent section splitting
- 📊 Output metrics (words & characters)
- 📥 Export options:
  - TXT
  - PDF
- 💬 Chat-style interface with history
- ⚙️ Dual modes:
  - Summary (fast)
  - Detailed (full pipeline)

---

## 🏗️ Architecture

Frontend:
- Streamlit UI (chat interface, tabs, export)

Backend:
- Autonomous agent (`run_agent`)
- Handles:
  - Query processing
  - AI response generation
  - Structured formatting

---

## 🔄 Workflow

1. User enters query
2. Frontend sends input to backend
3. Backend (`run_agent`) processes request
4. Response is cleaned & split into sections
5. UI displays:
   - Research
   - Analysis
   - Summary


---

## ⚙️ Installation

```bash
pip install -r requirements.txt