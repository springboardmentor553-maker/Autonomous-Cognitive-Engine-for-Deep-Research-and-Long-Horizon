# Aether ResearchBot — AI-Powered Research Platform

A multi-agent AI research platform with **real-time web search**, **hallucination detection**, and a polished ChatGPT-style interface. A supervisor orchestrates specialist agents — Researcher, Writer, Reviewer, and Hallucination Guard — to produce grounded, source-backed research reports.

> **What makes this different?** Unlike basic LLM chatbots, Aether ResearchBot fetches **real web data** via Tavily, cross-references AI claims against sources, and flags unverified statements — turning vague LLM output into trustworthy research.

---

## ✨ Key Features

| Feature | Description |
|---------|-------------|
| 🌐 **Tavily Web Search** | Real-time web results injected into the research pipeline — no more relying on stale training data |
| 🔬 **Hallucination Guard** | Extracts factual claims from AI output and verifies them against web sources (✅ verified / ⚠️ unverified) |
| 🔍 **Ambiguity Detection** | Detects duplicate queries using Jaccard similarity and redirects to existing research |
| 🗑️ **Garbage Collector** | Background daemon that cleans up stale tasks and orphaned files automatically |
| 🛡️ **Multi-Agent Pipeline** | Supervisor → Researcher → Writer → Hallucination Guard → Reviewer |
| 💬 **ChatGPT-Style UI** | Conversation history sidebar, voice input, PDF export, real-time progress indicators |
| 👤 **User Authentication** | Sign up / sign in with SQLite-backed user accounts |
| 📄 **PDF Export** | Professional research reports with cover page, TOC, and branded formatting |

---

## Project Structure

```
Aether_ResearchBot/
│
├── app.py                    # ★ Flask backend — all API routes + workflow engine
├── database.py               # SQLAlchemy models (User, Conversation, Message, ResearchTask)
├── web_search.py             # Tavily API wrapper with retry logic + fallback
├── query_matcher.py          # Ambiguous query detector (Jaccard similarity)
├── garbage_collector.py      # Background daemon for stale task/file cleanup
├── hallucination_guard.py    # Claim extraction + verification against web sources
│
├── brains/                   # Agent logic
│   ├── researcher.py         # Researcher agent (web-augmented)
│   ├── writer.py             # Writer agent (report synthesis)
│   ├── reviewer.py           # Reviewer agent (quality verdict)
│   ├── supervisor.py         # Supervisor (routing logic)
│   └── filetools.py          # Virtual file system utilities
│
├── workflow/                 # LangGraph wiring
│   ├── flow.py               # Planning workflow
│   ├── multi_agent_flow.py   # Multi-agent state machine
│   ├── multi_agent.py        # Agent node definitions
│   ├── multi_agent_state.py  # Shared AgentState TypedDict
│   └── memory_state.py       # Memory/state helpers
│
├── ui/
│   ├── app.html              # Login / signup page
│   └── dashboard.html        # Main chat interface (ChatGPT-style)
│
├── admin/
│   └── admin_dashboard.html  # Admin panel (user management, task monitoring)
│
├── virtual_fs/               # Runtime output directory (auto-created)
├── instance/                 # SQLite database files
├── .env                      # API keys (not committed)
├── requirements.txt          # Python dependencies
└── README.md                 # This file
```

---

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure Environment

Create a `.env` file (or edit the existing one):

```env
# LLM Provider (NVIDIA NIM / OpenAI-compatible)
OPENAI_API_KEY=your-nvidia-or-openai-key
OPENAI_MODEL=meta/llama-3.1-8b-instruct

# Tavily Web Search (free tier: 1,000 searches/month)
# Get your key at https://tavily.com
TAVILY_API_KEY=your-tavily-api-key

# Optional: LangSmith tracing
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=your-langsmith-key
```

### 3. Run the Server

```bash
python app.py
```

Open http://localhost:5000 in your browser.

### 4. Default Login

| Role | Email | Password |
|------|-------|----------|
| Admin | nivi303.jk@gmail.com | admin123 |
| Demo User | demo@research.ai | demo123 |

---

## How It Works

### Research Pipeline (6 Steps)

```
User Query
    │
    ▼
┌─────────────────┐
│  Ambiguity Check │  ← Is this a duplicate query? If yes → redirect
└────────┬────────┘
         │ new query
         ▼
┌─────────────────┐
│   1. Supervisor  │  ← Plans the 5-step research architecture
└────────┬────────┘
         ▼
┌─────────────────┐
│  2. Researcher   │  ← Calls Tavily API for 5 web results,
│  + Web Search    │    then feeds sources to LLM as context
└────────┬────────┘
         ▼
┌─────────────────┐
│   3. Writer      │  ← Synthesizes research into a polished report
└────────┬────────┘
         ▼
┌─────────────────────┐
│ 4. Hallucination    │  ← Extracts claims, cross-references with
│    Guard            │    web sources, scores confidence (✅/⚠️)
└────────┬────────────┘
         ▼
┌─────────────────┐
│  5. Reviewer     │  ← Quality audit of the final report
└────────┬────────┘
         ▼
    Final Report
    + Source URLs
    + Confidence Score
    + Verification Summary
```

### Web Search Integration

Instead of asking the LLM *"tell me about X"* (which just recalls training data), the researcher:

1. **Calls Tavily Search API** → fetches 5 real, current web results (title, URL, content)
2. **Injects sources into the LLM prompt** → "Based on these web sources: {...}, write a report about: X"
3. **Stores source URLs** → saved alongside the message in the database
4. **Displays sources in the UI** → clickable links at the bottom of each response

### Hallucination Guard

After the Writer produces a report, the guard:

1. **Extracts claims** via LLM (numbers, dates, statistics, named entities)
2. **Verifies each claim** against Tavily source texts using exact + fuzzy matching
3. **Generates a summary**: ✅ Verified claims / ⚠️ Unverified claims
4. **Calculates confidence score** (e.g., 70% = 7 of 10 claims verified)

This is **non-blocking** — it adds transparency without preventing output.

### Ambiguity Detection

Before starting new research, the system:

1. Compares the query against all existing conversation titles + first messages
2. Uses **Jaccard similarity** on normalized tokens (stopwords removed)
3. If similarity ≥ 0.6, returns a redirect with a toast notification
4. User can click **"Open"** to see existing research or **"Research Anyway"** to force a new search

### Garbage Collector

A background daemon thread runs every 30 minutes:

- **Stale tasks**: Removes completed/errored tasks older than 1 hour from memory
- **Orphan files**: Deletes `virtual_fs/` files older than 24 hours
- **Manual trigger**: `GET /api/admin/gc` returns cleanup stats

---

## API Reference

### Auth
| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/auth/signup` | Create a new user account |
| POST | `/api/auth/login` | Login with email/password |

### Conversations
| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/conversations?email=...` | List user's conversations |
| POST | `/api/conversations` | Create a new conversation |
| GET | `/api/conversations/:id` | Get conversation with messages |
| DELETE | `/api/conversations/:id` | Delete a conversation |
| PUT | `/api/conversations/:id/rename` | Rename a conversation |
| GET | `/api/conversations/search?email=...&q=...` | Search conversations |

### Research
| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/research/submit` | Submit a research query (supports `force: true` to bypass dedup) |
| GET | `/api/research/status/:task_id` | Poll task progress + results |

### Admin
| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/tasks` | List all research tasks |
| GET | `/api/users` | List all users |
| GET | `/api/user/:id/tasks` | List tasks for a specific user |
| GET | `/api/admin/gc` | Trigger garbage collection + get stats |
| GET | `/api/files/:filename` | Download a virtual_fs file |

---

## LLM Call Budget Per Research Query

| Agent | Calls | Purpose |
|-------|-------|---------|
| Supervisor | 0 | Hardcoded routing (no LLM needed) |
| Web Search | 0 | Tavily API (separate from LLM) |
| Researcher | 1 | Background + findings (grounded in web sources) |
| Writer | 1 | Synthesis report |
| Hallucination Guard | 1 | Claim extraction |
| Reviewer | 1 | Quality audit |
| **Total** | **4** | + 1 Tavily API call |

---

## Database Schema

```
users
├── id, username, email, password_hash, user_type, created_at, last_login

conversations
├── id, user_id (FK→users), title, created_at, updated_at

messages
├── id, conversation_id (FK→conversations), role, content
├── search_sources (JSON — Tavily source URLs)
├── confidence_score (float — hallucination guard score)
├── created_at

research_tasks
├── id, task_id, user_id (FK→users), conversation_id (FK→conversations)
├── task_description, status, completed_steps, files_created
├── created_at, completed_at
```

---

## Configuration

### Environment Variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `OPENAI_API_KEY` | Yes | — | NVIDIA NIM or OpenAI API key |
| `OPENAI_MODEL` | No | `meta/llama-3.1-8b-instruct` | LLM model name |
| `TAVILY_API_KEY` | No | — | Tavily web search key (free tier: 1,000/month) |
| `LANGCHAIN_TRACING_V2` | No | `false` | Enable LangSmith tracing |
| `LANGCHAIN_API_KEY` | No | — | LangSmith API key |

### Without Tavily

If no `TAVILY_API_KEY` is set, the system degrades gracefully:
- Research uses LLM training data only (no web search)
- Hallucination Guard skips verification (returns report as-is)
- A warning is logged: `[WEB SEARCH] No TAVILY_API_KEY set`

---

## Common Errors

| Error | Cause | Fix |
|-------|-------|-----|
| `invalid_api_key` | Wrong NVIDIA/OpenAI key | Check `OPENAI_API_KEY` in `.env` |
| `[WEB SEARCH] tavily-python not installed` | Missing dependency | `pip install tavily-python` |
| `Working outside application context` | Flask-SQLAlchemy threading issue | Already handled — db calls wrapped in `app.app_context()` |
| `model not found` | Model not available on provider | Change `OPENAI_MODEL` in `.env` |
| LangSmith 422 payload error | Oversized tracing messages | Set `LANGCHAIN_TRACING_V2=false` |

---

## Legacy CLI Mode

The original CLI pipeline is still available for standalone testing:

```bash
python milestone4.py "Impact of AI on modern healthcare"
```

This runs the Supervisor → Researcher → Writer → Reviewer pipeline in the terminal without the web UI or database.

---

## License

Internal project — not open-sourced.
