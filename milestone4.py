"""
Milestone 4: Combined Planning + File Ops + Multi-Agent Collaboration
- Milestone 1: 5-step TODO planning
- Milestone 2: Virtual file system (read/write)
- Milestone 3: Supervisor → Researcher → Writer → Reviewer delegation
No UI — runs from terminal.
"""
import os
import sys
import time
from pathlib import Path
from datetime import datetime
from langchain_core.messages import HumanMessage

# Disable LangSmith tracing
os.environ["LANGCHAIN_TRACING_V2"] = "false"

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL    = os.getenv("OLLAMA_MODEL",    "llama3.2:1b")

# ── Virtual File System ────────────────────────────────────────────────────────
FS_DIR = Path("virtual_fs")
FS_DIR.mkdir(exist_ok=True)

def fs_write(filename, content):
    (FS_DIR / filename).write_text(content, encoding="utf-8")
    size = len(content)
    print(f"    📄 Saved: {filename} ({size} bytes)")

def fs_read(filename):
    p = FS_DIR / filename
    return p.read_text(encoding="utf-8") if p.exists() else ""

def fs_list():
    return sorted([f.name for f in FS_DIR.iterdir() if f.is_file()])

def fs_clear():
    for f in FS_DIR.iterdir():
        if f.is_file():
            f.unlink()

def fs_stats():
    files = fs_list()
    total = sum((FS_DIR / f).stat().st_size for f in files)
    return {"files": files, "count": len(files), "total_bytes": total}

# ── Delegation Tracker ─────────────────────────────────────────────────────────
delegation_log = []

def log_delegation(from_agent, to_agent, task):
    entry = {
        "time":  datetime.now().strftime("%H:%M:%S"),
        "from":  from_agent,
        "to":    to_agent,
        "task":  task
    }
    delegation_log.append(entry)
    print(f"\n  {'─'*60}")
    print(f"  🔀 DELEGATION  [{entry['time']}]")
    print(f"     FROM : {from_agent.upper()}")
    print(f"     TO   : {to_agent.upper()}")
    print(f"     TASK : {task}")
    print(f"  {'─'*60}")

# ── Agents ─────────────────────────────────────────────────────────────────────
from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage

def make_llm(temperature=0.7, num_predict=300):
    return ChatOllama(
        model=OLLAMA_MODEL,
        base_url=OLLAMA_BASE_URL,
        temperature=temperature,
        num_predict=num_predict
    )

# ── SUPERVISOR ─────────────────────────────────────────────────────────────────
def supervisor(user_task):
    print(f"\n{'═'*65}")
    print(f"  🛡️  SUPERVISOR  — Planning workflow")
    print(f"{'═'*65}")

    todos = [
        {"id": 1, "description": f"Research phase 1: background on '{user_task[:50]}'", "status": "pending"},
        {"id": 2, "description": f"Research phase 2: key findings on '{user_task[:50]}'", "status": "pending"},
        {"id": 3, "description": f"Research phase 3: trends/outlook on '{user_task[:50]}'", "status": "pending"},
        {"id": 4, "description": "Write comprehensive report from all research findings", "status": "pending"},
        {"id": 5, "description": "Review and validate the final report for quality",      "status": "pending"},
    ]

    print(f"\n  📋 TODO LIST — 5 steps created:")
    for t in todos:
        print(f"     [{t['id']}] {t['description']}")

    fs_write("todo_list.txt",
             "\n".join(f"[{t['id']}] {t['description']}" for t in todos))

    # Delegate all tasks
    log_delegation("Supervisor", "Researcher", "Research phases 1, 2 & 3")
    researcher(user_task)

    todos[0]["status"] = "done"
    todos[1]["status"] = "done"
    todos[2]["status"] = "done"

    log_delegation("Supervisor", "Writer", "Write final report")
    writer(user_task)
    todos[3]["status"] = "done"

    log_delegation("Supervisor", "Reviewer", "Review final report")
    reviewer()
    todos[4]["status"] = "done"

    # Mark todos complete
    fs_write("todo_list.txt",
             "\n".join(f"[✓] {t['description']}" for t in todos))

    return todos

# ── RESEARCHER ─────────────────────────────────────────────────────────────────
def researcher(user_task):
    print(f"\n{'═'*65}")
    print(f"  🔬 RESEARCHER  — Gathering information (1 LLM call → 3 files)")
    print(f"{'═'*65}")

    llm = make_llm(temperature=0.7, num_predict=300)

    response = llm.invoke([
        SystemMessage(content="You are a research agent. Write concise factual findings. Be brief."),
        HumanMessage(content=(
            f"Research: {user_task}\n"
            "Write 3 short sections:\n"
            "PHASE1: (background, 2 sentences)\n"
            "PHASE2: (key findings, 2 sentences)\n"
            "PHASE3: (trends/outlook, 2 sentences)"
        ))
    ])

    full = response.content
    phases = {"PHASE1:": "", "PHASE2:": "", "PHASE3:": ""}
    current_key = None
    for line in full.splitlines():
        for key in phases:
            if line.strip().startswith(key):
                current_key = key
                phases[key] += line + "\n"
                break
        else:
            if current_key:
                phases[current_key] += line + "\n"

    for i, key in enumerate(["PHASE1:", "PHASE2:", "PHASE3:"], 1):
        text = phases[key].strip() or f"Research phase {i} for: {user_task}"
        fs_write(f"research_step{i}.txt", text)

    print(f"  ✓ Research complete — 3 files written to virtual_fs/")

# ── WRITER ─────────────────────────────────────────────────────────────────────
def writer(user_task):
    print(f"\n{'═'*65}")
    print(f"  ✍️  WRITER     — Reading research files & writing report")
    print(f"{'═'*65}")

    research_files = sorted([f for f in fs_list() if f.startswith("research_")])
    research_content = ""
    for fname in research_files:
        print(f"     📖 Reading: {fname}")
        research_content += fs_read(fname)[:200] + "\n"

    llm = make_llm(temperature=0.7, num_predict=350)
    response = llm.invoke([
        SystemMessage(content="You are a professional writer. Write a concise structured report."),
        HumanMessage(content=(
            f"Task: {user_task}\n\nResearch:\n{research_content}\n\n"
            "Write: Title, Introduction (2 sentences), Findings (3 bullets), Conclusion (1 sentence)."
        ))
    ])

    fs_write("final_report.txt", response.content)
    print(f"  ✓ Report written to virtual_fs/final_report.txt")

# ── REVIEWER ───────────────────────────────────────────────────────────────────
def reviewer():
    print(f"\n{'═'*65}")
    print(f"  🔍 REVIEWER   — Reviewing final report")
    print(f"{'═'*65}")

    report = fs_read("final_report.txt")[:400]
    print(f"     📖 Reading: final_report.txt")

    llm = make_llm(temperature=0.3, num_predict=120)
    response = llm.invoke([
        SystemMessage(content="You are a quality reviewer. Give 2-sentence feedback. End with Verdict: Approved or Verdict: Needs Revision."),
        HumanMessage(content=f"Review:\n{report}")
    ])

    fs_write("review.txt", response.content)
    print(f"  ✓ Review written to virtual_fs/review.txt")

# ── MAIN ───────────────────────────────────────────────────────────────────────
def run(user_task: str):
    start = time.time()

    print(f"\n{'═'*65}")
    print(f"  🚀 MILESTONE 4 — Multi-Agent Research Pipeline")
    print(f"  Model : {OLLAMA_MODEL}")
    print(f"  Task  : {user_task[:60]}")
    print(f"{'═'*65}")

    # Clear previous run
    fs_clear()
    delegation_log.clear()

    # Run pipeline
    todos = supervisor(user_task)

    elapsed = time.time() - start

    # ── Final Summary ──────────────────────────────────────────────────────────
    stats = fs_stats()

    print(f"\n{'═'*65}")
    print(f"  ✅ COMPLETE  ({elapsed:.1f}s)")
    print(f"{'═'*65}")

    print(f"\n  📋 TODO STATUS:")
    for t in todos:
        print(f"     [✓] {t['description']}")

    print(f"\n  📁 VIRTUAL FILE SYSTEM  ({stats['count']} files, {stats['total_bytes']} bytes):")
    for fname in stats["files"]:
        size = (FS_DIR / fname).stat().st_size
        print(f"     • {fname:<30} {size} bytes")

    print(f"\n  🔀 DELEGATION LOG  ({len(delegation_log)} events):")
    for d in delegation_log:
        print(f"     [{d['time']}] {d['from']} → {d['to']}  |  {d['task']}")

    print(f"\n  📄 FINAL REPORT PREVIEW:")
    report = fs_read("final_report.txt")
    preview = report[:600] + ("..." if len(report) > 600 else "")
    for line in preview.splitlines():
        print(f"     {line}")

    print(f"\n  📝 REVIEW:")
    review = fs_read("review.txt")
    for line in review.splitlines():
        print(f"     {line}")

    print(f"\n{'═'*65}\n")

if __name__ == "__main__":
    task = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else \
        input("Enter research task: ").strip()

    if not task:
        task = "Impact of artificial intelligence on modern healthcare systems"

    run(task)