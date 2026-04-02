"""
Milestone 4: Combined Planning + File Ops + Multi-Agent Collaboration
- Milestone 1: 5-step TODO planning with descriptive step names
- Milestone 2: Virtual file system with meaningful filenames
- Milestone 3: Supervisor → Researcher → Writer → Reviewer delegation
No UI — runs from terminal.
"""
import os
import sys
import time
import re
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

def _slug(text: str) -> str:
    """Pick the most topic-specific word from the task as a filename prefix.
    Skips all generic/filler/verb words and picks the longest remaining word
    — which is almost always the actual subject/topic."""
    stopwords = {
        # articles, prepositions, conjunctions
        "the","a","an","of","in","on","at","to","for","and","or","but","nor",
        "with","about","into","onto","from","by","as","its","it","this","that",
        # common task verbs
        "use","using","used","uses","explain","explains","explained",
        "describe","describes","described","analyze","analyzes","analysed",
        "analyse","create","creates","created","make","makes","made",
        "write","writes","written","give","gives","given","tell","tells",
        "show","shows","find","finds","found","get","gets","do","does","did",
        "build","builds","built","compare","compares","compared","define",
        "generate","report","research","study","review","overview","summary",
        "comprehensive","detailed","brief","simple","basic","advanced",
        # question words
        "how","what","why","when","which","who","where","is","are","was",
        "were","be","been","being","can","will","would","should","could",
        # generic nouns that add no topic info
        "impact","effect","role","use","usage","application","system",
        "model","models","approach","method","methods","process","topic",
        "information","data","results","output","plan","list","ideas",
        "fundamentals","basics","concepts","principles","introduction","guide",
    }
    words = re.sub(r"[^a-z0-9 ]", "", text.lower()).split()
    candidates = [w for w in words if w not in stopwords and len(w) > 3]
    if not candidates:
        candidates = [w for w in words if len(w) > 2]
    # Pick the longest candidate — longest word is usually the most specific topic noun
    best = max(candidates, key=len) if candidates else "research"
    return best[:20]

def fs_write(filename, content):
    (FS_DIR / filename).write_text(content, encoding="utf-8")
    size = len(content)
    print(f"    📄 Saved  → virtual_fs/{filename}  ({size} bytes)")

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
    print(f"\n  {'─'*62}")
    print(f"  🔀 DELEGATION  [{entry['time']}]")
    print(f"     FROM : {from_agent.upper()}")
    print(f"     TO   : {to_agent.upper()}")
    print(f"     TASK : {task}")
    print(f"  {'─'*62}")

# ── LLM factory ───────────────────────────────────────────────────────────────
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
    print("  🛡️  SUPERVISOR  — Orchestrating the full research workflow")
    print(f"{'═'*65}")

    slug = _slug(user_task)

    # Build meaningful filenames up front so every agent uses the same names
    filenames = {
        "phase1":  f"{slug}_background.txt",
        "phase2":  f"{slug}_findings.txt",
        "phase3":  f"{slug}_outlook.txt",
        "report":  f"{slug}_final_report.txt",
        "review":  f"{slug}_review.txt",
        "todos":   f"{slug}_todo_list.txt",
    }

    todos = [
        {"id": 1, "step": "Background Research",
         "description": f"Gather background context and foundational knowledge on '{user_task[:50]}'",
         "output": filenames["phase1"], "status": "pending"},
        {"id": 2, "step": "Key Findings",
         "description": f"Identify and document the most important findings and data points on '{user_task[:50]}'",
         "output": filenames["phase2"], "status": "pending"},
        {"id": 3, "step": "Trends & Outlook",
         "description": f"Analyse current trends, future directions, and strategic implications of '{user_task[:50]}'",
         "output": filenames["phase3"], "status": "pending"},
        {"id": 4, "step": "Report Writing",
         "description": "Synthesise all three research phases into a structured, professional final report",
         "output": filenames["report"], "status": "pending"},
        {"id": 5, "step": "Quality Review",
         "description": "Review the final report for accuracy, clarity, completeness, and logical structure",
         "output": filenames["review"], "status": "pending"},
    ]

    print("\n  📋 TODO LIST — 5 steps planned:")
    for t in todos:
        print(f"     [{t['id']}] {t['step']:<20}  →  {t['output']}")
        print(f"          {t['description']}")

    todo_text = "\n".join(
        f"[{t['id']}] {t['step']}\n    Task   : {t['description']}\n    Output : {t['output']}"
        for t in todos
    )
    fs_write(filenames["todos"], todo_text)

    # ── Delegate to Researcher
    log_delegation("Supervisor", "Researcher",
                   "Produce background, findings, and outlook files (3 phases in one pass)")
    researcher(user_task, filenames)
    todos[0]["status"] = todos[1]["status"] = todos[2]["status"] = "done"

    # ── Delegate to Writer
    log_delegation("Supervisor", "Writer",
                   "Read all research files and compose the final structured report")
    writer(user_task, filenames)
    todos[3]["status"] = "done"

    # ── Delegate to Reviewer
    log_delegation("Supervisor", "Reviewer",
                   "Evaluate the final report and return a quality verdict")
    reviewer(filenames)
    todos[4]["status"] = "done"

    # Update todo file to mark all done
    todo_done = "\n".join(
        f"[✓] {t['step']}\n    Task   : {t['description']}\n    Output : {t['output']}"
        for t in todos
    )
    fs_write(filenames["todos"], todo_done)

    return todos, filenames

# ── RESEARCHER ─────────────────────────────────────────────────────────────────
def researcher(user_task, filenames):
    print(f"\n{'═'*65}")
    print("  🔬 RESEARCHER  — Gathering information across 3 research phases")
    print("  Strategy : Single LLM call → split into 3 named output files")
    print(f"{'═'*65}")

    llm = make_llm(temperature=0.7, num_predict=350)

    response = llm.invoke([
        SystemMessage(content=(
            "You are a research agent. Write concise, factual, well-structured findings. "
            "Each phase must be clearly labelled and contain 3-4 sentences."
        )),
        HumanMessage(content=(
            f"Research topic: {user_task}\n\n"
            "Respond using exactly these three labelled sections:\n"
            "PHASE1: (background — what this topic is, its origin, and why it matters)\n"
            "PHASE2: (key findings — the most important facts, data points, and insights)\n"
            "PHASE3: (trends & outlook — current direction, future implications, emerging developments)"
        ))
    ])

    full = response.content

    # Parse phases from LLM output
    phases = {"PHASE1:": [], "PHASE2:": [], "PHASE3:": []}
    current_key = None
    for line in full.splitlines():
        matched = False
        for key in phases:
            if line.strip().upper().startswith(key):
                current_key = key
                phases[key].append(line)
                matched = True
                break
        if not matched and current_key:
            phases[current_key].append(line)

    mapping = {
        "PHASE1:": filenames["phase1"],
        "PHASE2:": filenames["phase2"],
        "PHASE3:": filenames["phase3"],
    }
    labels = {
        "PHASE1:": "Background Research",
        "PHASE2:": "Key Findings",
        "PHASE3:": "Trends & Outlook",
    }

    for key, fname in mapping.items():
        text = "\n".join(phases[key]).strip()
        if not text:
            text = f"{labels[key]} for: {user_task}\n(Content not separately parsed — see full research output.)"
        fs_write(fname, text)

    print("\n  ✓ Research complete — 3 files written to virtual_fs/")

# ── WRITER ─────────────────────────────────────────────────────────────────────
def writer(user_task, filenames):
    print(f"\n{'═'*65}")
    print("  ✍️  WRITER     — Reading research files and composing final report")
    print("  Strategy : Reads all 3 research files → produces 1 structured report")
    print(f"{'═'*65}")

    research_content = ""
    for key in ["phase1", "phase2", "phase3"]:
        fname = filenames[key]
        print(f"     📖 Reading : {fname}")
        research_content += f"\n[{fname}]\n" + fs_read(fname)[:220] + "\n"

    llm = make_llm(temperature=0.7, num_predict=400)
    response = llm.invoke([
        SystemMessage(content=(
            "You are a professional report writer. Produce a clear, well-structured report. "
            "Use concise language. Every section should add value."
        )),
        HumanMessage(content=(
            f"Task: {user_task}\n\n"
            f"Research material:\n{research_content}\n\n"
            "Write a professional report with:\n"
            "- Title\n"
            "- Introduction (2 sentences)\n"
            "- Key Findings (3 bullet points)\n"
            "- Analysis (2 sentences connecting the findings)\n"
            "- Conclusion (1 sentence with a forward-looking statement)"
        ))
    ])

    fs_write(filenames["report"], response.content)
    print(f"\n  ✓ Final report written → {filenames['report']}")

# ── REVIEWER ───────────────────────────────────────────────────────────────────
def reviewer(filenames):
    print(f"\n{'═'*65}")
    print("  🔍 REVIEWER   — Evaluating the final report for quality assurance")
    print("  Strategy : Reads final report → returns structured verdict")
    print(f"{'═'*65}")

    report_fname = filenames["report"]
    print(f"     📖 Reading : {report_fname}")
    report = fs_read(report_fname)[:500]

    llm = make_llm(temperature=0.3, num_predict=150)
    response = llm.invoke([
        SystemMessage(content=(
            "You are a quality reviewer. Assess the report on completeness, clarity, and structure. "
            "Give 3 short sentences of feedback. End with exactly: Verdict: Approved  or  Verdict: Needs Revision"
        )),
        HumanMessage(content=f"Review this report:\n\n{report}")
    ])

    fs_write(filenames["review"], response.content)
    print(f"\n  ✓ Review written → {filenames['review']}")

# ── MAIN ───────────────────────────────────────────────────────────────────────
def run(user_task: str):
    start = time.time()

    print(f"\n{'═'*65}")
    print("🚀 MILESTONE 4 — Multi-Agent Research Pipeline")
    print(f"  Model  : {OLLAMA_MODEL}")
    print(f"  Task   : {user_task[:60]}")
    print(f"  FS Dir : {FS_DIR.absolute()}")
    print(f"{'═'*65}")

    fs_clear()
    delegation_log.clear()

    todos, filenames = supervisor(user_task)

    elapsed = time.time() - start
    stats   = fs_stats()

    # ── Final Summary ──────────────────────────────────────────────────────────
    print(f"\n{'═'*65}")
    print(f"  ✅ PIPELINE COMPLETE  ({elapsed:.1f}s)")
    print(f"{'═'*65}")

    print("\n  📋 TODO STATUS — all 5 steps completed:")
    for t in todos:
        print(f"     [✓] Step {t['id']}: {t['step']:<20}  →  {t['output']}")

    print(f"\n  📁 VIRTUAL FILE SYSTEM  ({stats['count']} files · {stats['total_bytes']} bytes total):")
    for fname in stats["files"]:
        size = (FS_DIR / fname).stat().st_size
        print(f"     • {fname:<45} {size:>6} bytes")

    print(f"\n  🔀 DELEGATION LOG  ({len(delegation_log)} delegation events):")
    for d in delegation_log:
        print(f"     [{d['time']}]  {d['from']:<12} →  {d['to']:<12}  |  {d['task']}")

    print(f"\n  📄 FINAL REPORT  ({filenames['report']}):")
    report = fs_read(filenames["report"])
    preview = report[:700] + ("..." if len(report) > 700 else "")
    for line in preview.splitlines():
        print(f"     {line}")

    print(f"\n  📝 REVIEW  ({filenames['review']}):")
    for line in fs_read(filenames["review"]).splitlines():
        print(f"     {line}")

    print(f"\n{'═'*65}\n")

if __name__ == "__main__":
    task = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else \
        input("Enter research task: ").strip()

    if not task:
        task = "Impact of artificial intelligence on modern healthcare systems"

    run(task)
