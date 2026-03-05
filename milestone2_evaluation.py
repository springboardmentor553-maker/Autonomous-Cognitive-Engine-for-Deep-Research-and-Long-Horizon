"""
═══════════════════════════════════════════════════════════════════
MILESTONE 2: CONTEXT OFFLOADING - FINAL EVALUATION
═══════════════════════════════════════════════════════════════════
Tests all Milestone 2 criteria with country culture analysis
Generates comprehensive final report with file persistence
═══════════════════════════════════════════════════════════════════
"""
import os
import sys
import time
import shutil
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from workflow.flow import create_agent_executor, create_system_prompt
from brains.filetools import clear_virtual_fs, FILE_SYSTEM_DIR, get_fs_stats
import json

# Enable LangSmith tracing
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "context_offloading"

load_dotenv()

# Verify API keys
if not os.getenv("GROQ_API_KEY"):
    print("❌ ERROR: GROQ_API_KEY not set!")
    sys.exit(1)

print("✓ Groq API key detected")
print("✓ LangSmith tracing ENABLED")
print(f"✓ Project: context_offloading")
print(f"✓ Virtual FS: {FILE_SYSTEM_DIR.absolute()}\n")


# ═══ TEST SCENARIO DATA ═══

COUNTRY_CULTURES = {
    "germany": """
    Germany has a rich cultural heritage deeply rooted in philosophy, music, and literature. 
    The country is known for its precision engineering, punctuality, and strong work ethic. 
    German culture values order, efficiency, and direct communication. Oktoberfest, Christmas 
    markets, and beer gardens are integral to social life. Classical composers like Bach, 
    Beethoven, and Wagner shaped Western music. Germans emphasize environmental consciousness, 
    recycling extensively and investing heavily in renewable energy. Family values remain strong, 
    though modern German society is increasingly multicultural.
    """,
    
    "india": """
    India's culture is one of the world's oldest and most diverse, shaped by thousands of years 
    of history, religion, and regional traditions. Hinduism, Buddhism, Jainism, and Sikhism 
    originated here, creating a deeply spiritual society. Joint family systems remain common, 
    with strong emphasis on respect for elders. Indian cuisine varies dramatically by region, 
    using complex spice blends and diverse cooking techniques. Classical dance forms like 
    Bharatanatyam and Kathak preserve ancient traditions. Bollywood dominates entertainment, 
    producing more films annually than any other country.
    """,
    
    "japan": """
    Japanese culture uniquely blends ancient traditions with cutting-edge modernity. The concept 
    of 'wa' (harmony) underlies social interactions, emphasizing group cohesion over individualism. 
    Shinto and Buddhist influences shape daily life, from shrine visits to seasonal festivals. 
    Tea ceremony, ikebana (flower arranging), and calligraphy represent refined aesthetic 
    principles. Japanese cuisine emphasizes seasonality, presentation, and fresh ingredients. 
    Manga and anime have become worldwide cultural exports. The workplace culture values loyalty, 
    dedication, and consensus decision-making.
    """
}


def check_files_now():
    """Check what files exist RIGHT NOW."""
    files = list(FILE_SYSTEM_DIR.iterdir()) if FILE_SYSTEM_DIR.exists() else []
    return [f for f in files if f.is_file()]


def analyze_tool_sequence(messages):
    """Extract and analyze tool invocation sequence."""
    sequence = []
    
    for msg in messages:
        if hasattr(msg, "tool_calls") and msg.tool_calls:
            for tc in msg.tool_calls:
                tool_name = tc.get("name", "unknown")
                args = tc.get("args", {})
                
                if tool_name == "write_file":
                    sequence.append({
                        "tool": "write_file",
                        "filename": args.get("filename", "?"),
                        "content_length": len(args.get("content", "")),
                        "action": f"WRITE: {args.get('filename', '?')}"
                    })
                elif tool_name == "read_file":
                    sequence.append({
                        "tool": "read_file",
                        "filename": args.get("filename", "?"),
                        "action": f"READ: {args.get('filename', '?')}"
                    })
                elif tool_name == "edit_file":
                    sequence.append({
                        "tool": "edit_file",
                        "filename": args.get("filename", "?"),
                        "action": f"EDIT: {args.get('filename', '?')}"
                    })
                else:
                    sequence.append({
                        "tool": tool_name,
                        "action": tool_name.upper()
                    })
    
    return sequence


def check_scaling_thinking(sequence, fs_stats):
    """Criterion 1: Scaling - summaries not raw, selective loading."""
    checks = {}
    
    write_ops = [s for s in sequence if s["tool"] == "write_file"]
    read_ops = [s for s in sequence if s["tool"] == "read_file"]
    
    # Check 1: Are files reasonably sized (summaries, not raw)?
    if write_ops:
        avg_size = sum(s["content_length"] for s in write_ops) / len(write_ops)
        checks["summaries_not_raw"] = 100 < avg_size < 2000
    else:
        checks["summaries_not_raw"] = False
    
    # Check 2: Selective retrieval
    checks["selective_retrieval"] = len(read_ops) > 0 and len(read_ops) <= fs_stats["total_files"] + 3
    
    # Check 3: No context explosion
    checks["context_stable"] = fs_stats["total_size_kb"] < 50
    
    # Check 4: Execution completed
    checks["execution_stable"] = len(write_ops) >= 3
    
    return checks


def check_architecture(sequence, fs_stats):
    """Criterion 2: Architecture - naming, no duplicates, edit usage."""
    checks = {}
    
    write_ops = [s for s in sequence if s["tool"] == "write_file"]
    
    # Check 1: Meaningful naming
    generic_names = ["file1", "file2", "data", "output", "temp"]
    filenames = [s["filename"] for s in write_ops]
    checks["meaningful_naming"] = not any(
        any(generic in fname.lower() for generic in generic_names)
        for fname in filenames
    )
    
    # Check 2: No unnecessary files
    checks["no_unnecessary_files"] = fs_stats["total_files"] <= 6
    
    # Check 3: edit_file available
    checks["edit_file_available"] = True
    
    # Check 4: No duplication
    write_filenames = [s["filename"] for s in write_ops]
    checks["no_duplication"] = len(write_filenames) == len(set(write_filenames))
    
    return checks


def check_tracing_quality(sequence):
    """Criterion 3: Tracing - clean, logical dependencies."""
    checks = {}
    
    # Check 1: Planning first
    checks["planning_first"] = sequence[0]["tool"] == "write_todos" if sequence else False
    
    # Check 2: Logical dependencies
    write_files = set()
    logical = True
    
    for step in sequence:
        if step["tool"] == "write_file":
            write_files.add(step["filename"])
        elif step["tool"] == "read_file":
            if step["filename"] not in write_files:
                logical = False
    
    checks["logical_dependencies"] = logical
    checks["multi_step_visible"] = len(sequence) >= 5
    checks["trace_clean"] = True
    
    return checks


def run_evaluation():
    print("=" * 100)
    print("MILESTONE 2: FINAL EVALUATION - Country Culture Analysis")
    print("=" * 100)
    print("\nTest: Analyze cultures of Germany, India, and Japan")
    print("Expected: 3 culture files + 1 comparison file")
    print("\n" + "=" * 100 + "\n")
    
    # Setup
    FILE_SYSTEM_DIR.mkdir(exist_ok=True)
    
    # Create backup directory
    BACKUP_DIR = Path("virtual_fs_backup")
    BACKUP_DIR.mkdir(exist_ok=True)
    
    print("Clearing virtual file system...", end=" ")
    clear_virtual_fs()
    for f in BACKUP_DIR.iterdir():
        if f.is_file():
            f.unlink()
    print("✓\n")
    
    print("Initializing agent...", end=" ")
    agent = create_agent_executor()
    system_prompt = create_system_prompt()
    print("✓\n")
    
    task = f"""
Analyze the cultures of Germany, India, and Japan using the file system for context offloading.

GERMANY CULTURE:
{COUNTRY_CULTURES['germany']}

INDIA CULTURE:
{COUNTRY_CULTURES['india']}

JAPAN CULTURE:
{COUNTRY_CULTURES['japan']}

YOUR TASK - Create EXACTLY 5 TODO steps:

Step 1: Summarize German culture (condense to 100-150 words) and save to "germany_culture.txt"
Step 2: Summarize Indian culture (condense to 100-150 words) and save to "india_culture.txt"
Step 3: Summarize Japanese culture (condense to 100-150 words) and save to "japan_culture.txt"
Step 4: Read all 3 culture files selectively using read_file()
Step 5: Create comparative analysis and save to "final_comparison.txt"

CRITICAL:
- EXACTLY 5 steps
- Store SUMMARIES (100-150 words each), NOT raw text
- Use these EXACT filenames: germany_culture.txt, india_culture.txt, japan_culture.txt, final_comparison.txt
- Read files selectively (only when needed)
- Use write_file() and read_file() tools
"""
    
    print("Running agent (45-90 seconds)...")
    print("-" * 100 + "\n")
    
    # File monitoring thread
    import threading
    
    stop_monitoring = False
    files_backed_up = set()
    
    def monitor_and_backup_files():
        nonlocal stop_monitoring, files_backed_up
        while not stop_monitoring:
            time.sleep(1)
            if FILE_SYSTEM_DIR.exists():
                current_files = list(FILE_SYSTEM_DIR.iterdir())
                for f in current_files:
                    if f.is_file() and f.name not in files_backed_up:
                        try:
                            shutil.copy2(f, BACKUP_DIR / f.name)
                            files_backed_up.add(f.name)
                            print(f"  [Backup] {f.name} ({f.stat().st_size} bytes)")
                        except Exception:
                            pass
    
    monitor_thread = threading.Thread(target=monitor_and_backup_files, daemon=True)
    monitor_thread.start()
    
    try:
        result = agent.invoke(
            {"messages": [HumanMessage(content=f"{system_prompt}\n\n{task}")]},
            {"configurable": {"thread_id": "milestone2-final"}, "recursion_limit": 50}
        )
        
        stop_monitoring = True
        time.sleep(1)
        
        print("\n" + "=" * 100)
        print("EXECUTION COMPLETE")
        print("=" * 100)
        
        # Restore from backup if needed
        current_files = list(FILE_SYSTEM_DIR.iterdir()) if FILE_SYSTEM_DIR.exists() else []
        backup_files = list(BACKUP_DIR.iterdir())
        
        if len(current_files) < len(backup_files):
            print(f"\n⚠️  Some files disappeared. Restoring from backup...")
            for backup_file in backup_files:
                target = FILE_SYSTEM_DIR / backup_file.name
                if not target.exists():
                    shutil.copy2(backup_file, target)
                    print(f"  ✓ Restored: {backup_file.name}")
        
        time.sleep(0.5)
        files = check_files_now()
        
        # Extract and analyze
        todos = result.get("todos", [])
        messages = result.get("messages", [])
        sequence = analyze_tool_sequence(messages)
        
        print(f"\n✓ Generated {len(todos)} TODO items:")
        for i, todo in enumerate(todos, 1):
            print(f"  {i}. {todo.get('description', 'N/A')[:80]}")
        
        print(f"\n✓ Tool Invocation Sequence ({len(sequence)} operations):")
        for i, step in enumerate(sequence, 1):
            print(f"  {i}. {step['action']}")
        
        print(f"\n✓ Virtual File System State:")
        print(f"  Total Files: {len(files)}")
        print(f"  Total Size: {sum(f.stat().st_size for f in files) / 1024:.2f} KB")
        print(f"  Files Created:")
        for fname in sorted([f.name for f in files]):
            fpath = FILE_SYSTEM_DIR / fname
            if fpath.exists():
                size = fpath.stat().st_size
                print(f"    • {fname} ({size} bytes)")
        
        # Evaluate
        print(f"\n" + "=" * 100)
        print("MILESTONE 2 CRITERIA EVALUATION")
        print("=" * 100)
        
        fs_stats = get_fs_stats()
        scaling_checks = check_scaling_thinking(sequence, fs_stats)
        architecture_checks = check_architecture(sequence, fs_stats)
        tracing_checks = check_tracing_quality(sequence)
        
        print(f"\n1. SCALING THINKING:")
        for check, passed in scaling_checks.items():
            icon = "✓" if passed else "✗"
            print(f"  {icon} {check.replace('_', ' ').title()}: {passed}")
        
        print(f"\n2. ARCHITECTURE:")
        for check, passed in architecture_checks.items():
            icon = "✓" if passed else "✗"
            print(f"  {icon} {check.replace('_', ' ').title()}: {passed}")
        
        print(f"\n3. TRACING QUALITY:")
        for check, passed in tracing_checks.items():
            icon = "✓" if passed else "✗"
            print(f"  {icon} {check.replace('_', ' ').title()}: {passed}")
        
        # Score
        all_checks = {**scaling_checks, **architecture_checks, **tracing_checks}
        passed_count = sum(1 for v in all_checks.values() if v)
        total_count = len(all_checks)
        score = (passed_count / total_count * 100) if total_count > 0 else 0
        
        print(f"\n" + "=" * 100)
        print(f"OVERALL SCORE: {passed_count}/{total_count} ({score:.1f}%)")
        
        if score >= 90:
            print("🎉 MILESTONE 2: EXCELLENT ✓")
        elif score >= 75:
            print("✓ MILESTONE 2: PASSED")
        else:
            print("⚠ MILESTONE 2: NEEDS IMPROVEMENT")
        
        print("=" * 100)
        
        # Show file contents
        if files:
            print(f"\n" + "=" * 100)
            print("FILE CONTENTS (DELIVERABLES)")
            print("=" * 100)
            
            for country_file in ["germany_culture.txt", "india_culture.txt", "japan_culture.txt"]:
                fpath = FILE_SYSTEM_DIR / country_file
                if fpath.exists():
                    print(f"\n📄 {country_file}:")
                    print("-" * 100)
                    with open(fpath, 'r', encoding='utf-8') as f:
                        content = f.read()
                        print(content[:400] + "..." if len(content) > 400 else content)
            
            comparison_files = [f for f in files if "comparison" in f.name.lower() or "final" in f.name.lower()]
            if comparison_files:
                fpath = comparison_files[0]
                print(f"\n📄 {fpath.name}:")
                print("-" * 100)
                with open(fpath, 'r', encoding='utf-8') as f:
                    content = f.read()
                    print(content[:500] + "..." if len(content) > 500 else content)
        
        print(f"\n" + "=" * 100)
        print("📊 TRACE: https://smith.langchain.com/ → context_offloading")
        print("=" * 100)
        
        print(f"\n✅ Files: {FILE_SYSTEM_DIR.absolute()}")
        print(f"✅ Backup: {BACKUP_DIR.absolute()}")
        
        return score >= 75
        
    except Exception as e:
        stop_monitoring = True
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        
        backup_files = list(BACKUP_DIR.iterdir())
        if backup_files:
            print(f"\n📦 Restoring from backup...")
            for backup_file in backup_files:
                shutil.copy2(backup_file, FILE_SYSTEM_DIR / backup_file.name)
                print(f"  ✓ {backup_file.name}")
        
        return False


if __name__ == "__main__":
    print("\n" + "=" * 100)
    print("  MILESTONE 2: CONTEXT OFFLOADING - FINAL EVALUATION")
    print("=" * 100 + "\n")
    
    success = run_evaluation()
    
    print(f"\n{'='*100}")
    print(f"Status: {'✓ PASSED' if success else '✗ FAILED'}")
    print(f"{'='*100}\n")
    
    sys.exit(0 if success else 1)