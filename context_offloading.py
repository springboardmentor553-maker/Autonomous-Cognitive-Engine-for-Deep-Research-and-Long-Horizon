"""
Context Offloading Test — Ollama (llama3.2:1b)
Tests agent's ability to create 3 separate summary files (one per country)
then synthesise a comparative analysis from them.
"""
import os
import sys
import time
from pathlib import Path
from langchain_core.messages import HumanMessage
from workflow.flow import create_agent_executor, create_system_prompt
from brains.filetools import clear_virtual_fs, FILE_SYSTEM_DIR

# Disable LangSmith tracing
os.environ["LANGCHAIN_TRACING_V2"] = "false"

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL    = os.getenv("OLLAMA_MODEL",    "llama3.2:1b")

print(f"✓ Ollama URL : {OLLAMA_BASE_URL}")
print(f"✓ Model      : {OLLAMA_MODEL}")
print(f"✓ Storage    : {FILE_SYSTEM_DIR.absolute()}\n")


# ── Country culture source text ───────────────────────────────────────────────

COUNTRY_CULTURES = {
    "germany": """
    Germany has a rich cultural heritage deeply rooted in philosophy, music, and literature.
    The country is known for its precision engineering, punctuality, and strong work ethic.
    German culture values order, efficiency, and direct communication. Oktoberfest, Christmas
    markets, and beer gardens are integral to social life. Classical composers like Bach,
    Beethoven, and Wagner shaped Western music. Germans emphasize environmental consciousness,
    recycling extensively and investing heavily in renewable energy.
    """,
    "india": """
    India's culture is one of the world's oldest and most diverse, shaped by thousands of years
    of history, religion, and regional traditions. Hinduism, Buddhism, Jainism, and Sikhism
    originated here, creating a deeply spiritual society. Joint family systems remain common,
    with strong emphasis on respect for elders. Indian cuisine varies dramatically by region,
    using complex spice blends. Bollywood dominates entertainment, producing more films annually
    than any other country.
    """,
    "japan": """
    Japanese culture uniquely blends ancient traditions with cutting-edge modernity. The concept
    of 'wa' (harmony) underlies social interactions, emphasizing group cohesion over individualism.
    Shinto and Buddhist influences shape daily life, from shrine visits to seasonal festivals.
    Tea ceremony, ikebana, and calligraphy represent refined aesthetic principles. Japanese cuisine
    emphasizes seasonality and presentation. Manga and anime have become worldwide cultural exports.
    """
}


def check_files():
    if not FILE_SYSTEM_DIR.exists():
        return []
    return [f for f in FILE_SYSTEM_DIR.iterdir() if f.is_file()]


def run_context_offloading_test():
    print("=" * 80)
    print("CONTEXT OFFLOADING TEST — Country Culture Analysis")
    print("=" * 80)
    print("\nRequired output files:")
    print("  1. germany_culture.txt")
    print("  2. india_culture.txt")
    print("  3. japan_culture.txt")
    print("  4. final_comparison.txt")
    print("\n" + "=" * 80 + "\n")

    FILE_SYSTEM_DIR.mkdir(exist_ok=True)
    print("Clearing virtual file system...", end=" ")
    clear_virtual_fs()
    print("✓\n")

    print("Initializing agent...", end=" ")
    agent         = create_agent_executor()
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

YOUR TASK — Create EXACTLY 5 TODO steps:

Step 1: Summarize German culture (100-150 words) → save to "germany_culture.txt"
Step 2: Summarize Indian culture (100-150 words) → save to "india_culture.txt"
Step 3: Summarize Japanese culture (100-150 words) → save to "japan_culture.txt"
Step 4: Read all 3 culture files using read_file()
Step 5: Write comparative analysis → save to "final_comparison.txt"

CRITICAL:
- EXACTLY 5 steps
- Store SUMMARIES (100-150 words), NOT raw text
- Use exact filenames listed above
- Use write_file() and read_file() tools
"""

    print("Running agent (45-90 seconds)...")
    print("-" * 80 + "\n")

    try:
        result = agent.invoke(
            {"messages": [HumanMessage(content=f"{system_prompt}\n\n{task}")]},
            {"configurable": {"thread_id": "countries-culture"}, "recursion_limit": 50}
        )

        time.sleep(0.5)

        print("\n" + "=" * 80)
        print("EXECUTION COMPLETE")
        print("=" * 80)

        todos    = result.get("todos", [])
        messages = result.get("messages", [])
        current_files = check_files()

        # ── TODOs ──
        print(f"\n✓ TODOs created: {len(todos)}")
        for i, todo in enumerate(todos, 1):
            print(f"  {i}. {todo.get('description', 'N/A')[:85]}")

        # ── Tool calls ──
        tool_counts     = {}
        write_ops       = []
        read_ops        = []

        for msg in messages:
            if hasattr(msg, "tool_calls") and msg.tool_calls:
                for tc in msg.tool_calls:
                    name = tc.get("name", "unknown")
                    tool_counts[name] = tool_counts.get(name, 0) + 1
                    args = tc.get("args", {})
                    if name == "write_file":
                        write_ops.append({"filename": args.get("filename", "?"),
                                          "size": len(args.get("content", ""))})
                    elif name == "read_file":
                        read_ops.append(args.get("filename", "?"))

        print("\n✓ Tool invocations:")
        for tool, count in sorted(tool_counts.items()):
            print(f"  • {tool}: {count}x")

        if write_ops:
            print(f"\n✓ Write operations ({len(write_ops)}):")
            for op in write_ops:
                print(f"  • {op['filename']}  ({op['size']} bytes)")

        if read_ops:
            print(f"\n✓ Read operations ({len(read_ops)}):")
            for f in read_ops:
                print(f"  • {f}")

        # ── Files on disk ──
        print(f"\n✓ Files in virtual_fs/ ({len(current_files)} found):")
        for f in sorted(current_files, key=lambda x: x.name):
            print(f"  • {f.name}  ({f.stat().st_size} bytes)")

        # ── Validation ──
        files_present = {f.name for f in current_files}
        checks = {
            "write_todos called":        "write_todos"  in tool_counts,
            "write_file called ≥4x":     tool_counts.get("write_file", 0) >= 4,
            "read_file called ≥3x":      tool_counts.get("read_file",  0) >= 3,
            "germany_culture.txt exists": "germany_culture.txt"   in files_present,
            "india_culture.txt exists":   "india_culture.txt"     in files_present,
            "japan_culture.txt exists":   "japan_culture.txt"     in files_present,
            "final_comparison.txt exists":"final_comparison.txt"  in files_present,
            "exactly 5 TODOs":            len(todos) == 5,
        }

        print("\n" + "=" * 80)
        print("VALIDATION")
        print("=" * 80)
        for label, passed in checks.items():
            print(f"  {'✓' if passed else '✗'} {label}")

        all_passed = all(checks.values())
        print("\n" + "=" * 80)
        print("🎉 PASSED ✓" if all_passed else "⚠️  PARTIAL PASS")
        print("=" * 80)

        # ── File previews ──
        if current_files:
            print("\n" + "=" * 80)
            print("FILE PREVIEWS")
            print("=" * 80)
            order = ["germany_culture.txt", "india_culture.txt",
                     "japan_culture.txt", "final_comparison.txt"]
            for fname in order:
                matches = [f for f in current_files if f.name == fname]
                if matches:
                    text = matches[0].read_text(encoding="utf-8")
                    preview = text[:350] + ("..." if len(text) > 350 else "")
                    print(f"\n📄 {fname}")
                    print("-" * 60)
                    print(preview)

        return all_passed

    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback; traceback.print_exc()
        return False


if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("  CONTEXT OFFLOADING — Country Culture Analysis")
    print("=" * 80 + "\n")

    success = run_context_offloading_test()

    print(f"\n{'='*80}")
    print(f"Result  : {'✅ PASSED' if success else '❌ FAILED'}")
    print(f"Files at: {FILE_SYSTEM_DIR.absolute()}")
    print(f"{'='*80}\n")

    sys.exit(0 if success else 1)
