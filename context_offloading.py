"""
═══════════════════════════════════════════════════════════════════
CONTEXT OFFLOADING TEST: File System with LangSmith Tracing
═══════════════════════════════════════════════════════════════════
Tests agent's ability to create 3 SEPARATE summary files (one per paragraph)
═══════════════════════════════════════════════════════════════════
"""
import os
import sys
import time
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from workflow.flow import create_agent_executor, create_system_prompt
from brains.filetools import clear_virtual_fs, FILE_SYSTEM_DIR
import json

# Enable LangSmith tracing
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "context_offloading"

load_dotenv()

# Verify API keys
if not os.getenv("GROQ_API_KEY"):
    print("❌ ERROR: GROQ_API_KEY not set!")
    sys.exit(1)

if not os.getenv("LANGCHAIN_API_KEY"):
    print("⚠️  WARNING: LANGCHAIN_API_KEY not set - tracing disabled")

print("✓ Groq API key detected")
print("✓ LangSmith tracing ENABLED")
print(f"✓ Project: context_offloading")
print(f"✓ Virtual FS: {FILE_SYSTEM_DIR.absolute()}\n")


# ═══ TEST SCENARIO - 3 COUNTRY CULTURES ═══

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


def run_context_offloading_test():
    print("=" * 90)
    print("CONTEXT OFFLOADING TEST: Country Culture Analysis")
    print("=" * 90)
    print("\nObjective: Analyze 3 countries' cultures using file system")
    print("\nRequired Files:")
    print("  1. germany_culture.txt (summary of German culture)")
    print("  2. india_culture.txt (summary of Indian culture)")
    print("  3. japan_culture.txt (summary of Japanese culture)")
    print("  4. final_comparison.txt (comparative analysis of all 3)")
    print("\n" + "=" * 90 + "\n")
    
    # Ensure directory exists
    FILE_SYSTEM_DIR.mkdir(exist_ok=True)
    print(f"Virtual FS directory: {FILE_SYSTEM_DIR.absolute()}")
    
    # Clear previous test files
    print("Clearing virtual file system...", end=" ")
    clear_virtual_fs()
    print("✓")
    
    # Verify it's empty
    initial_files = check_files_now()
    print(f"Files before test: {len(initial_files)}\n")
    
    # Initialize agent
    print("Initializing agent...", end=" ")
    agent = create_agent_executor()
    system_prompt = create_system_prompt()
    print("✓\n")
    
    # Create task
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
    
    print("Running agent...")
    print("(This may take 45-90 seconds)")
    print("-" * 90 + "\n")
    
    try:
        result = agent.invoke(
            {"messages": [HumanMessage(content=f"{system_prompt}\n\n{task}")]},
            {"configurable": {"thread_id": "countries-culture"}, "recursion_limit": 50}
        )
        
        # Small delay to ensure file operations complete
        time.sleep(0.5)
        
        print("\n" + "=" * 90)
        print("EXECUTION COMPLETE")
        print("=" * 90)
        
        # Check files NOW
        current_files = check_files_now()
        
        # Analyze results
        todos = result.get("todos", [])
        messages = result.get("messages", [])
        
        print(f"\n✓ Generated {len(todos)} TODO items:")
        for i, todo in enumerate(todos, 1):
            print(f"  {i}. {todo.get('description', 'N/A')[:85]}")
        
        # Check tool invocations
        tool_calls = {}
        write_operations = []
        read_operations = []
        
        for msg in messages:
            if hasattr(msg, "tool_calls") and msg.tool_calls:
                for tc in msg.tool_calls:
                    tool_name = tc.get("name", "unknown")
                    tool_calls[tool_name] = tool_calls.get(tool_name, 0) + 1
                    
                    if tool_name == "write_file":
                        args = tc.get("args", {})
                        write_operations.append({
                            "filename": args.get("filename", "?"),
                            "size": len(args.get("content", ""))
                        })
                    elif tool_name == "read_file":
                        args = tc.get("args", {})
                        read_operations.append(args.get("filename", "?"))
        
        print(f"\n✓ Tool Invocations:")
        for tool, count in sorted(tool_calls.items()):
            print(f"  • {tool}: {count} time(s)")
        
        if write_operations:
            print(f"\n✓ Write Operations ({len(write_operations)} total):")
            for i, op in enumerate(write_operations, 1):
                print(f"  {i}. {op['filename']} ({op['size']} bytes)")
        
        if read_operations:
            print(f"\n✓ Read Operations ({len(read_operations)} total):")
            for i, fname in enumerate(read_operations, 1):
                print(f"  {i}. {fname}")
        
        # Check files in directory
        print(f"\n✓ Files in Virtual FS:")
        print(f"  Directory: {FILE_SYSTEM_DIR.absolute()}")
        
        if current_files:
            print(f"  Found {len(current_files)} file(s):")
            for f in sorted(current_files, key=lambda x: x.name):
                size = f.stat().st_size
                print(f"    • {f.name} ({size} bytes)")
        else:
            print(f"  ⚠ No files found!")
        
        # Validation
        print(f"\n" + "=" * 90)
        print("VALIDATION RESULTS")
        print("=" * 90)
        
        # Check for required files
        required_files = ["germany_culture.txt", "india_culture.txt", "japan_culture.txt"]
        files_present = {f.name for f in current_files}
        
        checks = {
            "write_todos_called": "write_todos" in tool_calls,
            "write_file_called_4x": "write_file" in tool_calls and tool_calls["write_file"] >= 4,
            "read_file_called_3x": "read_file" in tool_calls and tool_calls["read_file"] >= 3,
            "germany_created": "germany_culture.txt" in files_present,
            "india_created": "india_culture.txt" in files_present,
            "japan_created": "japan_culture.txt" in files_present,
            "final_comparison_created": "final_comparison.txt" in files_present,
            "todo_count_correct": len(todos) == 5
        }
        
        for check, passed in checks.items():
            icon = "✓" if passed else "✗"
            print(f"  {icon} {check.replace('_', ' ').title()}: {passed}")
        
        # Detailed file check
        print(f"\n  Required Country Files:")
        for req_file in required_files:
            exists = req_file in files_present
            icon = "✓" if exists else "✗"
            print(f"    {icon} {req_file}: {'Created' if exists else 'Missing'}")
        
        comparison_exists = "final_comparison.txt" in files_present
        icon = "✓" if comparison_exists else "✗"
        print(f"\n  Comparison File:")
        print(f"    {icon} final_comparison.txt: {'Created' if comparison_exists else 'Missing'}")
        
        all_passed = all(checks.values())
        
        print(f"\n" + "=" * 90)
        if all_passed:
            print("🎉 CONTEXT OFFLOADING TEST: PASSED ✓")
            print("   All 3 country culture files + comparison created successfully!")
        else:
            print("⚠  CONTEXT OFFLOADING TEST: PARTIAL PASS")
            print("   Not all required files were created")
            print(f"\n   Expected: germany_culture.txt, india_culture.txt, japan_culture.txt, final_comparison.txt")
            print(f"   Got: {len(current_files)} files - {', '.join(f.name for f in current_files)}")
        print("=" * 90)
        
        # Show file contents
        if current_files:
            print(f"\n" + "=" * 90)
            print("FILE CONTENTS")
            print("=" * 90)
            
            # Show country files first
            for country_file in ["germany_culture.txt", "india_culture.txt", "japan_culture.txt"]:
                matching = [f for f in current_files if f.name == country_file]
                if matching:
                    f = matching[0]
                    print(f"\n📄 {f.name}:")
                    print("-" * 90)
                    try:
                        with open(f, 'r', encoding='utf-8') as file:
                            content = file.read()
                            print(content[:400] + "..." if len(content) > 400 else content)
                    except Exception as e:
                        print(f"Error reading: {e}")
            
            # Show comparison file
            comparison_files = [f for f in current_files if "comparison" in f.name.lower()]
            if comparison_files:
                f = comparison_files[0]
                print(f"\n📄 {f.name} (Comparative Analysis):")
                print("-" * 90)
                try:
                    with open(f, 'r', encoding='utf-8') as file:
                        content = file.read()
                        print(content[:500] + "..." if len(content) > 500 else content)
                except Exception as e:
                    print(f"Error reading: {e}")
        
        print(f"\n" + "=" * 90)
        print("VIEW DETAILED TRACE IN LANGSMITH:")
        print("https://smith.langchain.com/ → Project: context_offloading")
        print("\nIn the trace, you should see:")
        print("  1. write_todos called with 5 steps")
        print("  2. write_file('germany_culture.txt', ...)")
        print("  3. write_file('india_culture.txt', ...)")
        print("  4. write_file('japan_culture.txt', ...)")
        print("  5. read_file('germany_culture.txt')")
        print("  6. read_file('india_culture.txt')")
        print("  7. read_file('japan_culture.txt')")
        print("  8. write_file('final_comparison.txt', ...)")
        print("=" * 90)
        
        return all_passed
        
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    print("\n" + "=" * 90)
    print("  CONTEXT OFFLOADING: Country Culture Analysis")
    print("  Test Case: Germany, India, Japan")
    print("=" * 90 + "\n")
    
    success = run_context_offloading_test()
    
    print(f"\n{'='*90}")
    print("TEST SUMMARY")
    print(f"{'='*90}")
    print(f"Status: {'✓ PASSED' if success else '✗ FAILED'}")
    print(f"Files should be in: {FILE_SYSTEM_DIR.absolute()}")
    print(f"{'='*90}\n")
    
    sys.exit(0 if success else 1)