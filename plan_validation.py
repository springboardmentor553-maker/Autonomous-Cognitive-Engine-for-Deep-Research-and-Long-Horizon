"""
═══════════════════════════════════════════════════════════════════
PLAN VALIDATION: UNIFIED WORKFLOW (MILESTONE 1 + MILESTONE 2)
═══════════════════════════════════════════════════════════════════
Demonstrates complete workflow:
Milestone 1: Planning (write_todos creates 5-step plan)
Milestone 2: Execution (file operations execute the plan)

UPDATED: Tests MULTIPLE scenarios for comprehensive validation
Success Criteria: >80% pass rate across all scenarios
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

# Configure LangSmith tracing
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "milestone2-multi-prompt"

load_dotenv()

# Verify API key
if not os.getenv("GROQ_API_KEY"):
    print("❌ ERROR: GROQ_API_KEY not set!")
    sys.exit(1)

print("✓ Groq API key detected")
print("✓ LangSmith tracing ENABLED")
print(f"✓ Project: milestone2-multi-prompt")
print(f"✓ Storage: {FILE_SYSTEM_DIR.absolute()}\n")


# ─────────────────────────────────────────────────────────────────
# TEST SCENARIOS (5 Different Task Types)
# ─────────────────────────────────────────────────────────────────

TEST_SCENARIOS = [
    {
        "name": "Country Cultures",
        "task": """
Analyze the cultures of Germany, India, and Japan using file storage.

GERMANY CULTURE:
Germany has a rich cultural heritage deeply rooted in philosophy, music, and literature.
The country is known for its precision engineering, punctuality, and strong work ethic.
German culture values order, efficiency, and direct communication. Oktoberfest, Christmas
markets, and beer gardens are integral to social life. Classical composers like Bach,
Beethoven, and Wagner shaped Western music. Germans emphasize environmental consciousness,
recycling extensively and investing heavily in renewable energy. Family values remain strong,
though modern German society is increasingly multicultural.

INDIA CULTURE:
India's culture is one of the world's oldest and most diverse, shaped by thousands of years 
of history, religion, and regional traditions. Hinduism, Buddhism, Jainism, and Sikhism 
originated here, creating a deeply spiritual society. Joint family systems remain common, 
with strong emphasis on respect for elders. Indian cuisine varies dramatically by region, 
using complex spice blends and diverse cooking techniques. Classical dance forms like 
Bharatanatyam and Kathak preserve ancient traditions. Bollywood dominates entertainment, 
producing more films annually than any other country.

JAPAN CULTURE:
Japanese culture uniquely blends ancient traditions with cutting-edge modernity. The concept 
of 'wa' (harmony) underlies social interactions, emphasizing group cohesion over individualism. 
Shinto and Buddhist influences shape daily life, from shrine visits to seasonal festivals. 
Tea ceremony, ikebana (flower arranging), and calligraphy represent refined aesthetic 
principles. Japanese cuisine emphasizes seasonality, presentation, and fresh ingredients. 
Manga and anime have become worldwide cultural exports. The workplace culture values loyalty, 
dedication, and consensus decision-making.

YOUR TASK - Create EXACTLY 5 steps:
Step 1: Summarize German culture (100-150 words) → "germany_culture.txt"
Step 2: Summarize Indian culture (100-150 words) → "india_culture.txt"
Step 3: Summarize Japanese culture (100-150 words) → "japan_culture.txt"
Step 4: Read all 3 culture files using read_file()
Step 5: Create comparative analysis → "final_comparison.txt"

CRITICAL:
- EXACTLY 5 steps
- Store SUMMARIES (100-150 words each), NOT raw text
- Use EXACT filenames specified
- Read files selectively
""",
        "expected_files": ["germany_culture.txt", "india_culture.txt", "japan_culture.txt", "final_comparison.txt"],
        "min_write_ops": 4,
        "min_read_ops": 3
    },
    {
        "name": "AI Frameworks",
        "task": """
Analyze and compare 3 AI frameworks: TensorFlow, PyTorch, and JAX.

TENSORFLOW:
Developed by Google Brain. Uses static computation graphs (though eager execution available). 
Strong production deployment tools (TF Serving, TF Lite). Extensive ecosystem (Keras integration). 
Widely used in industry. Steeper learning curve. Good mobile/embedded support.

PYTORCH:
Developed by Facebook AI Research. Uses dynamic computation graphs. Pythonic and intuitive. 
Dominant in research community. Easy debugging. Strong GPU acceleration. Growing production 
tools (TorchServe). Preferred for NLP and computer vision research.

JAX:
Developed by Google Research. Functional programming paradigm. Automatic differentiation. 
Composable transformations (grad, jit, vmap, pmap). NumPy-compatible API. Excellent for 
scientific computing and ML research. Steeper learning curve.

YOUR TASK - Create EXACTLY 5 steps:
Step 1: Summarize TensorFlow → "tensorflow_summary.txt"
Step 2: Summarize PyTorch → "pytorch_summary.txt"
Step 3: Summarize JAX → "jax_summary.txt"
Step 4: Read all 3 framework files
Step 5: Create comparison report → "framework_comparison.txt"

CRITICAL:
- Each summary 100-150 words
- Use exact filenames specified
- Read files before comparison
""",
        "expected_files": ["tensorflow_summary.txt", "pytorch_summary.txt", "jax_summary.txt", "framework_comparison.txt"],
        "min_write_ops": 4,
        "min_read_ops": 3
    },
    {
        "name": "Climate Regions",
        "task": """
Analyze climate change impacts on 3 regions.

ARCTIC:
Warming twice as fast as global average. Sea ice extent declined 13% per decade since 1979. 
Permafrost thawing releases methane, accelerating warming. Indigenous communities face 
displacement. Wildlife habitats disrupted. Ocean acidification affects marine ecosystems.

AMAZON:
Deforestation rates reached 10,000 km² annually. Biodiversity loss accelerates. Carbon sink 
capacity declining. Rainfall patterns changing, affecting agriculture. Indigenous lands under 
pressure. Fire frequency increasing.

SAHARA:
Desert expansion southward at 10km per decade. Water scarcity intensifying. Agricultural 
productivity declining. Migration patterns shifting. Solar energy potential increasing. Dust 
storms affecting air quality globally.

YOUR TASK - Create EXACTLY 5 steps:
Step 1: Summarize Arctic impacts → "arctic_summary.txt"
Step 2: Summarize Amazon impacts → "amazon_summary.txt"
Step 3: Summarize Sahara impacts → "sahara_summary.txt"
Step 4: Read all 3 regional files
Step 5: Create global analysis → "climate_analysis.txt"

CRITICAL:
- Store summaries (100-150 words each)
- Use exact filenames specified
- Read files before synthesis
""",
        "expected_files": ["arctic_summary.txt", "amazon_summary.txt", "sahara_summary.txt", "climate_analysis.txt"],
        "min_write_ops": 4,
        "min_read_ops": 3
    },
    {
        "name": "Programming Languages",
        "task": """
Compare 3 programming languages for web development.

PYTHON:
High-level, interpreted language. Clean, readable syntax. Extensive libraries (Django, Flask). 
Strong in data science and ML. Slower execution speed. Great for rapid prototyping. Large 
community support.

JAVASCRIPT:
Native browser language. Event-driven, asynchronous programming. Node.js for backend. Huge 
ecosystem (npm). Essential for frontend development. Fast execution in browsers. Constantly 
evolving (ES6+).

GO:
Compiled, statically typed. Developed by Google. Excellent concurrency (goroutines). Fast 
execution speed. Simple syntax. Growing web framework ecosystem (Gin, Echo). Strong in 
microservices and APIs.

YOUR TASK - Create EXACTLY 5 steps:
Step 1: Summarize Python → "python_summary.txt"
Step 2: Summarize JavaScript → "javascript_summary.txt"
Step 3: Summarize Go → "go_summary.txt"
Step 4: Read all 3 language files
Step 5: Create comparison report → "language_comparison.txt"

CRITICAL:
- Each summary 100-150 words
- Use exact filenames specified
- Read files before final comparison
""",
        "expected_files": ["python_summary.txt", "javascript_summary.txt", "go_summary.txt", "language_comparison.txt"],
        "min_write_ops": 4,
        "min_read_ops": 3
    },
    {
        "name": "Historical Revolutions",
        "task": """
Compare 3 historical revolutions.

FRENCH REVOLUTION (1789-1799):
Overthrew monarchy. Established republic. Reign of Terror. Napoleon's rise. Inspired democratic 
movements globally. Social class restructuring. Declaration of Rights of Man.

AMERICAN REVOLUTION (1775-1783):
Independence from Britain. Constitutional democracy. Bill of Rights. Federal system established. 
Influenced global independence movements. Enlightenment ideals.

RUSSIAN REVOLUTION (1917):
Overthrew Tsar. Communist government established. Civil war. Soviet Union formed. Global 
ideological impact. Industrialization push. Red vs White conflict.

YOUR TASK - Create EXACTLY 5 steps:
Step 1: Summarize French Revolution → "french_revolution.txt"
Step 2: Summarize American Revolution → "american_revolution.txt"
Step 3: Summarize Russian Revolution → "russian_revolution.txt"
Step 4: Read all 3 revolution files
Step 5: Create comparative analysis → "revolution_comparison.txt"

CRITICAL:
- Each summary 100-150 words
- Store in separate files
- Read before synthesis
""",
        "expected_files": ["french_revolution.txt", "american_revolution.txt", "russian_revolution.txt", "revolution_comparison.txt"],
        "min_write_ops": 4,
        "min_read_ops": 3
    }
]


# ─────────────────────────────────────────────────────────────────
# VALIDATION FUNCTIONS
# ─────────────────────────────────────────────────────────────────

def analyze_tool_sequence(messages):
    """Extract tool invocation sequence from messages."""
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


def validate_scenario_result(scenario_name, result, expected_files, min_write_ops, min_read_ops):
    """Validate a single scenario's results."""
    todos = result.get("todos", [])
    messages = result.get("messages", [])
    sequence = analyze_tool_sequence(messages)
    
    # Count tool operations
    write_ops = [s for s in sequence if s["tool"] == "write_file"]
    read_ops = [s for s in sequence if s["tool"] == "read_file"]
    
    # Check files created
    files_created = set()
    if FILE_SYSTEM_DIR.exists():
        files_created = {f.name for f in FILE_SYSTEM_DIR.iterdir() if f.is_file()}
    
    # Validation checks
    checks = {
        "planning_completed": "write_todos" in [s["tool"] for s in sequence],
        "exactly_5_todos": len(todos) == 5,
        "write_file_used": len(write_ops) >= min_write_ops,
        "read_file_used": len(read_ops) >= min_read_ops,
        "expected_files_created": all(f in files_created for f in expected_files),
        "meaningful_filenames": not any(
            f in fname.lower() 
            for fname in files_created 
            for f in ["file1", "file2", "data", "temp"]
        )
    }
    
    passed_count = sum(1 for v in checks.values() if v)
    total_count = len(checks)
    score = (passed_count / total_count * 100)
    
    return {
        "scenario": scenario_name,
        "checks": checks,
        "passed": passed_count,
        "total": total_count,
        "score": score,
        "files_created": len(files_created),
        "write_ops": len(write_ops),
        "read_ops": len(read_ops)
    }


def print_unified_workflow():
    """Display the unified workflow diagram."""
    print("\n" + "=" * 100)
    print("UNIFIED WORKFLOW: PLANNING + EXECUTION")
    print("=" * 100)
    workflow = """
┌─────────────────────────────────────────────────────────────────────┐
│ USER INPUT: Complex multi-step research task                        │
└─────────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────────┐
│ MILESTONE 1: PLANNING PHASE                                         │
├─────────────────────────────────────────────────────────────────────┤
│ Agent calls: write_todos()                                          │
│ Result: 5-step plan                                                 │
└─────────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────────┐
│ MILESTONE 2: EXECUTION PHASE                                        │
├─────────────────────────────────────────────────────────────────────┤
│ For each TODO step:                                                 │
│   - Process information                                             │
│   - write_file() to store summaries                                 │
│   - read_file() to retrieve context                                 │
│   - edit_file() to update if needed                                 │
└─────────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────────┐
│ RESULT: Plan created + Files stored                                 │
│ ✅ TODOs: 5 steps in memory                                         │
│ ✅ Files: Persistent on disk                                        │
└─────────────────────────────────────────────────────────────────────┘
    """
    print(workflow)
    print("=" * 100 + "\n")


def run_plan_validation():
    """Run unified plan validation across ALL scenarios."""
    # Show workflow diagram first
    print_unified_workflow()
    
    print("=" * 100)
    print("PLAN VALIDATION: MULTI-PROMPT TEST (MILESTONE 1 + 2)")
    print("=" * 100)
    print(f"\nTesting {len(TEST_SCENARIOS)} different task scenarios...")
    print("Success Criteria: >80% pass rate across all scenarios\n")
    print("=" * 100)
    
    # Setup
    FILE_SYSTEM_DIR.mkdir(exist_ok=True)
    BACKUP_DIR = Path("virtual_fs_backup")
    BACKUP_DIR.mkdir(exist_ok=True)
    
    print("\nInitializing agent... ", end="")
    agent = create_agent_executor()
    system_prompt = create_system_prompt()
    print("✓\n")
    
    results = []
    
    for i, scenario in enumerate(TEST_SCENARIOS, 1):
        print(f"\n{'=' * 100}")
        print(f"SCENARIO {i}/{len(TEST_SCENARIOS)}: {scenario['name']}")
        print(f"{'=' * 100}")
        
        # Clear file system for each test
        clear_virtual_fs()
        for f in BACKUP_DIR.iterdir():
            if f.is_file():
                f.unlink()
        
        print(f"\nRunning task (45-90 seconds)...")
        
        try:
            result = agent.invoke(
                {"messages": [HumanMessage(content=f"{system_prompt}\n\n{scenario['task']}")]},
                {
                    "configurable": {"thread_id": f"scenario-{i}"},
                    "recursion_limit": 50
                }
            )
            
            # Validate
            validation = validate_scenario_result(
                scenario['name'],
                result,
                scenario['expected_files'],
                scenario['min_write_ops'],
                scenario['min_read_ops']
            )
            results.append(validation)
            
            # Display results
            print(f"\n✓ TODOs Created: {len(result.get('todos', []))}")
            print(f"✓ Write Operations: {validation['write_ops']}")
            print(f"✓ Read Operations: {validation['read_ops']}")
            print(f"✓ Files Created: {validation['files_created']}")
            
            icon = "🎉" if validation['score'] >= 80 else "⚠️"
            print(f"\n{icon} SCENARIO SCORE: {validation['passed']}/{validation['total']} ({validation['score']:.1f}%)")
            
        except Exception as e:
            print(f"\n❌ ERROR: {e}")
            results.append({
                "scenario": scenario['name'],
                "score": 0,
                "passed": 0,
                "total": 6,
                "error": str(e)
            })
        
        # Small delay between tests
        time.sleep(1)
    
    # Overall Summary
    print(f"\n{'=' * 100}")
    print("OVERALL VALIDATION RESULTS")
    print(f"{'=' * 100}")
    
    passed_scenarios = sum(1 for r in results if r['score'] >= 80)
    total_scenarios = len(results)
    overall_score = (passed_scenarios / total_scenarios * 100)
    
    print(f"\nScenarios Passed: {passed_scenarios}/{total_scenarios}")
    print(f"Overall Success Rate: {overall_score:.1f}%")
    
    print(f"\n{'Scenario':<25} {'Score':<10} {'Files':<10} {'Write':<10} {'Read':<10} {'Status'}")
    print("-" * 100)
    
    for r in results:
        status = "✅ PASS" if r['score'] >= 80 else "❌ FAIL"
        score_display = f"{r['score']:.1f}%" if 'score' in r else "ERROR"
        files_display = r.get('files_created', 0)
        write_display = r.get('write_ops', 0)
        read_display = r.get('read_ops', 0)
        print(f"{r['scenario']:<25} {score_display:<10} {files_display:<10} {write_display:<10} {read_display:<10} {status}")
    
    print(f"\n{'=' * 100}")
    
    if overall_score >= 80:
        print("🎉 MILESTONE 2 VALIDATION: PASSED ✓")
        print(f"\nSuccess Criteria Met: {overall_score:.1f}% >= 80%")
        print("\nCapabilities Validated:")
        print("  ✅ File-based context offloading works across multiple task types")
        print("  ✅ write_file operations store intermediate results")
        print("  ✅ read_file operations retrieve stored context")
        print("  ✅ Meaningful filenames used consistently")
        print("  ✅ Planning + Execution workflow stable")
    else:
        print("⚠️ MILESTONE 2 VALIDATION: NEEDS IMPROVEMENT")
        print(f"\nSuccess Criteria Not Met: {overall_score:.1f}% < 80%")
        print("\nRecommended Actions:")
        print("  • Review system prompt for file usage instructions")
        print("  • Check tool definitions for write_file/read_file")
        print("  • Verify LangGraph state management for files")
    
    print(f"\n{'=' * 100}")
    print(f"LangSmith Project: milestone2-multi-prompt")
    print(f"URL: https://smith.langchain.com/")
    print(f"{'=' * 100}\n")
    
    return overall_score >= 80


if __name__ == "__main__":
    print("\n" + "=" * 100)
    print("  PLAN VALIDATION: UNIFIED WORKFLOW (MILESTONE 1 + MILESTONE 2)")
    print("  Planning Phase → Execution Phase → File Deliverables")
    print("=" * 100 + "\n")
    
    success = run_plan_validation()
    
    print(f"\n{'='*100}")
    print("FINAL SUMMARY")
    print(f"{'='*100}")
    print(f"Status: {'✅ PASSED' if success else '❌ FAILED'}")
    print("\nWhat was validated:")
    print("  • Milestone 1: 5-step planning capability")
    print("  • Milestone 2: File-based context offloading")
    print("  • Multi-scenario: 5 different task types tested")
    print("  • Unified workflow: Plan → Execute → Store → Retrieve → Synthesize")
    print(f"{'='*100}\n")
    
    sys.exit(0 if success else 1)
