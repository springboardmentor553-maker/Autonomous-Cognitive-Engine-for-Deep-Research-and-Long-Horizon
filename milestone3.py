import os
import json
import time
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.prebuilt import create_react_agent

from tools import (
    write_todos, write_file, read_file, edit_file, ls, task,
    VFS, VirtualFileSystem
)
import tools as tools_module
from evaluation import evaluate_test, print_evaluation_report

load_dotenv()

os.environ["LANGCHAIN_TRACING_V2"] = os.getenv("LANGCHAIN_TRACING_V2", "true")
os.environ["LANGCHAIN_API_KEY"]     = os.getenv("LANGCHAIN_API_KEY", "")
os.environ["LANGCHAIN_PROJECT"]     = os.getenv("LANGCHAIN_PROJECT", "milestone_3_delegation")

llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.1)

# ── System Prompt ─────────────────────────────────────────────────────────────

system_prompt_text = """You are an autonomous cognitive engine for deep research and long-horizon tasks.

STRICT WORKFLOW:

STEP 1 — Call write_todos FIRST to create a 4-step plan. Do nothing else before this.

STEP 2 — Execute each TODO in order:
   Step A: Call task() with agent_name="research_agent"
           input_data = topic in 5 words or less (e.g. "LLMs in software productivity")
   Step B: Save result → write_file("research.txt", <result from task()>)
   Step C: Read file   → read_file("research.txt") to get the content
   Step D: Call task() with agent_name="summarization_agent"
           input_data = the actual content returned by read_file (NOT the filename)
   Step E: Save result → write_file("summary.txt", <result from task()>)

STEP 3 — Write your final answer using the saved content.

CRITICAL RULES:
- input_data for task() must be 5 words or less for research_agent
- NEVER pass a filename like "research.txt" as input_data to summarization_agent
- ALWAYS pass the actual text content to summarization_agent
- Do not repeat words in input_data
"""

# ── Test cases ────────────────────────────────────────────────────────────────

TEST_INPUTS = [
    "Research the impact of large language models on software development productivity and summarize the key findings.",
    "Investigate recent advances in quantum computing hardware and produce a structured summary report.",
    "Research the current state of autonomous vehicle regulations globally and summarize the key points.",
    "Find recent developments in renewable energy adoption and create a structured summary of the main trends.",
    "Research the applications of CRISPR gene-editing in medicine and summarize the most important breakthroughs.",
]


# ── Main evaluation loop ──────────────────────────────────────────────────────

def run_evaluation():

    agent = create_react_agent(
        llm,
        tools=[write_todos, write_file, read_file, edit_file, ls, task]
    )

    print(f"--- Starting Milestone 3 Evaluation ({len(TEST_INPUTS)} Test Cases) ---\n")

    passed_count = 0

    for i, test_task in enumerate(TEST_INPUTS):

        # Reset VFS for each test
        tools_module.VFS = VirtualFileSystem()

        print(f"\nTEST {i+1}/{len(TEST_INPUTS)}: {test_task}")
        print("-" * 70)

        inputs = {
            "messages": [
                SystemMessage(content=system_prompt_text),
                HumanMessage(content=test_task)
            ]
        }

        try:
            result = agent.invoke(inputs)

            # Show delegation results
            task_outputs = [
                json.loads(m.content)
                for m in result["messages"]
                if hasattr(m, "name") and m.name == "task" and m.content
            ]

            if task_outputs:
                print(f"\n✅ task() called {len(task_outputs)} time(s)")
                for idx, out in enumerate(task_outputs, 1):
                    icon    = "✅" if out.get("status") == "success" else "❌"
                    agent_n = out.get("agent", "unknown")
                    snippet = str(out.get("result", ""))[:150]
                    print(f"   {icon} Delegation {idx} → {agent_n}")
                    print(f"      {snippet}...")
            else:
                print("❌ ERROR: Agent did not call the task() delegation tool.")

            # Save output
            os.makedirs("outputs", exist_ok=True)
            fname = f"outputs/m3_test_{i+1}.json"
            with open(fname, "w") as f:
                json.dump({
                    "task":        test_task,
                    "delegations": task_outputs,
                    "vfs_state":   str(tools_module.VFS.list_files())
                }, f, indent=2)
            print(f"\n   Saved to {fname}")

            # Run evaluation (imported from evaluation.py)
            report = evaluate_test(result)
            print_evaluation_report(report, i + 1)

            if report["passed"]:
                passed_count += 1

        except Exception as e:
            print(f"❌ CRITICAL ERROR in Test {i+1}: {str(e)}")

        print("-" * 70)

        if i < len(TEST_INPUTS) - 1:
            print("Waiting 15 seconds to avoid rate limit...")
            time.sleep(15)

    # ── Final result ──────────────────────────────────────────────────────────
    total   = len(TEST_INPUTS)
    pct     = (passed_count / total) * 100
    passing = pct >= 80

    print(f"\n{'=' * 70}")
    print(f"MILESTONE 3 FINAL RESULT : {passed_count}/{total} passed ({pct:.0f}%)")
    print(f"Required                 : >80% (4 out of 5)")
    print(f"{'✅ MILESTONE 3 COMPLETE' if passing else '❌ MILESTONE 3 INCOMPLETE'}")
    print(f"{'=' * 70}\n")


if __name__ == "__main__":
    run_evaluation()