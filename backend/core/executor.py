from backend.sub_agents.research_agent import research_agent
from backend.sub_agents.analysis_agent import analysis_agent
from backend.sub_agents.summarizer_agent import summarizer_agent
from backend.sub_agents.web_search_agent import search_web
from backend.memory.memory_manager import MemoryManager
from backend.core.evaluator import evaluate_output   # ✅ ADDED

import time
import re

memory = MemoryManager()


# 🔥 SAFE CALL (handles API errors)
def safe_call(agent_function, input_text, retries=5):
    last_error = None

    for attempt in range(retries):
        try:
            return agent_function(input_text)

        except Exception as e:
            last_error = str(e)

            if "503" in last_error:
                wait_time = 2 * (attempt + 1)
                print(f"⚠️ Server busy... retrying in {wait_time}s")
                time.sleep(wait_time)

            elif "429" in last_error or "RESOURCE_EXHAUSTED" in last_error:
                print("🚫 Rate limit hit!")

                match = re.search(r"retry in (\d+)", last_error.lower())
                wait_time = int(match.group(1)) + 2 if match else 30

                print(f"⏳ Waiting {wait_time}s...")
                time.sleep(wait_time)

            else:
                raise e

    return f"❌ Failed after retries: {last_error}"

def clean_output(text):
    import re

    text = text.strip()

    # ❌ Remove markdown stars (***, **)
    text = re.sub(r"\*{1,3}", "", text)

    # ❌ Remove repeated dots (..., ..)
    text = re.sub(r"\.{2,}", ".", text)

    # ❌ Fix bullet mess
    text = text.replace("• *", "•")
    text = text.replace("* •", "•")

    # ❌ Remove duplicate lines
    lines = []
    seen = set()

    for line in text.split("\n"):
        line = line.strip()
        if line and line not in seen:
            lines.append(line)
            seen.add(line)

    text = "\n".join(lines)

    # ✅ Add spacing for sections
    text = text.replace("📌 DETAILED REPORT", "\n📌 DETAILED REPORT\n")
    text = text.replace("🧠 ANALYSIS", "\n\n🧠 ANALYSIS\n")
    text = text.replace("📊 FINAL SUMMARY", "\n\n📊 FINAL SUMMARY\n")

    # ✅ Clean bullet formatting
    text = text.replace("•", "\n•")

    # ✅ Clean numbered list
    for i in range(1, 10):
        text = text.replace(f"{i}.", f"\n{i}.")

    # ✅ Final cleanup (remove extra blank lines)
    final = "\n".join([l.strip() for l in text.split("\n") if l.strip()])

    return final

def run_agent(objective, output_format="summary"):

    print("\n" + "="*50)
    print(f"🧠 Task: {objective}")
    print("="*50)

    try:
        past_context = memory.recall(objective)
        context_text = "\n".join([item["content"] for item in past_context]) if past_context else ""

        # ===============================
        # 🚀 SUMMARY MODE
        # ===============================
        if output_format == "summary":

            summary_prompt = f"""
Create a clean structured summary.

Rules:
- Max 120 words
- No repetition

Topic: {objective}
"""

            result = safe_call(summarizer_agent, summary_prompt)

            # ✅ ADD EVALUATION
           # evaluate_output(result)

            return result

        # ===============================
        # 🚀 DETAILED MODE
        # ===============================

        web_data = search_web(objective)
        if not web_data.strip():
            web_data = "No relevant web data found."

        print("✅ Web search done")

        research_input = f"""
Topic: {objective}

Use:
- Latest trends
- Examples
- Statistics

Context:
{context_text}

Web Data:
{web_data}

Limit: 400 words max
"""

        research = safe_call(research_agent, research_input)
        if "❌" in research:
            return research

        print("✅ Research done")

        analysis = safe_call(analysis_agent, research)
        if "❌" in analysis:
            return analysis

        print("✅ Analysis done")

        final_output = safe_call(summarizer_agent, research)
        if "❌" in final_output:
            return final_output

        # 💾 MEMORY
        memory.after_run(objective, final_output, todos=[])

        output = f"""
📌 DETAILED REPORT

{research}

━━━━━━━━━━━━━━━━━━━━━━

🧠 ANALYSIS

{analysis}

━━━━━━━━━━━━━━━━━━━━━━

📊 FINAL SUMMARY

{final_output}
""".strip()

        output = clean_output(output)

        # ✅ ADD EVALUATION HERE
        evaluate_output(output)

        return output

    except Exception as e:
        return f"❌ Error: {str(e)}"