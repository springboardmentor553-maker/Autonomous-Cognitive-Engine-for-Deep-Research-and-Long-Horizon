from backend.sub_agents.research_agent import research_agent
from backend.sub_agents.analysis_agent import analysis_agent
from backend.sub_agents.summarizer_agent import summarizer_agent


def task_tool(task):
    try:
        research = research_agent(task)
        analysis = analysis_agent(research)
        summary = summarizer_agent(analysis)

        return f"""
📌 Research Output:
{research}

--------------------------------------------------

🔍 Analysis Output:
{analysis}

--------------------------------------------------

✅ Final Summary:
{summary}
"""
    except Exception as e:
        return f"❌ Error occurred: {str(e)}"