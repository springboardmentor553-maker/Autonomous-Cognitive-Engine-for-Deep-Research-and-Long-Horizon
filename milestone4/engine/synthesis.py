from tools.file_tools import read_file, ls
from utils.llm import llm

def synthesize_results(state):

    files = ls(state)

    summaries = []

    # Only take LIMITED content per file
    for file in files[:8]:   # limit files
        content = read_file(state, file)

        # truncate large files
        content = content[:500]

        summary = llm.invoke(
            f"Summarize this in 3 lines:\n{content}"
        ).content

        summaries.append(summary)

    combined = "\n".join(summaries)

    final_report = llm.invoke(
        f"""
        Create a clean structured report from:

        {combined}

        Include:
        - Title
        - Key Insights
        - Conclusion
        """
    )

    return final_report.content