from __future__ import annotations

from app.models import LLMClient
from app.state import GraphState
from storage.file_store import list_files, read_file


def synthesize_results(state: GraphState) -> str:
    llm = LLMClient()
    combined_data: list[str] = []

    for filename in list_files(state):
        combined_data.append(f"FILE: {filename}\n{read_file(filename, state)}")

    final_report = llm.predict(
        "\n\n".join(combined_data) or "No intermediate files were created.",
        system_prompt=(
            "You are the final synthesis node. Combine all intermediate outputs into one "
            "report with sections for overview, findings, evidence, limitations, and conclusion."
        ),
    )
    state["final_report"] = final_report
    return final_report
