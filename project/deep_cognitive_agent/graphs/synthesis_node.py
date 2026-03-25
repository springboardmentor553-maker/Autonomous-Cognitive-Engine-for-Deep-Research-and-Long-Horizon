"""
Synthesis Node - Milestone 2: Selective Retrieval + Final Summary

Demonstrates selective retrieval discipline:
  - Reads ONLY the key output files (unified_model.txt, comparison.txt)
  - Does NOT re-read all individual research summaries
  - Builds final output from processed/synthesized content only

Architecture:  START → plan → execute → **synthesize** → END
"""

import re
import time
import os
import json

from langchain_core.messages import AIMessage

from tools.vfs.read_file import read_file
from tools.vfs.ls import ls
from utils.helpers import (
    parse_retry_after,
    is_rate_limit_error,
    is_server_overload_error,
    invoke_with_retry,
    sanitize_llm_output,
)


def _extract_json_object(text: str) -> dict:
    """Best-effort parse of a JSON object from model output."""
    if not text:
        return {}
    try:
        return json.loads(text)
    except Exception:
        pass

    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        return {}

    try:
        return json.loads(match.group(0))
    except Exception:
        return {}


def _evaluate_quality(llm, task: str, candidate: str) -> dict:
    """LLM-as-judge quality evaluator returning normalized scoring JSON."""
    eval_prompt = (
        "You are a strict quality evaluator. Score ONLY the given final report.\n"
        "Return valid JSON only (no prose, no markdown) with this exact schema:\n"
        "{\n"
        "  \"score\": int,\n"
        "  \"breakdown\": {\"accuracy\": int, \"coverage\": int, \"structure\": int, \"actionability\": int, \"coherence\": int},\n"
        "  \"must_fix\": [string],\n"
        "  \"strengths\": [string]\n"
        "}\n"
        "Scoring rules:\n"
        "- Each breakdown field is 0-100\n"
        "- score is overall 0-100\n"
        "- Be strict and evidence-based\n"
        "- If report contains <think> tags, reduce score heavily\n\n"
        f"Original user task:\n{task}\n\n"
        f"Candidate report:\n{candidate}"
    )

    raw = invoke_with_retry(llm, eval_prompt)
    parsed = _extract_json_object(raw)

    score = parsed.get("score", 0)
    try:
        score = int(score)
    except Exception:
        score = 0

    breakdown = parsed.get("breakdown", {}) if isinstance(parsed.get("breakdown", {}), dict) else {}
    must_fix = parsed.get("must_fix", []) if isinstance(parsed.get("must_fix", []), list) else []
    strengths = parsed.get("strengths", []) if isinstance(parsed.get("strengths", []), list) else []

    return {
        "score": max(0, min(100, score)),
        "breakdown": breakdown,
        "must_fix": must_fix,
        "strengths": strengths,
    }


def _refine_with_feedback(llm, task: str, current: str, quality_report: dict) -> str:
    """Refine report based on evaluator feedback while preserving structure."""
    must_fix = quality_report.get("must_fix", [])
    breakdown = quality_report.get("breakdown", {})

    refine_prompt = (
        "Rewrite and improve the report below to maximize quality.\n"
        "Hard requirements:\n"
        "1) Keep this exact section order with clear headings: Overview, Key Findings, Analysis, Recommendations, Conclusion\n"
        "2) Improve factual precision, coverage, and actionable recommendations\n"
        "3) Remove filler and redundancy\n"
        "4) DO NOT include any <think> tags or hidden reasoning\n"
        "5) Return only the improved report text\n\n"
        f"Original task:\n{task}\n\n"
        f"Quality breakdown:\n{json.dumps(breakdown, ensure_ascii=False)}\n"
        f"Must-fix issues:\n{json.dumps(must_fix, ensure_ascii=False)}\n\n"
        f"Current report:\n{current}"
    )
    refined = invoke_with_retry(llm, refine_prompt)
    return sanitize_llm_output(refined)


def _rewrite_from_sources(llm, task: str, sources: str, quality_report: dict) -> str:
    """Regenerate a high-quality final report directly from sources and feedback."""
    must_fix = quality_report.get("must_fix", [])
    rewrite_prompt = (
        "Generate a high-quality final report from the source material.\n"
        "Hard requirements:\n"
        "1) Section order must be: Overview, Key Findings, Analysis, Recommendations, Conclusion\n"
        "2) Be comprehensive but concise, with concrete and actionable recommendations\n"
        "3) Ensure logical flow and remove redundancy\n"
        "4) Do not include hidden reasoning tags like <think>\n"
        "5) Return only the final report\n\n"
        f"Task:\n{task}\n\n"
        f"Priority fixes from prior evaluation:\n{json.dumps(must_fix, ensure_ascii=False)}\n\n"
        f"Source material:\n{sources}"
    )
    rewritten = invoke_with_retry(llm, rewrite_prompt)
    return sanitize_llm_output(rewritten)


# ── Node Function ────────────────────────────────────────────────────

def synthesize_node(state: dict, llm) -> dict:
    """
    Synthesis node: selectively reads key output files from VFS
    (NOT all files) and generates a final structured summary.

    Selective retrieval logic:
      1. If unified_model.txt exists → read only that (already refined)
      2. If comparison_analysis.txt exists → read that as supplement
      3. Only fall back to all files if no key files found

    This demonstrates:
      ✔ Selective retrieval — not blind loading
      ✔ Memory efficiency — reads processed outputs, not raw data
      ✔ Architectural discipline — clear reasoning for each read

    Args:
        state: Current AgentState dict.
        llm:   ChatGroq (or compatible) LLM instance.

    Returns:
        Partial state update with final_output, todos, trace_log,
        and messages.
    """
    files = dict(state.get("files", {}))
    vfs_state = {"files": files}
    trace_log = list(state.get("trace_log", []))
    user_task = ""
    if state.get("messages"):
        first_msg = state["messages"][0]
        user_task = getattr(first_msg, "content", "") if first_msg else ""

    # ── Step 1: List files and classify them ──
    file_list = ls(vfs_state)
    trace_log.append({
        "action": "ls",
        "file": None,
        "purpose": "Identify available files for selective retrieval",
        "step": "synthesis",
    })

    print(f"\n{'='*60}")
    print(f"[Synthesize Node] Files in VFS: {file_list}")
    print(f"{'='*60}")

    # Classify files into categories
    key_files = []      # unified models, comparisons
    summary_files = []  # individual research summaries

    for fname in sorted(file_list):
        if "unified" in fname or "model" in fname or "final" in fname:
            key_files.append(fname)
        elif "comparison" in fname or "analysis" in fname:
            key_files.append(fname)
        else:
            summary_files.append(fname)

    # ── Step 2: Selective retrieval — read only key files ──
    files_to_read = key_files if key_files else summary_files

    print(f"\n  Selective retrieval strategy:")
    print(f"    Key files (will read): {key_files}")
    print(f"    Summary files (skipped): {summary_files}")
    if not key_files:
        print(f"    Fallback: reading all {len(summary_files)} summary files")

    contents = []
    for fname in files_to_read:
        content = read_file(vfs_state, fname)
        purpose = (
            "Load key output for final synthesis"
            if fname in key_files
            else "Fallback: load summary for final synthesis"
        )
        trace_log.append({
            "action": "read_file",
            "file": fname,
            "purpose": purpose,
            "step": "synthesis",
        })
        print(f"    → read_file('{fname}'): {len(content)} chars")
        contents.append(f"--- {fname} ---\n{content}")

    combined_text = "\n\n".join(contents)

    # ── Step 3: Generate final structured summary with LLM ──
    prompt = (
        "You are given the key outputs from a multi-step research and "
        "analysis pipeline. Create ONE final structured summary that "
        "presents the complete findings.\n\n"
        "Use this structure:\n"
        "1. **Overview**: Brief introduction to the topic and scope\n"
        "2. **Key Findings**: Bullet points of the most important "
        "discoveries and insights\n"
        "3. **Analysis**: Deeper analysis connecting themes, patterns, "
        "and implications\n"
        "4. **Recommendations**: Actionable recommendations based on "
        "the analysis\n"
        "5. **Conclusion**: Final takeaway and future considerations\n\n"
        f"Key outputs from the pipeline:\n\n{combined_text}\n\n"
        "Write the final structured summary now:"
    )

    print(f"\n[Synthesize Node] Generating final summary from "
          f"{len(files_to_read)} key files...")
    print(f"  (Skipped {len(summary_files)} raw summary files — "
          f"using processed outputs only)")

    final_summary = None
    max_retries = 3
    for attempt in range(max_retries):
        try:
            response = llm.invoke(prompt)
            final_summary = sanitize_llm_output(response.content)
            break
        except Exception as e:
            err_str = str(e)
            if attempt < max_retries - 1:
                if is_rate_limit_error(err_str):
                    wait = parse_retry_after(err_str)
                    print(f"  ⏳ Rate limited. Waiting {wait}s...")
                    time.sleep(wait)
                    continue
                if is_server_overload_error(err_str):
                    wait = min(2 ** attempt * 10, 60)
                    print(f"  ⏳ Server overloaded (503). Waiting {wait}s...")
                    time.sleep(wait)
                    continue
            raise

    # ── Step 4: Quality-gated refinement loop (target defaults to 95) ──
    quality_target = int(os.getenv("QUALITY_TARGET_SCORE", "96"))
    max_refinements = int(os.getenv("QUALITY_MAX_REFINEMENTS", "5"))

    best_summary = final_summary
    best_quality = _evaluate_quality(llm, user_task, final_summary)

    print(f"[Synthesize Node] Initial quality score: {best_quality['score']}/100")

    current_summary = final_summary
    current_quality = best_quality
    for round_idx in range(1, max_refinements + 1):
        if current_quality["score"] >= quality_target:
            break

        print(
            f"[Synthesize Node] Refinement round {round_idx}/{max_refinements} "
            f"(score {current_quality['score']} < target {quality_target})"
        )
        # Strategy A: improve current best draft.
        candidate_refine = _refine_with_feedback(
            llm,
            user_task,
            best_summary,
            current_quality,
        )
        refine_quality = _evaluate_quality(
            llm,
            user_task,
            candidate_refine,
        )

        # Strategy B: full rewrite from source files + evaluator feedback.
        candidate_rewrite = _rewrite_from_sources(
            llm,
            user_task,
            combined_text,
            current_quality,
        )
        rewrite_quality = _evaluate_quality(
            llm,
            user_task,
            candidate_rewrite,
        )

        # Keep the best candidate from this round.
        if rewrite_quality["score"] >= refine_quality["score"]:
            current_summary, current_quality = candidate_rewrite, rewrite_quality
        else:
            current_summary, current_quality = candidate_refine, refine_quality

        if current_quality["score"] > best_quality["score"]:
            best_quality = current_quality
            best_summary = current_summary

    final_summary = best_summary

    trace_log.append({
        "action": "quality_evaluate",
        "file": None,
        "purpose": (
            f"Final output quality scored {best_quality['score']}/100 "
            f"(target: {quality_target}, max_refinements: {max_refinements})"
        ),
        "step": "synthesis",
    })

    # ── Mark any remaining todos as done ──
    todos = list(state.get("todos", []))
    for i in range(len(todos)):
        if todos[i].get("status") != "done":
            todos[i] = {**todos[i], "status": "done"}

    trace_log.append({
        "action": "synthesize",
        "file": None,
        "purpose": f"Final structured summary from {len(files_to_read)} key files",
        "step": "synthesis",
    })

    print(f"[Synthesize Node] Final summary generated "
          f"({len(final_summary)} chars)")

    return {
        "final_output": final_summary,
        "quality_score": best_quality["score"],
        "quality_target": quality_target,
        "quality_passed": best_quality["score"] >= quality_target,
        "quality_report": best_quality,
        "todos": todos,
        "trace_log": trace_log,
        "messages": [
            AIMessage(
                content=(
                    f"Final structured summary created from "
                    f"{len(files_to_read)} key files "
                    f"(skipped {len(summary_files)} raw summaries). "
                    f"Quality score: {best_quality['score']}/100 "
                    f"(target: {quality_target})."
                )
            )
        ],
    }
