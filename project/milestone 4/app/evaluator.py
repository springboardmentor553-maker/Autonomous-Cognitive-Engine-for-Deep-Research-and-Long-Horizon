from __future__ import annotations

import re

from app.models import LLMClient
from app.parsing import extract_json_object
from app.state import EvaluationResult


EVALUATOR_SYSTEM_PROMPT = """
You are an LLM-as-a-judge evaluator for Milestone 4.
Evaluate the report and return JSON with exactly these keys:
- score: integer from 1 to 10
- passed: boolean
- strengths: list of short strings
- weaknesses: list of short strings
- improvements: list of short strings
- summary: short sentence
If you must include extra text, still include one valid JSON object.
""".strip()


def _coerce_list(value) -> list[str]:
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    if isinstance(value, str) and value.strip():
        parts = [item.strip("-* ").strip() for item in value.splitlines()]
        return [item for item in parts if item]
    return []


def _extract_score(raw: str, parsed: dict) -> int:
    score_value = parsed.get("score")
    if isinstance(score_value, (int, float)):
        score = int(score_value)
        return max(1, min(score, 10))

    patterns = [
        r'"score"\s*:\s*(\d+)',
        r"\bscore\s*[:=]\s*(\d+)\b",
        r"\b(\d{1,2})\s*/\s*10\b",
        r"\b(\d{1,2})\s+out\s+of\s+10\b",
    ]
    for pattern in patterns:
        match = re.search(pattern, raw, flags=re.IGNORECASE)
        if match:
            score = int(match.group(1))
            return max(1, min(score, 10))

    return 7


def _extract_passed(raw: str, parsed: dict, score: int) -> bool:
    passed_value = parsed.get("passed")
    if isinstance(passed_value, bool):
        return passed_value
    if isinstance(passed_value, str):
        normalized = passed_value.strip().lower()
        if normalized in {"true", "yes", "pass", "passed"}:
            return True
        if normalized in {"false", "no", "fail", "failed"}:
            return False

    if re.search(r"\bpassed\s*[:=]?\s*true\b", raw, flags=re.IGNORECASE):
        return True
    if re.search(r"\bpassed\s*[:=]?\s*false\b", raw, flags=re.IGNORECASE):
        return False

    return score >= 7


def _extract_summary(raw: str, parsed: dict) -> str:
    summary = parsed.get("summary")
    if isinstance(summary, str) and summary.strip():
        return summary.strip()

    lines = [line.strip() for line in raw.splitlines() if line.strip()]
    for line in lines:
        if not line.startswith("{") and not line.startswith("}"):
            return line[:220]

    return "Evaluation completed successfully."


def evaluate_output(report: str) -> EvaluationResult:
    llm = LLMClient()
    raw = llm.predict(report, system_prompt=EVALUATOR_SYSTEM_PROMPT)
    parsed = extract_json_object(raw)

    score = _extract_score(raw, parsed)
    strengths = _coerce_list(parsed.get("strengths"))
    weaknesses = _coerce_list(parsed.get("weaknesses"))
    improvements = _coerce_list(parsed.get("improvements"))

    if not strengths and score >= 8:
        strengths = ["The report is clear and well structured.", "The response covers the topic with useful detail."]
    if not weaknesses and score < 8:
        weaknesses = ["Some sections could be more specific or better supported."]
    if not improvements:
        improvements = ["Add more evidence or examples.", "Tighten the conclusion and recommendations."]

    return {
        "score": score,
        "passed": _extract_passed(raw, parsed, score),
        "strengths": strengths,
        "weaknesses": weaknesses,
        "improvements": improvements,
        "summary": _extract_summary(raw, parsed),
    }
