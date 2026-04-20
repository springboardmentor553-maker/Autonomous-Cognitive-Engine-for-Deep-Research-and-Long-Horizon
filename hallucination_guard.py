"""
Hallucination Guard — Cross-references AI-generated claims against web search sources.
Produces a confidence score and verification summary appended to the final report.
"""
import json
import logging
from difflib import SequenceMatcher
from typing import List, Dict, Tuple, Optional

logger = logging.getLogger(__name__)


def extract_claims(report_text: str, llm) -> List[str]:
    """
    Use the LLM to extract key factual claims from the report.
    Returns a list of factual assertion strings.
    """
    try:
        prompt = (
            "Extract the key factual claims from this report. "
            "Focus on specific facts: numbers, dates, statistics, named entities, "
            "percentages, and concrete assertions. "
            "Return ONLY a JSON array of strings, each being one factual claim. "
            "Return at most 10 claims. "
            "Example: [\"The global AI market was valued at $150 billion in 2023\", "
            "\"GPT-4 was released by OpenAI in March 2023\"]\n\n"
            f"Report:\n{report_text[:3000]}"
        )
        response = llm.invoke(prompt)
        content = response.content.strip()

        # Try to parse JSON from the response
        # Handle cases where LLM wraps in ```json blocks
        if "```" in content:
            content = content.split("```")[1]
            if content.startswith("json"):
                content = content[4:]
            content = content.strip()

        claims = json.loads(content)
        if isinstance(claims, list):
            logger.info(f"[HALLUCINATION GUARD] Extracted {len(claims)} claims.")
            return claims[:10]
        return []

    except (json.JSONDecodeError, Exception) as e:
        logger.warning(f"[HALLUCINATION GUARD] Failed to extract claims: {e}")
        # Fallback: extract sentences with numbers/dates manually
        return _fallback_extract_claims(report_text)


def _fallback_extract_claims(text: str) -> List[str]:
    """
    Fallback claim extraction using simple heuristics.
    Finds sentences containing numbers, percentages, or dates.
    """
    import re
    sentences = re.split(r'(?<=[.!?])\s+', text)
    claims = []

    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence or len(sentence) < 20:
            continue
        # Check if sentence contains numbers, percentages, or year-like patterns
        if re.search(r'\d+%|\$[\d,]+|\b\d{4}\b|\b\d+\.\d+\b|\bmillion\b|\bbillion\b|\btrillion\b', sentence):
            claims.append(sentence[:200])
            if len(claims) >= 10:
                break

    return claims


def verify_claims(claims: List[str], source_texts: List[str]) -> Tuple[List[Dict], List[Dict]]:
    """
    Verify each claim against the source texts using fuzzy matching.
    
    Returns:
        (verified_claims, unverified_claims) — each is a list of
        {claim, confidence, source_match} dicts.
    """
    verified = []
    unverified = []

    # Combine all source texts for matching
    combined_sources = "\n".join(source_texts).lower()

    for claim in claims:
        claim_lower = claim.lower()
        best_ratio = 0.0
        best_match = ""

        # Strategy 1: Direct substring check (strongest signal)
        if claim_lower in combined_sources:
            verified.append({
                "claim": claim,
                "confidence": 1.0,
                "method": "exact_match"
            })
            continue

        # Strategy 2: Check key fragments (numbers, named entities)
        import re
        key_fragments = re.findall(r'\d+%|\$[\d,]+|\b\d{4}\b|\b[A-Z][a-z]+(?:\s[A-Z][a-z]+)*\b', claim)
        fragment_hits = sum(1 for frag in key_fragments if frag.lower() in combined_sources)
        fragment_ratio = fragment_hits / max(len(key_fragments), 1)

        # Strategy 3: Fuzzy match against each source
        for source_text in source_texts:
            # Compare claim against sliding windows of source text
            source_lower = source_text.lower()
            words = source_lower.split()
            claim_words = claim_lower.split()
            window_size = len(claim_words) + 5

            for i in range(0, max(len(words) - window_size, 1)):
                window = " ".join(words[i:i + window_size])
                ratio = SequenceMatcher(None, claim_lower, window).ratio()
                if ratio > best_ratio:
                    best_ratio = ratio
                    best_match = window[:100]

        # Combined confidence from fragments + fuzzy matching
        combined_confidence = max(fragment_ratio * 0.7, best_ratio)

        if combined_confidence >= 0.45:
            verified.append({
                "claim": claim,
                "confidence": round(combined_confidence, 2),
                "method": "fuzzy_match"
            })
        else:
            unverified.append({
                "claim": claim,
                "confidence": round(combined_confidence, 2),
                "method": "no_match"
            })

    return verified, unverified


def generate_verification_report(verified: List[Dict], unverified: List[Dict]) -> str:
    """
    Generate a human-readable verification summary with icons.
    """
    total = len(verified) + len(unverified)
    if total == 0:
        return "\n\n---\n**🔬 Verification:** No specific claims detected to verify."

    confidence_pct = round((len(verified) / total) * 100)

    lines = [
        "\n\n---",
        f"**🔬 Claim Verification Summary** — Confidence: **{confidence_pct}%** ({len(verified)}/{total} claims verified)\n"
    ]

    if verified:
        for v in verified[:5]:
            lines.append(f"- ✅ {v['claim']}")

    if unverified:
        lines.append("")
        for u in unverified[:5]:
            lines.append(f"- ⚠️ {u['claim']} *(unverified — may be from AI training data)*")

    return "\n".join(lines)


def guard(report: str, search_results: List[Dict], llm) -> Dict:
    """
    Main entry point for the Hallucination Guard.
    
    Args:
        report: The AI-generated report text
        search_results: List of web search results [{title, url, snippet, content}]
        llm: The LangChain LLM instance
        
    Returns:
        {
            report: str (original report + verification summary appended),
            confidence_score: float (0.0 - 1.0),
            verification_summary: str,
            verified_count: int,
            unverified_count: int
        }
    """
    if not search_results:
        logger.info("[HALLUCINATION GUARD] No search results — skipping verification.")
        return {
            "report": report,
            "confidence_score": None,
            "verification_summary": "\n\n---\n**🔬 Verification:** Skipped (no web sources available).",
            "verified_count": 0,
            "unverified_count": 0
        }

    try:
        # 1. Extract claims from the report
        claims = extract_claims(report, llm)

        if not claims:
            return {
                "report": report,
                "confidence_score": 1.0,
                "verification_summary": "\n\n---\n**🔬 Verification:** No specific factual claims detected.",
                "verified_count": 0,
                "unverified_count": 0
            }

        # 2. Get source texts for comparison
        source_texts = [r.get("content", r.get("snippet", "")) for r in search_results]

        # 3. Verify claims against sources
        verified, unverified = verify_claims(claims, source_texts)

        # 4. Calculate confidence score
        total = len(verified) + len(unverified)
        confidence_score = len(verified) / total if total > 0 else 1.0

        # 5. Generate verification summary
        verification_summary = generate_verification_report(verified, unverified)

        # 6. Append summary to report
        enhanced_report = report + verification_summary

        logger.info(
            f"[HALLUCINATION GUARD] Confidence: {confidence_score:.0%} "
            f"({len(verified)} verified, {len(unverified)} unverified)"
        )

        return {
            "report": enhanced_report,
            "confidence_score": round(confidence_score, 2),
            "verification_summary": verification_summary,
            "verified_count": len(verified),
            "unverified_count": len(unverified)
        }

    except Exception as e:
        logger.error(f"[HALLUCINATION GUARD] Error: {e}")
        return {
            "report": report,
            "confidence_score": None,
            "verification_summary": f"\n\n---\n**🔬 Verification:** Error during verification ({e}).",
            "verified_count": 0,
            "unverified_count": 0
        }
