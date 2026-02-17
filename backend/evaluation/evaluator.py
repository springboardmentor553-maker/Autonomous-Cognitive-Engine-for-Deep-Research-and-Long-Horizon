# backend/evaluation/evaluator.py

from statistics import mean


def summarize_results(structural_results, determinism_results, latencies, retries):

    summary = {
        "structural_success_rate": round(
            sum(structural_results) / len(structural_results) * 100, 2
        ),
        "determinism_rate": round(
            sum(determinism_results) / len(determinism_results) * 100, 2
        ),
        "average_latency": round(mean(latencies), 2),
        "average_retry_count": round(mean(retries), 2),
        "all_structural_tests_passed": all(structural_results),
    }

    return summary