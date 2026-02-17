# backend/evaluation/experiment_runner.py

import time
from backend.runtime.executor import DeepCognitiveExecutor
from backend.evaluation.milestone1_tests import TEST_OBJECTIVES
from backend.evaluation.metrics import evaluate_structure, check_determinism
from backend.evaluation.evaluator import summarize_results


def run_milestone1_evaluation():

    executor = DeepCognitiveExecutor()

    structural_results = []
    determinism_results = []
    latencies = []
    retries = []

    print("\n==============================")
    print("Milestone 1 Evaluation Started")
    print("==============================")

    for idx, objective in enumerate(TEST_OBJECTIVES, start=1):

        print(f"\nTest {idx}: {objective[:70]}...")

        start_time = time.time()
        state = executor.run(objective)
        latency = time.time() - start_time

        checks = evaluate_structure(state)
        determinism = check_determinism(executor, objective)

        structural_results.append(all(checks.values()))
        determinism_results.append(determinism)
        latencies.append(latency)
        retries.append(state["planning_meta"]["retry_count"])

        print("Structural Checks:", checks)
        print("Deterministic:", determinism)
        print("Latency: {:.2f}s".format(latency))
        print("Retries:", state["planning_meta"]["retry_count"])

    summary = summarize_results(
        structural_results,
        determinism_results,
        latencies,
        retries
    )

    print("\n======================================")
    print("Milestone 1 Research Evaluation Summary")
    print("======================================")

    for key, value in summary.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    run_milestone1_evaluation()