import time
from statistics import mean
from backend.core.graph_builder import run_agent
from backend.evaluation.metrics import evaluate
from backend.evaluation.test_cases import TEST_TASKS
def run_single_experiment(task_id: int, task: str):
    print("=" * 80)
    print(f"[TASK {task_id}] STARTING")
    print(f"GOAL: {task}")
    print("=" * 80)
    start_time = time.time()
    state = run_agent(task)
    score = evaluate(state)
    end_time = time.time()
    duration = round(end_time - start_time, 2)
    print("\n--- PLANNING OUTPUT (TODOs) ---")
    for t in state.completed:
        print(f"✔ {t}")
    print("\n--- AGENT MEMORY ---")
    for m in state.memory:
        print(f"- {m}")
    print("\n--- REFLECTION ---")
    if state.reflection:
        print(state.reflection)
    else:
        print("No explicit reflection generated.")
    print("\n--- METRICS ---")
    print(f"Score        : {round(score * 100, 2)}%")
    print(f"Time Taken  : {duration} seconds")
    print(f"Tasks Done  : {len(state.completed)}")
    print("=" * 80)
    print(f"[TASK {task_id}] COMPLETED\n")
    return score
def main():
    print("\n🧠 AUTONOMOUS COGNITIVE ENGINE")
    print("Deep Research & Long-Horizon Task Evaluation\n")
    scores = []
    success_threshold = 0.80
    required_successes = 8
    for idx, task in enumerate(TEST_TASKS, start=1):
        score = run_single_experiment(idx, task)
        scores.append(score)
    avg_score = mean(scores)
    successful_tasks = sum(1 for s in scores if s >= success_threshold)
    # print("\n" + "#" * 80)
    print("📊 FINAL EVALUATION SUMMARY \n")
    # print("#" * 80)
    print(f"Total Tasks Evaluated      : {len(TEST_TASKS)}")
    print(f"Accuracy       : {successful_tasks}")
    print(f"Average Accuracy           : {round(avg_score * 100, 2)}%")
    if successful_tasks >= required_successes:
        print("\n✅ RESULT: SYSTEM MEETS RESEARCH CRITERIA")
        # print("✔ Solved at least 8/10 complex tasks")
        # print("✔ Accuracy ≥ 80%")
    else:
        print("\n❌ RESULT: SYSTEM DOES NOT MEET CRITERIA")
        print("✘ Needs improvement in planning/execution")
    print("\n🔬 NOTE:")
    print(
        "This evaluation framework is designed to be reproducible.\n"
        "Replacing simulated tools with real LLM calls will produce\n"
        "publishable experimental results."
    )
if __name__ == "__main__":
    main()