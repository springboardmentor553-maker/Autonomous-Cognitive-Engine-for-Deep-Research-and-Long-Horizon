# backend/evaluation/metrics.py

def evaluate_structure(state: dict) -> dict:
    todos = state["todos"]
    meta = state["planning_meta"]

    return {
        "task_count_correct": len(todos) == 6,
        "ids_sequential": all(t["id"] == i + 1 for i, t in enumerate(todos)),
        "descriptions_nonempty": all(len(t["description"].strip()) > 15 for t in todos),
        "validated_flag": meta["validated"] is True,
        "retry_reasonable": meta["retry_count"] <= 3,
    }


def check_determinism(executor, objective: str) -> bool:
    outputs = []

    for _ in range(3):
        state = executor.run(objective)
        titles = [t["title"] for t in state["todos"]]
        outputs.append(titles)

    return outputs[0] == outputs[1] == outputs[2]