from __future__ import annotations

import json
from pathlib import Path
import sys
from time import perf_counter

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from app.config import BENCHMARK_RUNS, REPORTS_DIR
from app.supervisor import Supervisor


TEST_PROMPTS = [
    "Research the role of AI agents in education and generate a final report.",
    "Summarize the benefits and challenges of renewable energy adoption.",
    "Research blockchain use cases in healthcare and provide a structured analysis.",
    "Compare online learning and classroom learning in a concise report.",
    "Analyze the impact of social media on student productivity.",
]


def main() -> None:
    supervisor = Supervisor()
    results = []

    for index in range(BENCHMARK_RUNS):
        prompt = TEST_PROMPTS[index % len(TEST_PROMPTS)]
        thread_id = f"benchmark-{index + 1}"
        started = perf_counter()
        run = supervisor.run(prompt, thread_id=thread_id)
        duration = round(perf_counter() - started, 2)

        results.append(
            {
                "run": index + 1,
                "thread_id": thread_id,
                "prompt": prompt,
                "duration_seconds": duration,
                "score": run["evaluation"]["score"],
                "passed": run["evaluation"]["passed"],
                "todo_count": len(run["todos"]),
                "file_count": len(run["files"]),
            }
        )

    output_path = Path(REPORTS_DIR) / "benchmark_results.json"
    output_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"Saved benchmark results to {output_path}")


if __name__ == "__main__":
    main()
