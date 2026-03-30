from uuid import uuid4

from app.config import DEFAULT_THREAD_ID
from app.supervisor import Supervisor


def main() -> None:
    user_request = input("Enter the user request: ").strip()
    if not user_request:
        print("A user request is required.")
        return

    supervisor = Supervisor()
    thread_id = f"{DEFAULT_THREAD_ID}-{uuid4().hex[:8]}"
    result = supervisor.run(user_request, thread_id=thread_id)

    print("\n=== TODO PLAN ===\n")
    for index, todo in enumerate(result["todos"], start=1):
        print(
            f"{index}. {todo['task']} | status={todo['status']} | "
            f"agent={todo.get('assigned_agent', 'n/a')} | file={todo.get('result_file', 'n/a')}"
        )

    print("\n=== FINAL REPORT ===\n")
    print(result["final_report"])

    print("\n=== EVALUATION ===\n")
    print(f"Score: {result['evaluation']['score']}/10")
    print(f"Passed: {result['evaluation']['passed']}")
    print(f"Summary: {result['evaluation']['summary']}")


if __name__ == "__main__":
    main()
