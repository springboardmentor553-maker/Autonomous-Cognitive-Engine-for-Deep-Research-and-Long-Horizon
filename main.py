from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from workflow.flow import create_agent_executor

# Load environment variables
load_dotenv()

def main():
    agent = create_agent_executor()

    task = "Design a structured research plan for analyzing cybersecurity threats."

    print("\nTASK:")
    print(task)
    print("\nRunning agent...\n")

    # IMPORTANT: thread_id is REQUIRED because you use MemorySaver()
    result = agent.invoke(
        {
            "messages": [HumanMessage(content=task)]
        },
        {
            "configurable": {
                "thread_id": "main-run"
            }
        }
    )

    todos = result.get("todos", [])

    print("Generated TODOs:\n")

    for t in todos:
        print(f"{t['index']}. {t['description']}")

    print("\nDone ✅")


if __name__ == "__main__":
    main()
