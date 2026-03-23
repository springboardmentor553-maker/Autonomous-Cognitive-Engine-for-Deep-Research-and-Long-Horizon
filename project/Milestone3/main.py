from supervisor.supervisor_agent import supervisor_graph

def run(user_request: str):
    print(f"\n{'='*60}")
    print(f"USER REQUEST: {user_request}")
    print(f"{'='*60}\n")

    initial_state = {
        "messages": [{"role": "user", "content": user_request}],
        "todos": [],
        "files": {},
        "remaining_steps": 20,
    }

    result = supervisor_graph.invoke(initial_state)

    final_msg = result["messages"][-1]
    print("\n--- FINAL RESPONSE ---")
    print(final_msg.content)

    if result.get("files"):
        print("\n--- VIRTUAL FILE SYSTEM ---")
        for fname, content in result["files"].items():
            print(f"\n[{fname}]\n{content}")

    return result

if __name__ == "__main__":
    run(
        "Search the web for the benefits of LangGraph for building "
        "AI agents. Summarize your findings and write a short report."
    )