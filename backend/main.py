from backend.core.graph_builder import build_execution_graph

if __name__ == "__main__":

    agent = build_execution_graph()

    while True:
        user_input = input("\nEnter complex objective (or 'exit'): ")

        if user_input.lower() == "exit":
            break

        agent(user_input)