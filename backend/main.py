from backend.runtime.executor import DeepCognitiveExecutor


def main():

    executor = DeepCognitiveExecutor()

    while True:
        request = input("Enter complex objective (or 'exit'): ")

        if request == "exit":
            break

        result = executor.run(request)

        print(result["final_output"])


if __name__ == "__main__":
    main()