from core.engine import CognitiveEngine


if __name__ == "__main__":

    engine = CognitiveEngine()

    user_task = input("Enter your research task: ")

    engine.run(user_task)