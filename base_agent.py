class BaseAgent:
    def __init__(self, name):
        self.name = name

    def think(self, task):
        return f"{self.name} thinking about {task}"

    def act(self, task):
        return f"{self.name} executing {task}"