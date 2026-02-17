class Executor:
    def __init__(self, tools):
        self.tools = tools
    def act(self, action: str, state):
        if action in self.tools:
            return self.tools[action](state)
        return f"No tool found for action: {action}"