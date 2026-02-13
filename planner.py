from agents.planning_agent import PlanningAgent


class Planner:

    def __init__(self):
        self.agent = PlanningAgent("Planning Agent")

    def generate_plan(self, task):
        return self.agent.create_plan(task)