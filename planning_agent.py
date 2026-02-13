from agents.base_agent import BaseAgent
from tools.planning_tools import PlanningTools


class PlanningAgent(BaseAgent):

    def __init__(self, name):
        super().__init__(name)
        self.planner = PlanningTools()

    def create_plan(self, task):
        return self.planner.write_todos(task)