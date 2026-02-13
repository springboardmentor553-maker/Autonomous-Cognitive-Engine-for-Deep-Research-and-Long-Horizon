class PlanningTools:

    def write_todos(self, task):
        # Simple task decomposition logic
        todos = [
            f"Research about {task}",
            f"Analyze collected information about {task}",
            f"Generate summary report for {task}"
        ]

        return todos