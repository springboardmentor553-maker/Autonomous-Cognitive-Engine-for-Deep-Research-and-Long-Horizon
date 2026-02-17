from backend.graph.graph_builder import build_graph


class DeepCognitiveExecutor:

    def __init__(self):
        self.graph = build_graph()

    def run(self, request: str):

        result = self.graph.invoke({
            "user_request": request,
            "todos": [],
            "planning_meta": {
                "total_tasks": 0,
                "validated": False,
                "retry_count": 0,
                "validation_errors": []
            },
            "final_output": ""
        })

        return result