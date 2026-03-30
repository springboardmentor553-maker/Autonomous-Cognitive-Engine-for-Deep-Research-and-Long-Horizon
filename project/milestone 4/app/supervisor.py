from __future__ import annotations

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph

from app.config import DEFAULT_THREAD_ID
from app.evaluator import evaluate_output
from app.executor import execute_next_todo
from app.planner import replan_todos, write_todos
from app.state import GraphState, create_initial_state
from app.synthesizer import synthesize_results


SYSTEM_PROMPT = """
You are a supervisor agent.

Always create TODO plans.
Delegate specialized tasks to sub-agents.
Use file tools for intermediate memory.
Re-plan when execution uncovers new work or failures.
Integrate all outputs before generating the final response.
""".strip()


def plan_node(state: GraphState) -> GraphState:
    if not state["todos"]:
        state["todos"] = write_todos(state["user_request"])
    return state


def execute_node(state: GraphState) -> GraphState:
    state["iteration"] += 1
    return execute_next_todo(state)


def replan_node(state: GraphState) -> GraphState:
    if state["queued_followups"]:
        new_todos = replan_todos(
            state["user_request"],
            state["todos"],
            state["queued_followups"],
        )
        state["todos"].extend(new_todos)
    state["queued_followups"] = []
    state["needs_replan"] = False
    return state


def synthesize_node(state: GraphState) -> GraphState:
    state["final_report"] = synthesize_results(state)
    return state


def evaluate_node(state: GraphState) -> GraphState:
    state["evaluation"] = evaluate_output(state["final_report"])
    return state


def route_after_execute(state: GraphState) -> str:
    if state["iteration"] >= state["max_iterations"]:
        return "synthesize"
    if state["needs_replan"]:
        return "replan"
    if any(todo["status"] == "pending" for todo in state["todos"]):
        return "execute"
    return "synthesize"


class Supervisor:
    def __init__(self) -> None:
        self.system_prompt = SYSTEM_PROMPT
        self.checkpointer = MemorySaver()

        workflow = StateGraph(GraphState)
        workflow.add_node("plan", plan_node)
        workflow.add_node("execute", execute_node)
        workflow.add_node("replan", replan_node)
        workflow.add_node("synthesize", synthesize_node)
        workflow.add_node("evaluate", evaluate_node)

        workflow.set_entry_point("plan")
        workflow.add_edge("plan", "execute")
        workflow.add_conditional_edges(
            "execute",
            route_after_execute,
            {
                "execute": "execute",
                "replan": "replan",
                "synthesize": "synthesize",
            },
        )
        workflow.add_edge("replan", "execute")
        workflow.add_edge("synthesize", "evaluate")
        workflow.add_edge("evaluate", END)

        self.graph = workflow.compile(checkpointer=self.checkpointer)

    def run(self, user_request: str, thread_id: str = DEFAULT_THREAD_ID) -> GraphState:
        return self.graph.invoke(
            create_initial_state(user_request, thread_id),
            config={"configurable": {"thread_id": thread_id}},
        )
