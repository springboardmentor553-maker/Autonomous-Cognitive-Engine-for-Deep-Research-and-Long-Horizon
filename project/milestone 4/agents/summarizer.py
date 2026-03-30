from __future__ import annotations

from typing import TypedDict

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition

from agents.base import BaseAgent
from app.models import LLMClient
from app.state import GraphState
from tools.storage_tools import build_storage_tools


class AgentState(TypedDict):
    messages: list


class SummarizerAgent(BaseAgent):
    def _compile_graph(self, shared_state: GraphState):
        tools = build_storage_tools(shared_state)
        llm = LLMClient().bind_tools(tools)
        tool_node = ToolNode(tools)

        def call_model(state: AgentState) -> AgentState:
            response = llm.invoke(state["messages"])
            return {"messages": state["messages"] + [response]}

        workflow = StateGraph(AgentState)
        workflow.add_node("agent", call_model)
        workflow.add_node("tools", tool_node)
        workflow.set_entry_point("agent")
        workflow.add_conditional_edges("agent", tools_condition, {"tools": "tools", "__end__": END})
        workflow.add_edge("tools", "agent")
        return workflow.compile()

    def run(self, task_text: str, shared_state: GraphState) -> str:
        graph = self._compile_graph(shared_state)
        result = graph.invoke(
            {
                "messages": [
                    SystemMessage(
                        content=(
                            "You are a summarizer sub-agent graph. "
                            "Use tools when they help, especially list_files_tool and read_file_tool, "
                            "and return a concise, polished summary."
                        )
                    ),
                    HumanMessage(content=task_text),
                ]
            }
        )
        return str(result["messages"][-1].content).strip()
