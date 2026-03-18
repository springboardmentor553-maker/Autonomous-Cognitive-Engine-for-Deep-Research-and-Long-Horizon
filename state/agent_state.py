"""
state/agent_state.py — Shared LangGraph state schema.

Milestone 1 : messages, todos, current_task_id, iteration, final_output, user_request
Milestone 2 : virtual_fs
Milestone 3 : delegation_log  (every sub-agent call recorded for LangSmith)
"""

from __future__ import annotations
from typing import Annotated
from typing_extensions import TypedDict
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


class TodoItem(TypedDict):
    id:           str
    description:  str
    status:       str   # "pending"|"in_progress"|"completed"|"delegated"|"failed"
    result:       str
    delegated_to: str   # sub-agent name if delegated, else ""


class VFSFile(TypedDict):
    content:    str
    created_at: str
    updated_at: str


class DelegationRecord(TypedDict):
    """
    One record per delegate_task call.
    LangSmith reads these to verify:
      - supervisor correctly called the task tool
      - correct sub-agent was chosen
      - result was returned and integrated
    """
    task_id:      str   # which TODO triggered the delegation
    agent_name:   str   # "summarizer" or "web_searcher"
    input_data:   str   # what was sent to the sub-agent
    result:       str   # what the sub-agent returned
    status:       str   # "completed" or "failed"
    delegated_at: str   # ISO timestamp


class AgentState(TypedDict):
    # Milestone 1
    messages:        Annotated[list[BaseMessage], add_messages]
    todos:           list[TodoItem]
    current_task_id: str | None
    iteration:       int
    final_output:    str | None
    user_request:    str

    # Milestone 2
    virtual_fs: dict[str, VFSFile]

    # Milestone 3
    delegation_log: list[DelegationRecord]