"""
tests/test_graph.py — Graph structure and router unit tests.

These tests do NOT call external APIs; they test routing logic and state
transitions using mock states.

Run with: pytest tests/test_graph.py -v
"""

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage


# ─── Router tests ─────────────────────────────────────────────────────────────

class TestRouters:
    def _make_state(self, **kwargs):
        base = {
            "messages": [],
            "todos": [],
            "virtual_fs": {},
            "current_task_id": None,
            "iteration": 0,
            "final_output": None,
            "user_request": "test request",
        }
        base.update(kwargs)
        return base

    def test_route_after_planner_with_tool_calls(self):
        from graph.router import route_after_planner
        ai_msg = AIMessage(content="", tool_calls=[{"name": "write_todos", "args": {"tasks": ["t1"]}, "id": "call_1"}])
        state = self._make_state(messages=[ai_msg])
        assert route_after_planner(state) == "tools"

    def test_route_after_planner_no_tool_calls(self):
        from graph.router import route_after_planner
        ai_msg = AIMessage(content="some response")
        state = self._make_state(messages=[ai_msg])
        assert route_after_planner(state) == "synthesiser"

    def test_route_after_tools_no_todos(self):
        from graph.router import route_after_tools
        state = self._make_state(todos=[])
        assert route_after_tools(state) == "executor"

    def test_route_after_tools_all_done(self):
        from graph.router import route_after_tools
        todos = [
            {"id": "t1", "status": "completed", "description": "x", "result": ""},
            {"id": "t2", "status": "failed", "description": "y", "result": ""},
        ]
        state = self._make_state(todos=todos)
        assert route_after_tools(state) == "synthesiser"

    def test_route_after_tools_pending_remain(self):
        from graph.router import route_after_tools
        todos = [
            {"id": "t1", "status": "completed", "description": "x", "result": ""},
            {"id": "t2", "status": "pending", "description": "y", "result": ""},
        ]
        state = self._make_state(todos=todos)
        assert route_after_tools(state) == "executor"

    def test_route_after_executor_tool_calls(self):
        from graph.router import route_after_executor
        ai_msg = AIMessage(content="", tool_calls=[{"name": "web_search", "args": {}, "id": "c1"}])
        state = self._make_state(messages=[ai_msg], iteration=1)
        assert route_after_executor(state) == "tools"

    def test_route_after_executor_no_tool_calls(self):
        from graph.router import route_after_executor
        ai_msg = AIMessage(content="Task complete.")
        state = self._make_state(messages=[ai_msg], iteration=1)
        assert route_after_executor(state) == "task_complete"

    def test_route_after_executor_iteration_limit(self):
        from graph.router import route_after_executor, MAX_ITERATIONS
        ai_msg = AIMessage(content="", tool_calls=[{"name": "web_search", "args": {}, "id": "c1"}])
        state = self._make_state(messages=[ai_msg], iteration=MAX_ITERATIONS)
        assert route_after_executor(state) == "synthesiser"

    def test_route_after_task_complete_more_tasks(self):
        from graph.router import route_after_task_complete
        todos = [
            {"id": "t1", "status": "completed", "description": "x", "result": ""},
            {"id": "t2", "status": "pending", "description": "y", "result": ""},
        ]
        state = self._make_state(todos=todos)
        assert route_after_task_complete(state) == "executor"

    def test_route_after_task_complete_all_done(self):
        from graph.router import route_after_task_complete
        todos = [{"id": "t1", "status": "completed", "description": "x", "result": ""}]
        state = self._make_state(todos=todos)
        assert route_after_task_complete(state) == "synthesiser"


# ─── VFS integration test (no LLM) ───────────────────────────────────────────

class TestVFSIntegration:
    """Test VFS write→read round-trip without any LLM calls."""

    def test_write_and_read_roundtrip(self):
        import json
        from tools.filesystem import bind_vfs, write_file, read_file
        from utils.helpers import utc_now

        vfs = {}
        bind_vfs(vfs)

        # Simulate what tools_node does when processing a write_file call
        write_result = write_file.invoke({"path": "/test/data.md", "content": "Hello VFS!"})
        parsed = json.loads(write_result)
        assert parsed["action"] == "write_file"

        # Manually apply the side effect (tools_node does this in the graph)
        vfs[parsed["path"]] = {
            "content": parsed["content"],
            "created_at": parsed["created_at"],
            "updated_at": parsed["updated_at"],
        }

        # Now read it back
        read_result = read_file.invoke({"path": "/test/data.md"})
        read_data = json.loads(read_result)
        assert read_data["content"] == "Hello VFS!"

    def test_edit_append(self):
        import json
        from tools.filesystem import bind_vfs, edit_file
        from utils.helpers import utc_now

        vfs = {"/notes.md": {"content": "Line 1", "updated_at": utc_now()}}
        bind_vfs(vfs)

        edit_result = edit_file.invoke({"path": "/notes.md", "mode": "append", "content": "\nLine 2"})
        parsed = json.loads(edit_result)
        assert parsed["action"] == "edit_file"

        # Simulate tools_node applying the append
        existing = vfs["/notes.md"]["content"]
        updated = existing + "\n" + parsed["content"]
        vfs["/notes.md"]["content"] = updated

        assert "Line 1" in vfs["/notes.md"]["content"]
        assert "Line 2" in vfs["/notes.md"]["content"]