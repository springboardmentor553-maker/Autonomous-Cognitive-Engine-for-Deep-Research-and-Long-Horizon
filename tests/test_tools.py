

import json
import pytest

# ─── Milestone 1: Planning tool ───────────────────────────────────────────────

class TestWriteTodos:
    def test_creates_todos(self):
        from tools.planning.write_todos import write_todos
        result = write_todos.invoke({"tasks": ["Search for X", "Summarise findings", "Write report"]})
        data = json.loads(result)
        assert data["status"] == "todos_created"
        assert data["count"] == 3
        todos = data["todos"]
        assert todos[0]["id"] == "task_1"
        assert todos[0]["status"] == "pending"
        assert todos[2]["description"] == "Write report"

    def test_todos_have_all_fields(self):
        from tools.planning.write_todos import write_todos
        result = write_todos.invoke({"tasks": ["Do something"]})
        data = json.loads(result)
        todo = data["todos"][0]
        for field in ("id", "description", "status", "result"):
            assert field in todo, f"Missing field: {field}"

    def test_empty_task_list(self):
        from tools.planning.write_todos import write_todos
        result = write_todos.invoke({"tasks": []})
        data = json.loads(result)
        assert data["count"] == 0
        assert data["todos"] == []


# ─── Milestone 2: Virtual File System tools ───────────────────────────────────

class TestWriteFile:
    def test_returns_write_action(self):
        from tools.filesystem.write_file import write_file
        result = write_file.invoke({"path": "/test/notes.md", "content": "Hello world"})
        data = json.loads(result)
        assert data["action"] == "write_file"
        assert data["path"] == "/test/notes.md"
        assert data["content"] == "Hello world"

    def test_prepends_slash(self):
        from tools.filesystem.write_file import write_file
        result = write_file.invoke({"path": "no_slash.txt", "content": "x"})
        data = json.loads(result)
        assert data["path"].startswith("/")

    def test_has_timestamps(self):
        from tools.filesystem.write_file import write_file
        result = write_file.invoke({"path": "/t.txt", "content": "x"})
        data = json.loads(result)
        assert "created_at" in data
        assert "updated_at" in data


class TestReadFile:
    def setup_method(self):
        """Bind a fresh VFS to the read_file tool before each test."""
        from tools.filesystem import bind_vfs
        from utils.helpers import utc_now
        self.vfs = {
            "/notes/test.md": {"content": "Test content here", "created_at": utc_now(), "updated_at": utc_now()}
        }
        bind_vfs(self.vfs)

    def test_reads_existing_file(self):
        from tools.filesystem.read_file import read_file
        result = read_file.invoke({"path": "/notes/test.md"})
        data = json.loads(result)
        assert data["content"] == "Test content here"

    def test_missing_file_returns_error(self):
        from tools.filesystem.read_file import read_file
        result = read_file.invoke({"path": "/does/not/exist.md"})
        data = json.loads(result)
        assert "error" in data
        assert "available_files" in data

    def test_prepends_slash(self):
        from tools.filesystem.read_file import read_file
        result = read_file.invoke({"path": "notes/test.md"})
        data = json.loads(result)
        assert data.get("content") == "Test content here"


class TestEditFile:
    def test_returns_edit_action(self):
        from tools.filesystem.edit_file import edit_file
        result = edit_file.invoke({"path": "/f.md", "mode": "append", "content": "more text"})
        data = json.loads(result)
        assert data["action"] == "edit_file"
        assert data["mode"] == "append"

    def test_invalid_mode(self):
        from tools.filesystem.edit_file import edit_file
        result = edit_file.invoke({"path": "/f.md", "mode": "badmode", "content": "x"})
        data = json.loads(result)
        assert "error" in data


class TestLs:
    def setup_method(self):
        from tools.filesystem import bind_vfs
        from utils.helpers import utc_now
        self.vfs = {
            "/research/topic_a.md": {"content": "AAA", "updated_at": utc_now()},
            "/research/topic_b.md": {"content": "BBBB", "updated_at": utc_now()},
            "/summaries/combined.md": {"content": "CC", "updated_at": utc_now()},
        }
        bind_vfs(self.vfs)

    def test_lists_all_files(self):
        from tools.filesystem.ls import ls
        result = ls.invoke({"directory": "/"})
        data = json.loads(result)
        assert data["count"] == 3

    def test_lists_subdirectory(self):
        from tools.filesystem.ls import ls
        result = ls.invoke({"directory": "/research"})
        data = json.loads(result)
        assert data["count"] == 2
        paths = [f["path"] for f in data["files"]]
        assert all(p.startswith("/research/") for p in paths)

    def test_empty_directory(self):
        from tools.filesystem.ls import ls
        result = ls.invoke({"directory": "/nonexistent"})
        data = json.loads(result)
        assert data["count"] == 0


# ─── Helpers ──────────────────────────────────────────────────────────────────

class TestHelpers:
    def test_next_pending_todo(self):
        from utils.helpers import next_pending_todo
        todos = [
            {"id": "t1", "status": "completed", "description": "done"},
            {"id": "t2", "status": "pending", "description": "next"},
            {"id": "t3", "status": "pending", "description": "later"},
        ]
        result = next_pending_todo(todos)
        assert result["id"] == "t2"

    def test_all_todos_done(self):
        from utils.helpers import all_todos_done
        todos = [
            {"id": "t1", "status": "completed"},
            {"id": "t2", "status": "failed"},
        ]
        assert all_todos_done(todos) is True
        todos[0]["status"] = "pending"
        assert all_todos_done(todos) is False

    def test_mark_todo(self):
        from utils.helpers import mark_todo
        todos = [{"id": "t1", "status": "pending", "description": "x", "result": ""}]
        updated = mark_todo(todos, "t1", "completed", "done!")
        assert updated[0]["status"] == "completed"
        assert updated[0]["result"] == "done!"
        # Original unchanged
        assert todos[0]["status"] == "pending"