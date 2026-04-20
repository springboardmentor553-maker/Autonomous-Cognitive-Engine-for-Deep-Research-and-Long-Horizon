"""
Garbage Collector — Periodic cleanup of stale tasks and orphaned virtual_fs files.
Runs as a daemon thread in the background.
"""
import os
import time
import threading
import logging
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)


class GarbageCollector:
    """
    Background garbage collector that cleans up:
    - In-memory completed/errored tasks older than max_task_age
    - Virtual FS files older than max_file_age
    """

    def __init__(self, tasks_dict: dict, fs_dir: Path,
                 max_task_age: int = 3600, max_file_age: int = 86400):
        """
        Args:
            tasks_dict: Reference to the in-memory tasks dictionary
            fs_dir: Path to the virtual_fs directory
            max_task_age: Max age in seconds for completed tasks (default: 1 hour)
            max_file_age: Max age in seconds for virtual_fs files (default: 24 hours)
        """
        self.tasks = tasks_dict
        self.fs_dir = Path(fs_dir)
        self.max_task_age = max_task_age
        self.max_file_age = max_file_age
        self._running = False
        self._thread = None
        self._stats = {
            "total_tasks_removed": 0,
            "total_files_removed": 0,
            "last_run": None,
            "run_count": 0
        }

    def collect_stale_tasks(self) -> int:
        """
        Remove completed/errored tasks older than max_task_age from memory.
        Returns the number of tasks removed.
        """
        now = time.time()
        stale_keys = []

        for task_id, task in list(self.tasks.items()):
            if task.status in ('complete', 'error'):
                # Check if task has a completed_at timestamp
                completed_at = getattr(task, 'completed_at', None)
                if completed_at:
                    age = now - completed_at
                    if age > self.max_task_age:
                        stale_keys.append(task_id)
                else:
                    # No timestamp — mark current time and skip for now
                    task.completed_at = now

        for key in stale_keys:
            del self.tasks[key]

        if stale_keys:
            logger.info(f"[GC] Removed {len(stale_keys)} stale tasks from memory: {stale_keys}")

        return len(stale_keys)

    def collect_orphan_files(self) -> int:
        """
        Delete virtual_fs files older than max_file_age.
        Returns the number of files removed.
        """
        if not self.fs_dir.exists():
            return 0

        now = time.time()
        removed = 0

        for file in self.fs_dir.iterdir():
            if not file.is_file():
                continue
            try:
                file_age = now - file.stat().st_mtime
                if file_age > self.max_file_age:
                    file.unlink()
                    removed += 1
                    logger.debug(f"[GC] Deleted old file: {file.name} (age: {file_age/3600:.1f}h)")
            except Exception as e:
                logger.warning(f"[GC] Failed to delete {file.name}: {e}")

        if removed:
            logger.info(f"[GC] Removed {removed} orphan files from virtual_fs/")

        return removed

    def run_collection(self) -> dict:
        """
        Run both collectors and return stats.
        """
        tasks_removed = self.collect_stale_tasks()
        files_removed = self.collect_orphan_files()

        self._stats["total_tasks_removed"] += tasks_removed
        self._stats["total_files_removed"] += files_removed
        self._stats["last_run"] = datetime.utcnow().isoformat()
        self._stats["run_count"] += 1

        logger.info(
            f"[GC] Collection complete — "
            f"tasks: {tasks_removed}, files: {files_removed}, "
            f"run #{self._stats['run_count']}"
        )

        return {
            "tasks_removed": tasks_removed,
            "files_removed": files_removed,
            "timestamp": self._stats["last_run"],
            "cumulative": {
                "total_tasks_removed": self._stats["total_tasks_removed"],
                "total_files_removed": self._stats["total_files_removed"],
                "run_count": self._stats["run_count"]
            }
        }

    def _gc_loop(self, interval: int):
        """Internal loop for the daemon thread."""
        logger.info(f"[GC] Daemon started — running every {interval}s")
        while self._running:
            time.sleep(interval)
            if self._running:
                try:
                    self.run_collection()
                except Exception as e:
                    logger.error(f"[GC] Error during collection: {e}")

    def start(self, interval: int = 1800):
        """
        Start garbage collection as a background daemon thread.
        Args:
            interval: Seconds between collection runs (default: 30 minutes)
        """
        if self._running:
            logger.warning("[GC] Already running.")
            return

        self._running = True
        self._thread = threading.Thread(
            target=self._gc_loop,
            args=(interval,),
            daemon=True,
            name="GarbageCollector"
        )
        self._thread.start()
        logger.info(f"[GC] Background daemon started (interval: {interval}s)")

    def stop(self):
        """Stop the daemon thread."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=5)
        logger.info("[GC] Daemon stopped.")

    def get_stats(self) -> dict:
        """Get cumulative GC statistics."""
        return {
            **self._stats,
            "active_tasks": len(self.tasks),
            "fs_files": len(list(self.fs_dir.iterdir())) if self.fs_dir.exists() else 0
        }
