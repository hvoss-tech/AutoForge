"""Exports that run in the background and report their progress.

A slicer project of a big model takes seconds (more the first time a
printer's profiles are downloaded). The client starts the export, polls its
progress for the progress bar, and downloads the file once it's done. Tasks
live in memory: they are short and belong to this server process.
"""

from __future__ import annotations

import os
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable

# Finished tasks whose file was never fetched are dropped after this long.
TASK_TTL = 30 * 60


class TaskCancelled(Exception):
    """The client cancelled the export."""


@dataclass
class ExportTask:
    id: str
    filename: str
    progress: float = 0.0
    stage: str = "Starting"
    done: bool = False
    cancelled: bool = False
    error_status: int | None = None
    error: str | None = None
    path: str | None = None
    # What the export wants the user to know (e.g. the slicer project's
    # printer preset and nozzle warnings): {"printer", "notes", "warnings"}.
    info: dict[str, Any] = field(default_factory=dict)
    created: float = field(default_factory=time.monotonic)

    def report(self, fraction: float, stage: str) -> None:
        """Progress callback for the worker; stops it once cancelled."""
        if self.cancelled:
            raise TaskCancelled()
        # Never backwards: a stage may report a lower start than the last.
        self.progress = max(self.progress, min(float(fraction), 1.0))
        self.stage = stage

    def status(self) -> dict[str, Any]:
        return {
            "task_id": self.id,
            "progress": round(self.progress, 4),
            "stage": self.stage,
            "done": self.done,
            "error": self.error,
            "error_status": self.error_status,
            "cancelled": self.cancelled,
            "printer": self.info.get("printer"),
            "notes": list(self.info.get("notes") or []),
            "warnings": list(self.info.get("warnings") or []),
        }


_tasks: dict[str, ExportTask] = {}
_lock = threading.Lock()


def _remove_file(task: ExportTask) -> None:
    if task.path and os.path.exists(task.path):
        try:
            os.remove(task.path)
        except OSError:
            pass
    task.path = None


def _purge() -> None:
    now = time.monotonic()
    with _lock:
        stale = [t for t in _tasks.values() if t.done and now - t.created > TASK_TTL]
        for t in stale:
            del _tasks[t.id]
    for t in stale:
        _remove_file(t)


def start(filename: str, work: Callable[[ExportTask], str]) -> ExportTask:
    """Run ``work(task)`` in a thread; it returns the finished file's path.

    ``work`` raises ``TaskError`` for a failure the user should read (with
    an HTTP status), anything else is reported as an internal error."""
    _purge()
    task = ExportTask(id=uuid.uuid4().hex, filename=filename)
    with _lock:
        _tasks[task.id] = task

    def run() -> None:
        try:
            path = work(task)
            if task.cancelled:
                if path and os.path.exists(path):
                    os.remove(path)
                return
            task.path = path
            task.progress, task.stage = 1.0, "Done"
        except TaskCancelled:
            pass
        except TaskError as e:
            task.error_status, task.error = e.status, e.detail
        except Exception as e:  # noqa: BLE001 - shown to the user, logged by the caller
            task.error_status, task.error = 500, f"The export failed: {e}"
        finally:
            task.done = True

    threading.Thread(target=run, name=f"export-{task.id[:8]}", daemon=True).start()
    return task


class TaskError(Exception):
    def __init__(self, status: int, detail: str):
        super().__init__(detail)
        self.status = status
        self.detail = detail


def get(task_id: str) -> ExportTask | None:
    with _lock:
        return _tasks.get(task_id)


def cancel(task_id: str) -> bool:
    task = get(task_id)
    if task is None:
        return False
    task.cancelled = True
    if task.done:
        _remove_file(task)
    return True


def take_file(task_id: str) -> tuple[str, str] | None:
    """The finished file (path, download name), handed out once: the task is
    forgotten and the caller deletes the file after sending it."""
    with _lock:
        task = _tasks.get(task_id)
        if task is None or not task.done or task.path is None:
            return None
        del _tasks[task_id]
    return task.path, task.filename
