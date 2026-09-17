import os
import json
import uuid
import threading
from typing import Optional
from datetime import datetime, timezone
from ..models import JobStatus, OptimizationSettings
from ..helpers.gpu_memory import release_pipeline_result

TERMINAL_STATUSES = ("completed", "failed", "cancelled")


class OptimizationService:
    def __init__(self, checkpoints_dir: str = "checkpoints"):
        self._jobs: dict[str, JobStatus] = {}
        self._results: dict[str, JobStatus] = {}
        self._settings: dict[str, OptimizationSettings] = {}
        self._cancel_events: dict[str, threading.Event] = {}
        self._pause_events: dict[str, threading.Event] = {}
        self._pipeline_results: dict[str, dict] = {}  # stored between Run and Prune
        self._checkpoints_dir = checkpoints_dir
        self._lock = threading.RLock()
        self._load_history()

    def _history_file(self) -> str:
        return os.path.join(self._checkpoints_dir, "history.json")

    def _load_history(self):
        path = self._history_file()
        if os.path.exists(path):
            try:
                with open(path) as f:
                    data = json.load(f)
                for job_id, record in data.items():
                    record_status = record.get("status") if isinstance(record, dict) else None
                    if not record_status:
                        continue
                    js = JobStatus(**record_status)
                    self._results[job_id] = js
                    self._jobs[job_id] = js
                    record_settings = record.get("settings")
                    if isinstance(record_settings, dict):
                        try:
                            self._settings[job_id] = OptimizationSettings(**record_settings)
                        except ValueError:
                            # Recorded before settings were range-checked;
                            # the job's status is still worth keeping.
                            pass
            except (json.JSONDecodeError, IOError, KeyError):
                pass

    def _save_history(self):
        with self._lock:
            os.makedirs(self._checkpoints_dir, exist_ok=True)
            data = {}
            for job_id, js in self._results.items():
                record: dict = {"status": js.model_dump(by_alias=True)}
                if job_id in self._settings:
                    record["settings"] = self._settings[job_id].model_dump(by_alias=True)
                data[job_id] = record
            with open(self._history_file(), "w") as f:
                json.dump(data, f, indent=2)

    def set_pipeline_result(self, job_id: str, result: dict):
        """Record a finished run's pipeline state (so Prune can reuse it) and
        release every older one.

        A pipeline result pins its whole optimizer — and therefore hundreds
        of MB of VRAM — on the training device. Keeping one per job meant
        optimizing several images in a row accumulated all of them, which is
        what made VRAM climb run after run until a later run OOM'd. Only the
        newest result can still be pruned from the UI (the Pruning dialog
        always targets the current job), so everything before it is dropped.
        """
        with self._lock:
            stale = [(jid, res) for jid, res in self._pipeline_results.items() if jid != job_id]
            for jid, _res in stale:
                self._pipeline_results.pop(jid, None)
            previous = self._pipeline_results.get(job_id)
            self._pipeline_results[job_id] = result
        # Outside the lock: releasing touches the GPU and can block, and no
        # other call needs to wait for that to read an unrelated job.
        for _jid, res in stale:
            release_pipeline_result(res)
        if previous is not None and previous is not result:
            release_pipeline_result(previous)

    def get_pipeline_result(self, job_id: str) -> dict | None:
        with self._lock:
            return self._pipeline_results.get(job_id)

    def clear_pipeline_result(self, job_id: str):
        with self._lock:
            result = self._pipeline_results.pop(job_id, None)
        release_pipeline_result(result)

    def clear_all_pipeline_results(self):
        """Release every retained result — used when a new run is about to
        allocate, so the previous image's optimizer isn't still resident
        while the new one builds its own tensors."""
        with self._lock:
            results = list(self._pipeline_results.values())
            self._pipeline_results.clear()
        for result in results:
            release_pipeline_result(result)

    def pipeline_result_job_ids(self) -> list[str]:
        """Test/diagnostic view of what is still holding device memory."""
        with self._lock:
            return list(self._pipeline_results)

    def create_job(self, settings: dict, job_id: str | None = None) -> JobStatus:
        with self._lock:
            jid = job_id or str(uuid.uuid4())
            js = JobStatus(
                job_id=jid,
                status="pending",
                started_at=datetime.now(timezone.utc).isoformat(),
                total_iterations=settings.get("iterations", 6000),
            )
            self._jobs[jid] = js
            self._settings[jid] = OptimizationSettings(**settings)
            self._cancel_events[jid] = threading.Event()
            self._pause_events[jid] = threading.Event()
            # Event semantics: is_set()=True means paused, is_set()=False means running
            # Start NOT paused
        return js

    def get_job(self, job_id: str) -> JobStatus | None:
        with self._lock:
            return self._jobs.get(job_id)

    def _update_status_locked(self, job_id: str, status: str, **kwargs) -> JobStatus | None:
        """Caller must hold self._lock."""
        js = self._jobs.get(job_id)
        if not js:
            return None
        js.status = status
        for k, v in kwargs.items():
            setattr(js, k, v)
        if status in ("completed", "failed", "cancelled"):
            js.completed_at = datetime.now(timezone.utc).isoformat()
            self._results[job_id] = js
            self._save_history()
        return js

    def update_status(self, job_id: str, status: str, **kwargs) -> JobStatus | None:
        with self._lock:
            js = self._jobs.get(job_id)
            if js is not None and status == "running":
                # While a job is paused, progress callbacks must not flip it back
                # to "running" — that is what makes the UI toggle back to the
                # Pause button even though the pipeline is actually blocked.
                if js.status == "paused":
                    ev = self._pause_events.get(job_id)
                    if ev is not None and ev.is_set():
                        status = "paused"
                        # The report describes work finished *before* the
                        # pause: a callback already on its way when the user
                        # clicked can only land afterwards. Applying its
                        # progress/phase/counts made a paused job appear to
                        # keep advancing (the pruning overlay's phase label
                        # moving on after Pause), so drop them and keep what
                        # was on screen when the pause took effect.
                        kwargs = {}
            # A job that already reached a terminal state must stay there.
            # Cancelling a job races the background thread's own updates: its
            # first "running" update (fired right after thread.start()) could
            # resurrect a cancelled job, and an exception or export finishing
            # after the cancel could turn it into "failed"/"completed".
            if js is not None and js.status in TERMINAL_STATUSES:
                return js
            return self._update_status_locked(job_id, status, **kwargs)

    def save_job_result(self, job_id: str, result: JobStatus):
        with self._lock:
            self._results[job_id] = result
            self._save_history()

    def get_result(self, job_id: str) -> JobStatus | None:
        with self._lock:
            return self._results.get(job_id)

    def get_history(self) -> list[JobStatus]:
        with self._lock:
            jobs = sorted(self._results.values(), key=lambda j: j.started_at or "", reverse=True)
        return jobs

    def get_latest_job(self) -> JobStatus | None:
        """The most recently started *optimization* job, whatever its
        status — used to restore the frontend to where it was after a page
        reload (an in-progress job reconnects its WS, a completed one gets
        its 3D result back), since `_jobs` (unlike `_results`) also holds
        jobs that haven't reached a terminal state yet.

        Excludes pruning jobs (`prune-*`): those are a secondary tracking
        entry for progress only — pruning writes its output into the
        *original* optimization job's directory, not its own — so treating
        one as "the current job" would point the UI at a job_id with no
        real output directory.
        """
        with self._lock:
            candidates = [j for j in self._jobs.values() if not j.job_id.startswith("prune-")]
            if not candidates:
                return None
            return max(candidates, key=lambda j: j.started_at or "")

    def get_active_optimization_job(self) -> JobStatus | None:
        with self._lock:
            for j in self._jobs.values():
                if not j.job_id.startswith("prune-") and j.status in ("pending", "running", "paused"):
                    return j
            return None

    def cancel_event(self, job_id: str) -> threading.Event | None:
        with self._lock:
            return self._cancel_events.get(job_id)

    def pause_event(self, job_id: str) -> threading.Event | None:
        with self._lock:
            return self._pause_events.get(job_id)

    def _is_terminal_locked(self, job_id: str) -> bool:
        """Pause/resume/cancel on a job that already finished (e.g. the user
        clicked Cancel just as it completed) must be a no-op: otherwise a
        completed job was overwritten to "cancelled" in history, or flipped to
        "paused"/"running" with no thread left to ever finish it."""
        js = self._jobs.get(job_id)
        return js is not None and js.status in TERMINAL_STATUSES

    def pause(self, job_id: str) -> bool:
        with self._lock:
            ev = self._pause_events.get(job_id)
            if ev:
                if self._is_terminal_locked(job_id):
                    return True
                ev.set()  # is_set=True = paused
                self._update_status_locked(job_id, "paused")
                return True
            return False

    def resume(self, job_id: str) -> bool:
        with self._lock:
            ev = self._pause_events.get(job_id)
            if ev:
                if self._is_terminal_locked(job_id):
                    return True
                ev.clear()  # is_set=False = running
                self._update_status_locked(job_id, "running")
                return True
            return False

    def cancel(self, job_id: str) -> bool:
        with self._lock:
            ev = self._cancel_events.get(job_id)
            if ev:
                if self._is_terminal_locked(job_id):
                    return True
                ev.set()
                self._update_status_locked(job_id, "cancelled")
                return True
            return False

    def get_settings(self, job_id: str) -> OptimizationSettings | None:
        with self._lock:
            return self._settings.get(job_id)


_service: OptimizationService | None = None
_init_lock = threading.Lock()


def get_optimization_service() -> OptimizationService:
    global _service
    if _service is None:
        with _init_lock:
            if _service is None:
                # Was a hardcoded "checkpoints" literal — never read
                # config.checkpoints_path at all, unlike every other
                # service (filament_service, image_service,
                # project_service). That meant job history was never
                # actually redirected by AUTOFORGE_WEBUI_CHECKPOINTS_DIR
                # (e.g. tests/webui/conftest.py's isolation fixture), so
                # every /api/optimize/start call from the webui test suite
                # wrote real job records straight into the real, shared
                # checkpoints/history.json — confirmed in practice (it had
                # accumulated 28 fake "Input image not found: whatever.png"
                # entries from automated test runs).
                from ..config import config
                _service = OptimizationService(checkpoints_dir=config.checkpoints_path)
    return _service


def reset_service():
    """Reset singleton — for test isolation."""
    global _service
    _service = None
