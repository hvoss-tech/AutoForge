import os
import json
import uuid
import threading
from typing import Optional
from datetime import datetime, timezone
from ..models import JobStatus, OptimizationSettings
from ..helpers.gpu_memory import release_pipeline_result

TERMINAL_STATUSES = ("completed", "failed", "cancelled")

# Key in a pipeline result dict naming the job whose solution its optimizer
# currently holds — see get_pipeline_result.
OWNER_KEY = "_owner_job_id"

# Cap on retained *terminal* job records (mirrors ProjectService.MAX_SNAPSHOTS).
# Without this, _results/history.json grew by one entry per finished job
# forever, and _save_history() rewrites the whole file on every completion.
MAX_HISTORY_JOBS = 200


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
        if not os.path.exists(path):
            return
        try:
            with open(path) as f:
                data = json.load(f)
        except (json.JSONDecodeError, IOError, ValueError):
            return
        if not isinstance(data, dict):
            # A non-object top level (e.g. an old-version file that was a
            # bare list) used to raise AttributeError here and take the
            # whole API down with it.
            return
        for job_id, record in data.items():
            # Each record is validated independently: one bad record (an
            # old version's field types, a truncated write) must not
            # prevent the rest from loading — and must not crash
            # OptimizationService.__init__, which would 500 every route.
            try:
                record_status = record.get("status") if isinstance(record, dict) else None
                if not isinstance(record_status, dict) or not record_status.get("status"):
                    continue
                js = JobStatus(**record_status)
            except (ValueError, TypeError, KeyError):
                # ValidationError is a ValueError; a non-dict `status` a
                # TypeError. Keep whatever else is healthy.
                continue
            self._results[job_id] = js
            self._jobs[job_id] = js
            record_settings = record.get("settings") if isinstance(record, dict) else None
            if isinstance(record_settings, dict):
                try:
                    self._settings[job_id] = OptimizationSettings(**record_settings)
                except ValueError:
                    # Recorded before settings were range-checked;
                    # the job's status is still worth keeping.
                    pass

    def _save_history(self):
        with self._lock:
            os.makedirs(self._checkpoints_dir, exist_ok=True)
            data = {}
            for job_id, js in self._results.items():
                # preview_image is a full base64 PNG: it belongs on the live
                # WS channel, not in a file that every startup re-reads and
                # that grows by one image per finished job, forever.
                record: dict = {"status": js.model_dump(by_alias=True, exclude={"preview_image"})}
                if job_id in self._settings:
                    record["settings"] = self._settings[job_id].model_dump(by_alias=True)
                data[job_id] = record
            tmp_path = self._history_file() + ".tmp"
            with open(tmp_path, "w") as f:
                json.dump(data, f, indent=2)
            os.replace(tmp_path, self._history_file())

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
            result[OWNER_KEY] = job_id
        # Outside the lock: releasing touches the GPU and can block, and no
        # other call needs to wait for that to read an unrelated job.
        for _jid, res in stale:
            release_pipeline_result(res)
        if previous is not None and previous is not result:
            release_pipeline_result(previous)

    def get_pipeline_result(self, job_id: str) -> dict | None:
        """The live pipeline result for ``job_id`` — only while its
        optimizer still holds *that job's* solution.

        Pruning mutates the one optimizer in place and then aliases it under
        the prune job's id, so the job it started from kept resolving to the
        same, now-pruned optimizer: undoing back to the unpruned result and
        editing a color re-rendered the *pruned* heights into the unpruned
        job's folder, its slider/base lookups answered with pruned data, and
        pruning it "again" started from the already-pruned solution. Once
        another job has claimed the optimizer (see claim_pipeline_result),
        the older id gets nothing, exactly like a result lost to a restart.
        """
        with self._lock:
            result = self._pipeline_results.get(job_id)
            if result and result.get(OWNER_KEY, job_id) != job_id:
                return None
            return result

    def claim_pipeline_result(self, new_owner_job_id: str, job_id: str) -> dict | None:
        """Hand ``job_id``'s result over to ``new_owner_job_id`` before
        mutating it (pruning). Returns the result, or None when ``job_id``
        doesn't currently own one."""
        with self._lock:
            result = self.get_pipeline_result(job_id)
            if result is not None:
                result[OWNER_KEY] = new_owner_job_id
            return result

    def alias_pipeline_result(self, new_job_id: str, existing_job_id: str) -> None:
        """Make ``new_job_id`` resolve to the same live pipeline result as
        ``existing_job_id``, without touching anything else.

        Used when a successful prune clones its output into its own job id
        (see api/pruning.py) — that job needs `get_pipeline_result` to find
        the same (still-mutating) optimizer for a follow-up prune or a
        slider-derivation call, but it must not go through
        `set_pipeline_result`: that method's "release every older result"
        cleanup would find the *existing* entry under a different key and
        release it — including calling `release_cuda_graph()` on the
        optimizer this new entry is about to alias — destroying the very
        result being registered. Both keys pointing at the same dict is
        safe: `release_pipeline_result` empties the dict in place, so
        whichever key is released first (by a later, real
        `set_pipeline_result` call) leaves the dict falsy and every other
        alias sees a no-op instead of a double-release.
        """
        with self._lock:
            result = self._pipeline_results.get(existing_job_id)
            if result is not None:
                self._pipeline_results[new_job_id] = result
                result[OWNER_KEY] = new_job_id

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
        # The auto-preview (api/init.py) optimizer pins its own GPU memory
        # independent of any optimization job's pipeline results — without
        # this, starting a run right after uploading an image kept the
        # preview's optimizer resident on the GPU for the whole training run.
        # Imported lazily: api.init imports this module at load time, so a
        # top-level import here would be circular.
        from ..api.init import release_init_result

        release_init_result()

    def pipeline_result_job_ids(self) -> list[str]:
        """Test/diagnostic view of what is still holding device memory."""
        with self._lock:
            return list(self._pipeline_results)

    def create_job(self, settings: dict, job_id: str | None = None) -> JobStatus:
        with self._lock:
            jid = job_id or str(uuid.uuid4())
            # input_image/flatforge ride along so a later page reload (or
            # another tab) can tell which result belongs to which image, and
            # which output shape the job produced — see JobStatus.
            js = JobStatus(
                job_id=jid,
                status="pending",
                started_at=datetime.now(timezone.utc).isoformat(),
                total_iterations=settings.get("iterations", 6000),
                input_image=settings.get("input_image") or None,
                flatforge=bool(settings.get("flatforge")) or None,
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
            # A pruning job (`prune-*`) that only reached "failed"/
            # "cancelled" never got its own output directory (api/pruning.py
            # only clones one on a *successful* pass) — recording it would
            # add a history row with no real result behind it. A
            # "completed" one is different now: it was cloned into its own
            # checkpoints/<prune_job_id> directory with a real pipeline
            # result aliased onto it (see alias_pipeline_result), making it
            # exactly as real a result as an optimization job — leaving it
            # out of history/`get_latest_job` was what let a history step
            # (or an F5 reload) that pointed at a *pruned* result fall back
            # to the job's original, unpruned files instead.
            if not job_id.startswith("prune-") or status == "completed":
                self._results[job_id] = js
                self._prune_history_locked()
                self._save_history()
        return js

    def _prune_history_locked(self) -> None:
        """Caller must hold self._lock. Evict the oldest terminal job
        records once the retained count exceeds MAX_HISTORY_JOBS, dropping
        their bookkeeping (settings/cancel/pause events) too so a
        long-running server doesn't accumulate these forever."""
        if len(self._results) <= MAX_HISTORY_JOBS:
            return
        oldest = sorted(self._results.values(), key=lambda j: j.started_at or "")
        overflow = len(self._results) - MAX_HISTORY_JOBS
        for js in oldest[:overflow]:
            jid = js.job_id
            self._results.pop(jid, None)
            self._jobs.pop(jid, None)
            self._settings.pop(jid, None)
            self._cancel_events.pop(jid, None)
            self._pause_events.pop(jid, None)

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

        Excludes *unfinished* pruning jobs (`prune-*`): those are a
        secondary tracking entry for progress only, with no output
        directory of their own until the prune actually succeeds. A
        *completed* prune job is different — it was cloned into its own
        checkpoints/<prune_job_id> directory (see api/pruning.py) and is a
        real, independent result, so it is included here: without this, a
        page reload after pruning fell back to the job's original,
        unpruned files instead of the pruned ones actually on screen.
        """
        with self._lock:
            candidates = [
                j for j in self._jobs.values()
                if not j.job_id.startswith("prune-") or j.status == "completed"
            ]
            if not candidates:
                return None
            return max(candidates, key=lambda j: j.started_at or "")

    def get_active_optimization_job(self) -> JobStatus | None:
        with self._lock:
            for j in self._jobs.values():
                if not j.job_id.startswith("prune-") and j.status in ("pending", "running", "paused"):
                    return j
            return None

    def get_any_active_job(self) -> JobStatus | None:
        """Any non-terminal job, pruning included.

        Pruning and optimization both drive the same GPU (and pruning drives
        the *same* FilamentOptimizer a later optimization would clear out),
        so one of either kind must gate the other. The optimization-only
        variant above deliberately ignores prune jobs for the "which result
        am I looking at" questions; this one is for the GPU mutex."""
        with self._lock:
            for j in self._jobs.values():
                if j.status in ("pending", "running", "paused"):
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
