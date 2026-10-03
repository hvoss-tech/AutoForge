"""The background threads that run GPU jobs (optimization, pruning).

A job is marked "cancelled" the moment the user asks, but its thread only
stops at its next check (a training step, a pruning phase, the end of the
height estimate or of an export) - seconds later, sometimes much more. The
job-status mutex (``get_any_active_job``) already let the next run start in
that window, putting two jobs on the GPU at once; the new run's
``clear_all_pipeline_results`` then also freed the optimizer the old thread
was still using. Starts check here too, until the old thread is gone.
"""

import threading

_lock = threading.Lock()
_threads: dict[str, threading.Thread] = {}


def start_worker(job_id: str, target) -> threading.Thread:
    thread = threading.Thread(target=target, daemon=True, name=f"autoforge-job-{job_id}")
    # Registered before it starts: registered after, running_worker() could
    # miss a thread that was already running.
    with _lock:
        _threads[job_id] = thread
    try:
        thread.start()
    except BaseException:
        with _lock:
            _threads.pop(job_id, None)
        raise
    return thread


def running_worker() -> str | None:
    """The id of a job whose thread is still running, or None."""
    with _lock:
        for job_id, thread in list(_threads.items()):
            if thread.is_alive() or thread.ident is None:  # running, or about to start
                return job_id
            del _threads[job_id]
    return None


def wait_until_idle(timeout: float) -> str | None:
    """Wait up to ``timeout`` seconds for every job thread to end (a finished
    job's thread still frees GPU memory for a moment after its status says
    "completed"). Returns the id of one still running, or None."""
    import time

    deadline = time.monotonic() + timeout
    while True:
        job_id = running_worker()
        if job_id is None:
            return None
        with _lock:
            thread = _threads.get(job_id)
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return job_id
        if thread is not None:
            thread.join(min(remaining, 0.1))


STILL_STOPPING = "The previous run is still stopping. Try again in a moment."
