"""Background PLY writes for live slider edits.

A live edit reaches the 3D view as vertex colors; the colored PLY on disk
only has to catch up so a reload/undo/another tab shows the edit. Building
and exporting it is by far the slowest part of an edit (~150ms and ~25MB at
default settings), so it runs here, off the request, and only for the
*latest* edit of each target once edits pause for a moment — dragging a
slider no longer builds a mesh per intermediate position.
"""

import logging
import threading
import time

logger = logging.getLogger(__name__)

_PERSIST_QUIET_S = 0.25
_persist_lock = threading.Lock()
_persist: dict[str, dict] = {}


def schedule_persist(target: str, write) -> None:
    with _persist_lock:
        state = _persist.setdefault(target, {"pending": None, "running": False, "idle": threading.Event()})
        state["pending"] = write
        state["at"] = time.monotonic()
        state["idle"].clear()
        if state["running"]:
            return
        state["running"] = True
    threading.Thread(target=_persist_loop, args=(target,), daemon=True).start()


def _persist_loop(target: str) -> None:
    with _persist_lock:
        state = _persist[target]
    while True:
        # Wait for a pause in the edits: while a slider is being dragged,
        # every intermediate position would otherwise get its own mesh.
        while True:
            with _persist_lock:
                quiet_for = time.monotonic() - state.get("at", 0.0)
            if quiet_for >= _PERSIST_QUIET_S:
                break
            time.sleep(_PERSIST_QUIET_S - quiet_for)
        with _persist_lock:
            write, state["pending"] = state["pending"], None
            if write is None:
                state["running"] = False
                state["idle"].set()
                return
        try:
            write()
        except Exception:
            logger.exception("Writing the slider-edited mesh for %s failed", target)


def wait_for_pending_mesh(target: str, timeout: float = 15.0) -> None:
    """Block until ``target``'s latest live edit is on disk (a no-op when
    nothing is pending). The mesh endpoints call this, so a reload never
    serves a mesh from before the last edit."""
    with _persist_lock:
        state = _persist.get(target)
        event = state["idle"] if state is not None and state["running"] else None
    if event is not None:
        event.wait(timeout)




def cancel_pending_mesh(target: str) -> None:
    """Forget ``target``'s not-yet-started write (its result is being thrown
    away, e.g. the auto-preview of an image that was just replaced). A write
    already running still finishes; wait_for_pending_mesh waits for it."""
    with _persist_lock:
        state = _persist.get(target)
        if state is not None:
            state["pending"] = None
