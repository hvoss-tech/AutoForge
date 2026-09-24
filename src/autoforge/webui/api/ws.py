import json
import asyncio
import logging
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from ..services.optimization_service import get_optimization_service

logger = logging.getLogger(__name__)

router = APIRouter()

# Track active preview WebSocket connections
_preview_connections: set[WebSocket] = set()
_main_loop: asyncio.AbstractEventLoop | None = None


def _send_to_all(msg: str):
    """Safely send a message to all connected preview clients (thread-safe)."""
    global _main_loop
    # Iterate a copy: this runs on the optimizer's worker thread while the
    # event loop adds/discards connections, and iterating the live set then
    # raises "Set changed size during iteration", dropping the update.
    for ws in list(_preview_connections):
        try:
            if _main_loop and not _main_loop.is_closed():
                future = asyncio.run_coroutine_threadsafe(ws.send_text(msg), _main_loop)
                # run_coroutine_threadsafe only schedules the coroutine — it
                # returns immediately, so the try/except around it can never
                # observe a failed send() (that happens later, on the event
                # loop). Without this callback, a client that vanished
                # without a clean close (network drop, tab killed) was never
                # pruned from _preview_connections by this path: every future
                # broadcast kept scheduling a doomed send to it forever.
                future.add_done_callback(lambda f, ws=ws: _drop_on_failure(f, ws))
        except Exception:
            _preview_connections.discard(ws)


def _drop_on_failure(future: "asyncio.Future", ws: WebSocket) -> None:
    """Runs on the event-loop thread (add_done_callback's contract), so
    mutating the set directly here is safe."""
    if future.cancelled() or future.exception() is not None:
        _preview_connections.discard(ws)


def broadcast_preview(
    image_data: str,
    job_id: str,
    sliders: list | None = None,
    iteration: int = 0,
    loss: float | None = None,
    min_layer: int | None = None,
    max_layer: int | None = None,
    render_id: str | None = None,
):
    """Send a preview update to all connected preview clients (thread-safe).

    Every connected browser tab shares this one channel, so `job_id` lets a
    client that isn't watching this job (a different tab, or a page loaded
    after this job started) ignore the update instead of silently picking up
    an unrelated job's colors/progress.
    """
    payload = {
        "type": "preview_update",
        "job_id": job_id,
        "image": image_data,
        "sliders": sliders or [],
        "iteration": iteration,
        "loss": loss,
    }
    if min_layer is not None:
        payload["min_layer"] = min_layer
    if max_layer is not None:
        payload["max_layer"] = max_layer
    if render_id:
        payload["render_id"] = str(render_id)
    _send_to_all(json.dumps(payload))


@router.websocket("/ws/optimize/{job_id}")
async def ws_optimize(websocket: WebSocket, job_id: str):
    await websocket.accept()
    svc = get_optimization_service()
    sent_terminal = False
    try:
        while True:
            job = svc.get_job(job_id)
            if not job:
                await websocket.send_json({"job_id": job_id, "status": "not_found", "error": "Job not found"})
                break
            # The base64 preview PNG rides only on /ws/preview, not here:
            # this socket polls every 0.5 s for every connected client.
            payload = job.model_dump()
            payload.pop("preview_image", None)
            await websocket.send_json(payload)
            if job.status in ("completed", "failed", "cancelled"):
                sent_terminal = True
                break
            await asyncio.sleep(0.5)
    except WebSocketDisconnect:
        pass
    finally:
        if not sent_terminal:
            pass


@router.websocket("/ws/preview")
async def ws_preview(websocket: WebSocket):
    global _main_loop
    _main_loop = asyncio.get_event_loop()
    await websocket.accept()
    _preview_connections.add(websocket)
    try:
        while True:
            data = await websocket.receive_text()
    except WebSocketDisconnect:
        pass
    finally:
        _preview_connections.discard(websocket)
