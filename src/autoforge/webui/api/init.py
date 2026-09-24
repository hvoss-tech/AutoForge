"""Auto-preview: as soon as an image + at least one active filament are
present, run just the heightmap-initialization algorithm (no gradient
descent) and show the result as a real 3D mesh — draped with the original
photo, not a trained material guess — so the user can start assigning
filament colors to layer bands manually via the color sliders before ever
clicking Run.
"""

import asyncio
import base64
import logging
import os
import threading
from typing import Any, Optional

import cv2
import numpy as np
from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

from ..config import config
from ..models import OptimizationSettings
from ..services.filament_service import get_filament_service
from ..services.optimization_service import get_optimization_service
from ..services.image_service import get_image_service
from ..helpers.pipeline_runner import build_init_preview, friendly_error_message
from ..helpers.colored_mesh import generate_colored_preview_mesh
from ..helpers.gpu_memory import release_pipeline_result
from ..helpers.mesh_persist import cancel_pending_mesh, wait_for_pending_mesh
from ..helpers.sliders import derive_base_from_result, derive_layer_range_from_result

logger = logging.getLogger(__name__)
router = APIRouter()

_state: dict[str, Any] = {"status": "idle", "preview_image": None, "error": None, "range": None}
_pipeline_result: Optional[dict[str, Any]] = None
_lock = threading.RLock()

# The build currently in flight (or None). ``generation`` distinguishes it
# from every build that started later: when a reset lands mid-build, the
# build's completion handler must NOT publish its result (it belongs to the
# image that was just replaced) — and the next /api/init/run must not start
# a *second* build on top of it, which is what OOM'd when switching between
# images on a nearly-full GPU.
_in_flight: Optional[dict[str, Any]] = None
_generation: int = 0


def _init_dir() -> str:
    return os.path.join(config.checkpoints_path, "_init")


def _mesh_path() -> str:
    # Same filename render_with_sliders() (helpers/slider_render.py) writes
    # when a slider edit re-renders the auto-preview mesh (api/preview.py's
    # INIT_JOB_SENTINEL path) — both write into checkpoints/_init/, and
    # GET /api/init/mesh always serves whichever one is newest.
    return os.path.join(_init_dir(), "final_model_colored.ply")


def get_init_pipeline_result() -> Optional[dict[str, Any]]:
    """Read access for other routers (preview/sliders) that fall back to
    the auto-preview state when there's no completed optimization job yet."""
    with _lock:
        return _pipeline_result


def release_init_result() -> None:
    """Release the auto-preview's retained optimizer (its GPU memory)
    without disturbing anything else about its displayed state.

    Called when a real optimization run is about to start: the auto-preview
    optimizer built by /api/init/run otherwise stays resident on the GPU for
    the whole training run that follows, on top of whatever
    clear_all_pipeline_results() already released."""
    global _pipeline_result
    with _lock:
        if _state["status"] == "initializing":
            # A build is in flight; it isn't safe to rip its result out from
            # under it here, and run_init's own GPU-mutex check (above)
            # already stops a new optimization/pruning job from starting
            # while a build is running.
            return
        result, _pipeline_result = _pipeline_result, None
    release_pipeline_result(result)


def _run_init_sync(input_image_path: str, filament_dicts: list[dict], settings_dict: dict) -> dict:
    """Runs on a worker thread (see asyncio.to_thread below)."""
    output_dir = os.path.join(config.checkpoints_path, "_init_build")
    result = build_init_preview(
        input_image_path=input_image_path,
        active_filaments=filament_dicts,
        output_dir=output_dir,
        settings=settings_dict,
    )

    optimizer = result["optimizer"]
    args = result["args"]
    _disc_global, disc_height_image = optimizer.get_discretized_solution(best=True)
    height_map_mm = disc_height_image.detach().cpu().numpy().astype(np.float32) * float(args.layer_height)

    color_image = np.ascontiguousarray(result["processing_img_np"])
    alpha_proc = result.get("alpha_proc")
    alpha_np = alpha_proc.detach().cpu().numpy() if alpha_proc is not None else None

    mesh = generate_colored_preview_mesh(
        height_map=height_map_mm,
        color_image=color_image,
        background_height=float(args.background_height),
        maximum_x_y_size=float(args.stl_output_size),
        alpha_mask=alpha_np,
    )
    os.makedirs(_init_dir(), exist_ok=True)
    # Temp file + rename, so the 3D view never downloads a half-written mesh.
    tmp_mesh = f"{_mesh_path()}.{os.getpid()}.{threading.get_ident()}.tmp.ply"
    mesh.export(tmp_mesh, encoding="binary")
    os.replace(tmp_mesh, _mesh_path())

    # Flat PNG kept too — /api/init/preview predates the 3D mesh and other
    # code (e.g. a text-only client) may still just want an image.
    preview_bgr = cv2.cvtColor(color_image, cv2.COLOR_RGB2BGR)
    _ok, buf = cv2.imencode(".png", preview_bgr)
    preview_b64 = base64.b64encode(buf.tobytes()).decode("utf-8") if _ok else None

    range_info = derive_layer_range_from_result(result) or {"min_layer": 0, "max_layer": int(args.max_layers)}

    return {
        "result": result,
        "preview_b64": preview_b64,
        "range": range_info,
        "base": derive_base_from_result(result),
    }


@router.post("/run")
async def run_init(settings: OptimizationSettings):
    global _state, _pipeline_result, _in_flight, _generation

    with _lock:
        if _state["status"] == "initializing":
            # A build is in flight. The reset that replaced it (if any)
            # keeps the status "initializing" until that build's worker
            # finishes, so a new build cannot start a second, concurrent
            # GPU job on top of it.
            raise HTTPException(400, "Already initializing")
        # The auto-preview build and a real optimization/pruning run all
        # drive the same GPU (optimization/pruning already gate each other
        # via the same check) — without this, uploading a new image while a
        # run is in progress started a second, concurrent GPU job.
        busy = get_optimization_service().get_any_active_job()
        if busy is not None:
            raise HTTPException(
                409,
                f"An optimization is {busy.status}. Let it finish or cancel it before starting a new preview.",
            )
        _state = {"status": "initializing", "preview_image": None, "error": None}
        superseded, _pipeline_result = _pipeline_result, None
        my_gen = _generation
        _in_flight = {"event": threading.Event(), "generation": my_gen}

    # The previous image's init optimizer is dead the moment this one starts
    # — and releasing it *before* building the new one halves the peak, which
    # is what OOM'd when switching between images on a nearly-full GPU.
    release_pipeline_result(superseded)

    filament_svc = get_filament_service()
    active = filament_svc.get_active()
    if not active:
        with _lock:
            if _in_flight and _in_flight.get("generation") == my_gen:
                _in_flight = None
            if _generation == my_gen:
                _state = {"status": "idle", "preview_image": None, "error": None}
        raise HTTPException(400, "Add at least one active filament first.")

    input_image_path = settings.input_image or ""
    if not input_image_path:
        with _lock:
            if _in_flight and _in_flight.get("generation") == my_gen:
                _in_flight = None
            if _generation == my_gen:
                _state = {"status": "idle", "preview_image": None, "error": None}
        raise HTTPException(400, "Upload an input image first.")

    # Only files under uploads/, resolved the same guarded way as
    # /api/optimize/start. This used to try the name as a path first: any
    # existing server path (absolute, or "../..") was accepted, and a file
    # of the same name in the server's working directory (the repo ships an
    # input.png) silently replaced the uploaded image in the preview.
    resolved_path = get_image_service().get_path(input_image_path)
    if resolved_path is None:
        with _lock:
            if _in_flight and _in_flight.get("generation") == my_gen:
                _in_flight = None
            if _generation == my_gen:
                _state = {"status": "idle", "preview_image": None, "error": None}
        raise HTTPException(404, f"Input image not found: {settings.input_image}")
    input_image_path = resolved_path

    filament_dicts = [
        {"color": f.color, "td": f.td, "name": f"{f.brand} - {f.name}",
         "brand": f.brand, "short_name": f.name, "owned": f.owned,
         "uuid": f.uuid, "filament_type": f.filament_type}
        for f in active
    ]
    settings_dict = settings.model_dump()
    # The preview only estimates heights; the focus-area mask weights the
    # optimizer's loss and plays no part in it. It is also an uploads/ file
    # name here, not a path the pipeline could open.
    settings_dict["priority_mask"] = ""

    box: dict[str, Any] = {}

    def _worker() -> None:
        try:
            box["built"] = _run_init_sync(input_image_path, filament_dicts, settings_dict)
        finally:
            _in_flight["event"].set()

    try:
        await asyncio.to_thread(_worker)
    except Exception as e:
        logger.exception("Auto-preview (init) failed")
        with _lock:
            if _in_flight and _in_flight.get("generation") == my_gen:
                _in_flight = None
            if _generation == my_gen:
                _state = {"status": "idle", "preview_image": None, "error": friendly_error_message(e)}
        release_pipeline_result(box.get("built", {}).get("result"))
        raise HTTPException(500, friendly_error_message(e))

    with _lock:
        if _in_flight and _in_flight.get("generation") == my_gen:
            _in_flight = None
        superseded_by_reset = _generation != my_gen

    if superseded_by_reset:
        # A reset landed mid-build: this result belongs to the image that was
        # just replaced. Hand its GPU memory back and tell the (now-stale)
        # caller so it doesn't publish state for a picture nobody is looking
        # at anymore.
        release_pipeline_result(box.get("built", {}).get("result"))
        raise HTTPException(409, "Auto-preview was reset while it was being prepared.")

    built = box["built"]
    with _lock:
        _pipeline_result = built["result"]
        _state = {
            "status": "ready",
            "preview_image": built["preview_b64"],
            "error": None,
            "range": built["range"],
            "base": built["base"],
        }

    return {
        "status": "ready",
        "preview_image": built["preview_b64"],
        "min_layer": built["range"]["min_layer"],
        "max_layer": built["range"]["max_layer"],
        "base": built["base"],
    }


@router.get("/status")
async def init_status():
    with _lock:
        # The layer range comes with it so a client that finds the preview
        # already "ready" (a reload, another tab) doesn't have to ask
        # /api/sliders/from-optimizer — which answers for the latest
        # *completed job*, not for this preview.
        return {
            "status": _state["status"],
            "error": _state.get("error"),
            # The resolved base color travels with it for the same reason:
            # with auto_background_color on, the color actually used is
            # picked by the pipeline, not by anything the client sent.
            "base": _state.get("base"),
            **(_state.get("range") or {}),
        }


@router.get("/preview")
async def init_preview():
    with _lock:
        if _state.get("preview_image"):
            return {"image": _state["preview_image"]}
    return {"image": None}


@router.get("/mesh")
async def init_mesh():
    # A live slider edit of the preview writes this file in the background.
    await asyncio.to_thread(wait_for_pending_mesh, "__init__")
    with _lock:
        ready = _state["status"] == "ready" and _pipeline_result is not None
    path = _mesh_path()
    if not ready or not os.path.exists(path):
        raise HTTPException(404, "No auto-preview mesh available yet")
    return FileResponse(path, media_type="application/octet-stream")


@router.post("/reset")
async def reset_init():
    """Forget the current auto-preview.

    Uploading a different image used to leave this state at "ready" with the
    *previous* image's mesh still on disk, so the 3D view kept serving it
    (``GET /api/init/mesh``) until the new init finished — the old picture
    shown as the new image's heightmap preview. Clearing it up front means
    the panel shows "building the preview" instead of something wrong, and
    the old optimizer's device memory is handed back at the same time.

    If a build is still in flight, this waits for it to finish: the
    replacement build must not run on the GPU next to it (that's what OOM'd
    when switching images), and the replaced build's result must be released
    before the reset is complete.
    """
    global _state
    in_flight_event = reset_init_state()
    # A slider-edit mesh of the replaced image still being written must land
    # before the new image's preview is built, never after it.
    await asyncio.to_thread(wait_for_pending_mesh, "__init__")
    if in_flight_event is not None:
        await asyncio.to_thread(in_flight_event.wait)
        # The replaced build's completion handler deliberately leaves the
        # status "initializing" while it is still running (that's what kept
        # a second build from starting); now that it is done, hand the state
        # back to idle.
        with _lock:
            if _state["status"] == "initializing":
                _state = {"status": "idle", "preview_image": None, "error": None}
    return {"status": "idle"}


def reset_init_state():
    """Drop the auto-preview: its state, its mesh file and its GPU memory.

    Returns the in-flight build's completion event (or ``None`` when no build
    is running) so the caller can wait it out before starting a new one."""
    global _state, _pipeline_result, _generation, _in_flight
    with _lock:
        _generation += 1
        in_flight_event = _in_flight["event"] if _in_flight is not None else None
        superseded, _pipeline_result = _pipeline_result, None
        if in_flight_event is None:
            _state = {"status": "idle", "preview_image": None, "error": None}
        # Build in flight: the status stays "initializing" until its worker
        # finishes — that is what makes /api/init/run refuse to start a
        # second, concurrent build.
    release_pipeline_result(superseded)
    cancel_pending_mesh("__init__")
    try:
        os.remove(_mesh_path())
    except OSError:
        # Never written, already gone, or not ours to delete — all fine.
        pass
    return in_flight_event
