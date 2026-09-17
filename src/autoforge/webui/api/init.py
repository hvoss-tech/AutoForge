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
from ..helpers.pipeline_runner import build_init_preview, friendly_error_message
from ..helpers.colored_mesh import generate_colored_preview_mesh
from ..helpers.sliders import derive_layer_range_from_result

logger = logging.getLogger(__name__)
router = APIRouter()

_state: dict[str, Any] = {"status": "idle", "preview_image": None, "error": None}
_pipeline_result: Optional[dict[str, Any]] = None
_lock = threading.RLock()


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
    mesh.export(_mesh_path(), encoding="binary")

    # Flat PNG kept too — /api/init/preview predates the 3D mesh and other
    # code (e.g. a text-only client) may still just want an image.
    preview_bgr = cv2.cvtColor(color_image, cv2.COLOR_RGB2BGR)
    _ok, buf = cv2.imencode(".png", preview_bgr)
    preview_b64 = base64.b64encode(buf.tobytes()).decode("utf-8") if _ok else None

    range_info = derive_layer_range_from_result(result) or {"min_layer": 0, "max_layer": int(args.max_layers)}

    return {"result": result, "preview_b64": preview_b64, "range": range_info}


@router.post("/run")
async def run_init(settings: OptimizationSettings):
    global _state, _pipeline_result

    with _lock:
        if _state["status"] == "initializing":
            raise HTTPException(400, "Already initializing")
        _state = {"status": "initializing", "preview_image": None, "error": None}

    filament_svc = get_filament_service()
    active = filament_svc.get_active()
    if not active:
        with _lock:
            _state = {"status": "idle", "preview_image": None, "error": None}
        raise HTTPException(400, "Add at least one active filament first.")

    input_image_path = settings.input_image or ""
    if not input_image_path:
        with _lock:
            _state = {"status": "idle", "preview_image": None, "error": None}
        raise HTTPException(400, "Upload an input image first.")

    if not os.path.exists(input_image_path):
        abs_path = os.path.join(config.uploads_path, input_image_path)
        if os.path.exists(abs_path):
            input_image_path = abs_path
        else:
            with _lock:
                _state = {"status": "idle", "preview_image": None, "error": None}
            raise HTTPException(404, f"Input image not found: {settings.input_image}")

    filament_dicts = [
        {"color": f.color, "td": f.td, "name": f"{f.brand} - {f.name}",
         "brand": f.brand, "short_name": f.name, "owned": f.owned,
         "uuid": f.uuid, "filament_type": f.filament_type}
        for f in active
    ]
    settings_dict = settings.model_dump()

    try:
        built = await asyncio.to_thread(_run_init_sync, input_image_path, filament_dicts, settings_dict)
    except Exception as e:
        logger.exception("Auto-preview (init) failed")
        with _lock:
            _state = {"status": "idle", "preview_image": None, "error": friendly_error_message(e)}
        raise HTTPException(500, friendly_error_message(e))

    with _lock:
        _pipeline_result = built["result"]
        _state = {"status": "ready", "preview_image": built["preview_b64"], "error": None}

    return {
        "status": "ready",
        "preview_image": built["preview_b64"],
        "min_layer": built["range"]["min_layer"],
        "max_layer": built["range"]["max_layer"],
    }


@router.get("/status")
async def init_status():
    with _lock:
        return {"status": _state["status"], "error": _state.get("error")}


@router.get("/preview")
async def init_preview():
    with _lock:
        if _state.get("preview_image"):
            return {"image": _state["preview_image"]}
    return {"image": None}


@router.get("/mesh")
async def init_mesh():
    with _lock:
        ready = _state["status"] == "ready" and _pipeline_result is not None
    path = _mesh_path()
    if not ready or not os.path.exists(path):
        raise HTTPException(404, "No auto-preview mesh available yet")
    return FileResponse(path, media_type="application/octet-stream")


def reset_init_state():
    """Test-only: clear module-level state between isolated test runs."""
    global _state, _pipeline_result
    with _lock:
        _state = {"status": "idle", "preview_image": None, "error": None}
        _pipeline_result = None
