import asyncio
import base64
import os
import threading
from collections import defaultdict

from fastapi import APIRouter, HTTPException

from ..config import config
from ..services.optimization_service import get_optimization_service
from ..services.filament_service import get_filament_service
from autoforge.Helper.FilamentHelper import hex_to_rgb
from ..helpers.mesh_persist import schedule_persist
from ..helpers.slider_render import (
    compute_slider_render,
    render_with_sliders,
    top_vertex_colors,
    write_slider_ply,
    write_slider_png,
)
from .ws import broadcast_preview
from .init import get_init_pipeline_result, _init_dir
from .outputs import EDITED_PLY, EDITED_PNG

router = APIRouter()

# Sent as `job_id` by the frontend while there's no completed optimization
# job yet — just the post-upload auto-preview (api/init.py) — so a slider
# edit made during that phase still gets a live re-render instead of doing
# nothing until the user runs a real optimization.
INIT_JOB_SENTINEL = "__init__"

_seq_lock = threading.Lock()
_latest_seq: dict[str, int] = {}
_render_locks: defaultdict[str, threading.Lock] = defaultdict(threading.Lock)

@router.post("/render-with-sliders")
async def render_preview(data: dict):
    """Recompute the composite preview + colored PLY for a completed
    optimization job, using an edited color-slider stack.

    The per-pixel height solution from that job is unchanged — only the
    layer→material assignment (and therefore the compositing) is redone,
    so this is cheap enough to run on every slider drag/edit.
    """
    sliders = data.get("sliders", [])
    svc = get_optimization_service()
    filament_svc = get_filament_service()

    # The frontend tells us which job it's actually looking at (its
    # `currentJob`). Without this we'd guess "most recently started
    # completed job", which silently diverges from the frontend's job once
    # the user has more than one completed run (e.g. after browsing History
    # or undoing to an older result) — the broadcast below is filtered by
    # job_id client-side, so a wrong guess here means the edit is computed
    # correctly but the client drops the update on the floor and the 3D
    # preview never refreshes.
    requested_job_id = data.get("job_id")

    if requested_job_id == INIT_JOB_SENTINEL:
        pipeline_result = get_init_pipeline_result()
        if not pipeline_result:
            raise HTTPException(400, "No auto-preview result to render yet")
        effective_job_id = INIT_JOB_SENTINEL
        output_dir = _init_dir()
        file_names = {}
    else:
        # Never fall back to "some other completed job" when the requested one
        # isn't renderable: that rendered the edit into an unrelated job's
        # output folder, and the broadcast (tagged with the other job's id)
        # was dropped by the client anyway.
        if not requested_job_id:
            raise HTTPException(400, "No job_id given")
        job = svc.get_job(requested_job_id)
        if job is None or job.status != "completed":
            raise HTTPException(400, "No completed optimization result to render")

        pipeline_result = svc.get_pipeline_result(job.job_id)
        if not pipeline_result:
            raise HTTPException(
                409,
                "This result can't be re-colored anymore: it was pruned since, or the "
                "server was restarted after it was computed. Go to the newest result, "
                "or run the optimization again, to edit its colors.",
            )
        effective_job_id = job.job_id
        output_dir = os.path.join(config.checkpoints_path, job.job_id)
        file_names = {"png_name": EDITED_PNG, "ply_name": EDITED_PLY}

    filament_lookup = {f.uuid: f.model_dump() for f in filament_svc.list()}
    # The frontend also sends the currently active filament list, which may
    # include filaments not (yet) present in the saved library.
    for f in data.get("active_filaments", []) or []:
        uuid_ = str(f.get("uuid", ""))
        if uuid_ and uuid_ not in filament_lookup:
            filament_lookup[uuid_] = f

    with _seq_lock:
        _latest_seq[effective_job_id] = seq = _latest_seq.get(effective_job_id, 0) + 1
        render_lock = _render_locks[effective_job_id]

    want_vertex_colors = bool(data.get("vertex_colors"))
    background_color = data.get("background_color")
    background_rgb = tuple(hex_to_rgb(background_color)) if background_color else None

    def _render():
        # One render per target at a time (they write the same files), and a
        # request that was overtaken by a newer edit while it waited is
        # skipped — otherwise a slow older render could finish last and
        # leave the preview showing a stale slider stack.
        with render_lock:
            if _latest_seq.get(effective_job_id) != seq:
                return "superseded"
            if not want_vertex_colors:
                return render_with_sliders(
                    pipeline_result, sliders, filament_lookup, output_dir,
                    background_rgb=background_rgb, **file_names,
                )
            render = compute_slider_render(pipeline_result, sliders, filament_lookup, background_rgb=background_rgb)
            if render is None:
                return None
            png_name = file_names.get("png_name", "final_model.png")
            ply_name = file_names.get("ply_name", "final_model_colored.ply")
            write_slider_png(render, output_dir, png_name)
            schedule_persist(effective_job_id, lambda: write_slider_ply(render, output_dir, ply_name))
            return {
                "image_b64": render["image_b64"],
                "vertex_colors": base64.b64encode(top_vertex_colors(render)).decode("ascii"),
                "top_vertex_count": int(len(render["top_vertex_pixels"])),
            }

    # Off the event loop: this is real GPU/CPU work, and running it inline
    # stalled every other request and websocket while a slider was dragged.
    result = await asyncio.to_thread(_render)
    if result == "superseded":
        return {"status": "superseded", "job_id": effective_job_id}
    if result is None:
        return {"status": "no_solution", "job_id": effective_job_id}

    if result["image_b64"]:
        # Deliberately NOT echoing `sliders` back here. The frontend is the
        # source of truth for its own edit — it already has this exact data
        # before the request goes out. Echoing it back over the shared /ws/
        # preview channel raced with any edit the user made *while this
        # request was in flight*: an older, now-stale echo could land after
        # a newer local change and silently overwrite it (sliders "snapping
        # back" to a previous position, or edits that intermittently didn't
        # stick). Optimization/pruning broadcasts still send `sliders`
        # because those genuinely carry new information the frontend
        # doesn't have (the server-derived stack from a fresh solution).
        #
        # `render_id` lets the tab that asked for this edit recognize it:
        # that tab already recolored its mesh from the response, so it must
        # not refetch the whole PLY because of this broadcast.
        broadcast_preview(result["image_b64"], job_id=effective_job_id, render_id=data.get("render_id"))

    response = {"status": "ok", "job_id": effective_job_id, "slider_count": len(sliders)}
    if "vertex_colors" in result:
        # RGB bytes (base64) for the mesh's top vertices, in vertex order —
        # the client writes them straight into the mesh it already shows.
        response["vertex_colors"] = result["vertex_colors"]
        response["top_vertex_count"] = result["top_vertex_count"]
    return response
