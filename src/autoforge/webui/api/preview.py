import os

from fastapi import APIRouter, HTTPException

from ..config import config
from ..services.optimization_service import get_optimization_service
from ..services.filament_service import get_filament_service
from ..helpers.slider_render import render_with_sliders
from .ws import broadcast_preview
from .init import get_init_pipeline_result, _init_dir

router = APIRouter()

# Sent as `job_id` by the frontend while there's no completed optimization
# job yet — just the post-upload auto-preview (api/init.py) — so a slider
# edit made during that phase still gets a live re-render instead of doing
# nothing until the user runs a real optimization.
INIT_JOB_SENTINEL = "__init__"


def _latest_completed_job():
    svc = get_optimization_service()
    for job in svc.get_history():
        if job.status == "completed":
            return job
    return None


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
    else:
        job = None
        if requested_job_id:
            candidate = svc.get_job(requested_job_id)
            if candidate and candidate.status == "completed":
                job = candidate
        if job is None:
            job = _latest_completed_job()
        if job is None:
            raise HTTPException(400, "No completed optimization result to render")

        pipeline_result = svc.get_pipeline_result(job.job_id)
        if not pipeline_result:
            raise HTTPException(400, "No optimization pipeline result found")
        effective_job_id = job.job_id
        output_dir = os.path.join(config.checkpoints_path, job.job_id)

    filament_lookup = {f.uuid: f.model_dump() for f in filament_svc.list()}
    # The frontend also sends the currently active filament list, which may
    # include filaments not (yet) present in the saved library.
    for f in data.get("active_filaments", []) or []:
        uuid_ = str(f.get("uuid", ""))
        if uuid_ and uuid_ not in filament_lookup:
            filament_lookup[uuid_] = f

    result = render_with_sliders(pipeline_result, sliders, filament_lookup, output_dir)
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
        broadcast_preview(result["image_b64"], job_id=effective_job_id)

    return {"status": "ok", "job_id": effective_job_id, "slider_count": len(sliders)}
