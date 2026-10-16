import asyncio
import os

from fastapi import APIRouter, HTTPException
from ..config import config
from ..services.optimization_service import get_optimization_service
from ..helpers.sliders import (
    derive_base_from_result,
    derive_sliders_from_result,
    derive_layer_range_from_result,
)
from ..helpers.result_store import ResultUnavailable, load_solution, using_result
from .init import get_init_pipeline_result

router = APIRouter()


def _from_result(job_id: str, derive):
    """``derive(result)`` for the job's result — rebuilt from its saved
    solution if need be — or None when there is none. Blocking (a rebuild
    runs on the GPU)."""
    try:
        with using_result(job_id) as result:
            return derive(result)
    except ResultUnavailable:
        return None

_DEFAULTS = {"sliders": [], "min_layer": 0, "max_layer": 75}


@router.get("/range")
async def get_layer_range(job_id: str):
    """``{"min_layer", "max_layer"}`` of a result's height map: what the
    color core runs up to. Cheap - the live result, else the job's saved
    solution, never a rebuild - because the page asks on every reload: it
    used to keep its placeholder range (75) there, so the color core ran far
    past the result's top layer."""
    return await asyncio.to_thread(_layer_range, job_id)


def _layer_range(job_id: str) -> dict:
    if job_id == "__init__":
        result = get_init_pipeline_result()
        range_info = derive_layer_range_from_result(result) if result else None
        if range_info:
            return range_info
        raise HTTPException(404, "No auto-preview")
    live = get_optimization_service().get_pipeline_result(job_id)
    if live:
        range_info = derive_layer_range_from_result(live)
        if range_info:
            return range_info
    checkpoints = os.path.realpath(config.checkpoints_path)
    job_dir = os.path.realpath(os.path.join(checkpoints, job_id))
    solution = load_solution(job_dir) if job_dir.startswith(checkpoints + os.sep) else None
    if solution is None:
        raise HTTPException(404, "No saved result for this job")
    heights = solution["heights"]
    if heights.size == 0:
        return {"min_layer": 0, "max_layer": 0}
    return {"min_layer": int(heights.min()), "max_layer": int(heights.max())}


@router.get("/from-optimizer")
async def get_sliders_from_optimizer(job_id: str | None = None):
    # Off the event loop: deriving the stack discretizes the solution on the
    # GPU, which stalled every other request and websocket meanwhile.
    return await asyncio.to_thread(_sliders_from_optimizer, job_id)


def _sliders_from_optimizer(job_id: str | None = None):
    """Return the slider stack derived from a completed job.

    The response shape is ``{"sliders": [...], "min_layer": int, "max_layer":
    int}`` where ``sliders`` use snake_case keys matching the frontend
    ``ColorSliderConfig`` type, and ``min_layer``/``max_layer`` bound the
    per-pixel height layers of the solution.

    With ``job_id`` it answers for *that* job's result — mirroring
    ``/api/sliders/base``'s ``job_id`` param: ``set_pipeline_result`` only
    ever retains the newest job's result, so without pinning to the job the
    UI is actually looking at, navigating back to an older (still
    "completed") job via undo/history left this endpoint returning the
    newest job's sliders while ``/base`` correctly answered for the older
    one — base color and slider colors visibly disagreed. Without it: falls
    back to the most recent completed job with a retained result.

    Falls back to the post-upload auto-preview state (api/init.py) when
    there's no completed job yet — but only for ``min_layer``/``max_layer``
    (the real heightmap-derived range): that state's ``disc_global`` (layer
    -> material assignment) is from an untrained optimizer and therefore
    meaningless, so deriving *segments* from it would show bogus slider
    positions the user never chose. Coloring during that phase is
    explicitly the user's job, not an auto-derived guess.
    """
    svc = get_optimization_service()

    if job_id:
        job = svc.get_job(job_id)
        if job and job.status == "completed":
            return _from_result(job.job_id, derive_sliders_from_result) or _DEFAULTS
        return _DEFAULTS

    for job in svc.get_history():
        if job.status != "completed":
            continue
        result = svc.get_pipeline_result(job.job_id)
        if not result:
            continue
        derived = derive_sliders_from_result(result)
        if derived:
            return derived
        return _DEFAULTS

    init_result = get_init_pipeline_result()
    if init_result:
        range_info = derive_layer_range_from_result(init_result)
        if range_info:
            return {"sliders": [], **range_info}

    return _DEFAULTS


@router.get("/base")
async def get_base_color(job_id: str | None = None):
    """The base/background slab as it was actually built: its resolved color,
    its height, and which active filament that color belongs to.

    The frontend shows the base as the bottom row of the color layers, and it
    cannot read that row off its own settings: ``auto_background_color``
    (on by default) makes the pipeline replace ``background_color`` with the
    active filament closest to the image's dominant color, and only the
    pipeline result knows which one that was.

    With ``job_id`` it answers for *that* job's result (the one the UI is
    looking at — undoing back to an older result must not show the newest
    run's base). Without it: the newest completed job, else the auto-preview,
    else ``null`` — the caller then falls back to its own settings.
    """
    svc = get_optimization_service()
    if job_id:
        job = svc.get_job(job_id)
        if job and job.status == "completed":
            base = await asyncio.to_thread(_from_result, job.job_id, derive_base_from_result)
            if base:
                return {"base": base, "source": job.job_id}
        return {"base": None, "source": None}

    for job in svc.get_history():
        if job.status != "completed":
            continue
        result = svc.get_pipeline_result(job.job_id)
        if result:
            return {"base": derive_base_from_result(result), "source": job.job_id}

    init_result = get_init_pipeline_result()
    if init_result:
        return {"base": derive_base_from_result(init_result), "source": "init"}

    return {"base": None, "source": None}
