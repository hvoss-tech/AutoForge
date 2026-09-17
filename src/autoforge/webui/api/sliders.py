from fastapi import APIRouter
from ..services.optimization_service import get_optimization_service
from ..helpers.sliders import derive_sliders_from_result, derive_layer_range_from_result
from .init import get_init_pipeline_result

router = APIRouter()

_DEFAULTS = {"sliders": [], "min_layer": 0, "max_layer": 75}


@router.get("/from-optimizer")
async def get_sliders_from_optimizer():
    """Return the slider stack derived from the most recent completed job.

    The response shape is ``{"sliders": [...], "min_layer": int, "max_layer":
    int}`` where ``sliders`` use snake_case keys matching the frontend
    ``ColorSliderConfig`` type, and ``min_layer``/``max_layer`` bound the
    per-pixel height layers of the solution.

    Falls back to the post-upload auto-preview state (api/init.py) when
    there's no completed job yet — but only for ``min_layer``/``max_layer``
    (the real heightmap-derived range): that state's ``disc_global`` (layer
    -> material assignment) is from an untrained optimizer and therefore
    meaningless, so deriving *segments* from it would show bogus slider
    positions the user never chose. Coloring during that phase is
    explicitly the user's job, not an auto-derived guess.
    """
    svc = get_optimization_service()
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
