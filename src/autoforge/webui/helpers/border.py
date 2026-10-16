"""Borders for the webui: the 3D view's preview mesh and the bordered STLs
of an export (geometry in Helper/BorderMesh).

A border is taken from the model a job actually exported, never from the
optimizer, so it can be changed at any time without a run, works for a
result restored from disk, and the preview shows exactly the plate the
export will merge in. The footprint of each model file is cached: reading a
full-size STL is the slow part (~1 s), and dragging the width slider asks
for a new border many times over the same model.
"""

from __future__ import annotations

import base64
import glob
import os
import threading
from collections import OrderedDict
from typing import Any, Optional

from autoforge.Helper.BorderMesh import (
    BorderError,
    BorderSpec,
    add_border_to_stl,
    border_ring_mesh,
    load_mesh,
    mesh_footprint,
    outline_size_mm,
    write_plate_stl,
)

from ..config import config
from .mesh_persist import wait_for_pending_mesh

INIT_JOB = "__init__"

_CACHE_SIZE = 4
_cache: "OrderedDict[tuple, Any]" = OrderedDict()
_cache_lock = threading.Lock()


def _job_dir(job_id: str) -> Optional[str]:
    checkpoints = os.path.realpath(config.checkpoints_path)
    path = os.path.realpath(os.path.join(checkpoints, job_id))
    if not path.startswith(checkpoints + os.sep):
        return None
    return path


def background_stl(job_dir: str) -> Optional[str]:
    """A FlatForge run's base-layer STL (OutputHelper.generate_flatforge_stls)."""
    found = sorted(glob.glob(os.path.join(glob.escape(job_dir), "Background_*.stl")))
    return found[0] if found else None


def model_file(job_id: str) -> Optional[str]:
    """The mesh a job's border is built around: its STL, else the 3D view's
    mesh (same footprint). The auto-preview has only the latter; a FlatForge
    run's STLs are separate pieces, none of them standing for the whole."""
    if job_id == INIT_JOB:
        from ..api.init import init_mesh_path

        wait_for_pending_mesh(INIT_JOB)
        path = init_mesh_path()
        return path if os.path.exists(path) else None
    job_dir = _job_dir(job_id)
    if job_dir is None or not os.path.isdir(job_dir):
        return None
    path = os.path.join(job_dir, "final_model.stl")
    if os.path.exists(path):
        return path
    wait_for_pending_mesh(job_id)
    path = os.path.join(job_dir, "final_model_colored.ply")
    return path if os.path.exists(path) else None


def _args_height(args: Any) -> Optional[float]:
    if args is None:
        return None
    value = args.get("background_height") if isinstance(args, dict) else getattr(args, "background_height", None)
    try:
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def background_height(job_id: str, fallback: Optional[float] = None) -> Optional[float]:
    """The base height the job's model was built with. A prune job's own
    settings are placeholders (api/pruning.py), so its saved solution or the
    live result is asked first. ``fallback`` (the client's setting) only
    for an auto-preview whose result was already released."""
    from ..services.optimization_service import get_optimization_service

    svc = get_optimization_service()
    if job_id == INIT_JOB:
        from ..api.init import get_init_pipeline_result

        result = get_init_pipeline_result()
        height = _args_height(result.get("args")) if result else None
        return fallback if height is None else height
    live = svc.get_pipeline_result(job_id)
    height = _args_height(live.get("args")) if live else None
    if height is None:
        from .result_store import load_solution

        job_dir = _job_dir(job_id)
        solution = load_solution(job_dir) if job_dir else None
        height = _args_height((solution or {}).get("meta", {}).get("args"))
    if height is None:
        settings = svc.get_settings(job_id)
        height = float(settings.background_height) if settings else None
    return height


def _footprint(path: str):
    stat = os.stat(path)
    key = (path, stat.st_mtime_ns, stat.st_size)
    with _cache_lock:
        if key in _cache:
            _cache.move_to_end(key)
            return _cache[key]
    footprint = mesh_footprint(*load_mesh(path))
    with _cache_lock:
        _cache[key] = footprint
        while len(_cache) > _CACHE_SIZE:
            _cache.popitem(last=False)
    return footprint


def _model_and_height(job_id: str, fallback_height: Optional[float]) -> tuple[str, float]:
    path = model_file(job_id)
    if path is None:
        raise FileNotFoundError("This result has no 3D model to add a border to.")
    height = background_height(job_id, fallback_height)
    if height is None:
        raise BorderError("The base height of this result is unknown.")
    return path, height


def border_preview(job_id: str, spec: BorderSpec, fallback_height: Optional[float] = None) -> dict[str, Any]:
    """The border for the 3D view, in the coordinates of the job's mesh:
    base64 float32 positions and uint32 triangle indices (empty without a
    border), plus the print's size with and without it."""
    path, height = _model_and_height(job_id, fallback_height)
    footprint = _footprint(path)
    x0, y0, x1, y1 = footprint.bounds()
    mesh = border_ring_mesh(footprint, spec, height)
    positions, indices = (b"", b"") if mesh is None else (mesh[0].tobytes(), mesh[1].tobytes())
    width, depth = outline_size_mm(footprint, spec)
    return {
        "positions": base64.b64encode(positions).decode("ascii"),
        "indices": base64.b64encode(indices).decode("ascii"),
        "background_height": height,
        "size_mm": [round(width, 2), round(depth, 2)],
        "model_size_mm": [round(x1 - x0, 2), round(y1 - y0, 2)],
    }


def write_bordered_stl(job_id: str, src: str, dst: str, spec: BorderSpec) -> None:
    """``src`` - the job's STL, or a FlatForge run's base-layer STL - with
    the border merged in, at ``dst``."""
    height = background_height(job_id)
    if height is None:
        raise BorderError("The base height of this result is unknown.")
    if src == background_stl(os.path.dirname(src)):
        # The base layer is the footprint, base-height thick: with the
        # border it is the whole plate.
        path = model_file(job_id)
        if path is None:
            raise FileNotFoundError("This result has no 3D model to add a border to.")
        write_plate_stl(_footprint(path), dst, spec, height)
        return
    add_border_to_stl(src, dst, spec, height)
