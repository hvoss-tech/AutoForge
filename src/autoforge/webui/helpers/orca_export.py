"""OrcaSlicer / Bambu Studio project (.3mf) for a webui result.

The project follows what the user sees: the colour layers of the posted
project (slider edits included) over the result's saved height map, and the
base filament the pipeline resolved. Without a posted project (an export
bundle requested by URL) it uses the result's own solution.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from autoforge.Helper.Export3MF import SwapPlan, export_3mf, swap_plan
from autoforge.Helper.FilamentHelper import normalize_hex

from .edited_export import edited_stack
from .result_store import load_solution


class ExportUnavailable(Exception):
    """The result can't be exported as a slicer project."""


def _hex(color: Any) -> str | None:
    try:
        return normalize_hex(color).lower() if color else None
    except ValueError:
        return None


def _library_filaments() -> dict[str, dict[str, Any]]:
    from ..services.filament_service import get_filament_service

    return {f.uuid: f.model_dump() for f in get_filament_service().list() if f.uuid}


def run_nozzle(job_dir: str, project: dict[str, Any] | None) -> float | None:
    """The nozzle (mm) the result was made for: the run's own, else the
    project settings'. None when neither says."""
    solution = load_solution(job_dir)
    args = ((solution or {}).get("meta") or {}).get("args") or {}
    settings = (project or {}).get("settings") or {}
    for value in (args.get("nozzle_diameter"), settings.get("nozzle_diameter")):
        try:
            if value is not None and float(value) > 0:
                return float(value)
        except (TypeError, ValueError):
            continue
    return None


def export_plan(job_dir: str, project: dict[str, Any] | None, base: dict[str, Any] | None = None) -> tuple[SwapPlan, float, float]:
    """``(plan, layer_height, background_height)`` for the result in ``job_dir``.

    ``project`` is the webui project (settings, colorSliders, activeFilaments);
    ``base`` the resolved base (``filament_uuid``, ``color``) the UI shows."""
    solution = load_solution(job_dir)
    if solution is None:
        raise ExportUnavailable("This result has no saved solution to build a slicer project from; run it again.")
    heights = solution["heights"]
    meta = solution["meta"] or {}
    args = meta.get("args") or {}
    settings = (project or {}).get("settings") or {}
    # The run's values first: the STL and the height map were built with
    # them, whatever the settings say now (changed, no new run yet).
    layer_height = float(args.get("layer_height") or settings.get("layer_height") or 0.04)
    background_height = float(args.get("background_height") or settings.get("background_height") or 0.24)
    max_layer = int(np.max(heights)) if heights.size else 0
    sliders = (project or {}).get("colorSliders") or []

    if not sliders:
        return (
            swap_plan(
                solution["stack"], max_layer, layer_height, background_height,
                meta.get("materials") or [], meta.get("background_material_index"),
                str(meta.get("background_color") or "#000000"),
            ),
            layer_height,
            background_height,
        )

    disc_global, uuids, material_sliders = edited_stack(sliders, max(max_layer, 1))
    if not uuids:
        raise ExportUnavailable("Assign filaments to the color layers first.")
    known = _library_filaments()
    for f in (project or {}).get("activeFilaments") or []:
        if isinstance(f, dict) and f.get("uuid"):
            known[str(f["uuid"])] = f
    # One material per filament: edited_stack splits a filament used with two
    # TDs into two materials, which would be a pointless swap here.
    materials: list[dict[str, Any]] = []
    index_of: dict[str, int] = {}
    remap = []
    for uuid, slider in zip(uuids, material_sliders):
        if uuid not in index_of:
            f = known.get(uuid) or {}
            index_of[uuid] = len(materials)
            materials.append({
                "color": f.get("color") or slider.get("color") or "#808080",
                "name": f.get("name") or "",
                "brand": f.get("brand") or "",
                "filament_type": f.get("filament_type") or f.get("filamentType") or "PLA",
            })
        remap.append(index_of[uuid])
    stack = np.asarray(remap, dtype=np.int64)[disc_global]

    base = base or {}
    base_color = _hex(base.get("color")) or _hex(settings.get("background_color")) or "#000000"
    base_uuid = str(base.get("filament_uuid") or settings.get("base_filament_uuid") or "")
    base_index = index_of.get(base_uuid) if base_uuid else None
    if base_index is None and base_uuid and base_uuid in known:
        f = known[base_uuid]
        base_index = len(materials)
        materials.append({
            "color": f.get("color") or base_color,
            "name": f.get("name") or "",
            "brand": f.get("brand") or "",
            "filament_type": f.get("filament_type") or "PLA",
        })
    if base_index is None:
        base_index = next((i for i, m in enumerate(materials) if _hex(m["color"]) == base_color), None)
    plan = swap_plan(stack, max_layer, layer_height, background_height, materials, base_index, base_color)
    return plan, layer_height, background_height


def write_project_3mf(
    out_path: str,
    stl_path: str,
    job_dir: str,
    project: dict[str, Any] | None,
    printer: str,
    manual_swaps: bool | None,
    title: str,
    base: dict[str, Any] | None = None,
    progress=None,
    autoforge_settings: bool = True,
) -> dict[str, Any]:
    """Write the project; returns export_3mf's info (preset used, notes, warnings)."""
    plan, layer_height, background_height = export_plan(job_dir, project, base)
    return export_3mf(
        out_path, stl_path, plan, printer, layer_height, background_height, manual_swaps,
        title=title, progress=progress, nozzle=run_nozzle(job_dir, project),
        autoforge_settings=autoforge_settings,
    )
