"""Swap instructions and HueForge project for a slider-edited result.

A slider edit only changes which filament prints which layer band (see
slider_render), so everything the export bundle needs to describe the
edited print - the swap instructions and the .hfp - follows from the edited
stack plus the result's unchanged height map. Without this, the zip shipped
the optimizer's own instructions next to an edited preview: printing from
them gave the colours the user had just replaced.
"""

from __future__ import annotations

import copy
import csv
import os
from typing import Any

import numpy as np

from autoforge.Helper.DeviceUtils import activate_device
from autoforge.Helper.FilamentHelper import normalize_hex
from autoforge.Helper.OutputHelper import generate_project_file, generate_swap_instructions, model_size_mm
from autoforge.webui.helpers.slider_render import _dedupe_overlapping_layers, _layer_material_indices


def edited_stack(sliders: list[dict], max_layers: int) -> tuple[np.ndarray, list[str], list[dict]]:
    """``(disc_global, material_uuids, material_sliders)`` for an edited
    slider stack: one material per distinct filament and TD (first slider
    that uses it), and each layer's index into that list. Two bands of one
    filament with different TDs render differently (slider_render uses each
    slider's own TD), so they stay two materials here too."""
    enabled = sorted(_dedupe_overlapping_layers(sliders), key=lambda s: int(s["layer"]))
    if not enabled:
        return np.zeros(0, dtype=np.int64), [], []
    slot_of_layer = _layer_material_indices(enabled, max_layers)
    keys: list[tuple[str, float]] = []
    uuids: list[str] = []
    material_sliders: list[dict] = []
    slot_to_material = []
    for s in enabled:
        key = (str(s.get("filament_uuid", "")), float(s.get("td") or 0.0))
        if key not in keys:
            keys.append(key)
            uuids.append(key[0])
            material_sliders.append(s)
        slot_to_material.append(keys.index(key))
    disc_global = np.array([slot_to_material[i] for i in slot_of_layer], dtype=np.int64)
    return disc_global, uuids, material_sliders


def _normalized_hex(color) -> str | None:
    if not color:
        return None
    try:
        return normalize_hex(color)
    except ValueError:
        return None


def write_edited_outputs(
    result: dict[str, Any],
    sliders: list[dict],
    filaments_by_uuid: dict[str, dict],
    out_dir: str,
    background_color: str | None = None,
) -> dict[str, str]:
    """Write ``swap_instructions.txt`` and ``project_file.hfp`` for the
    edited stack into ``out_dir``. Returns {file name: path}; empty when
    there is no solution or no enabled slider.

    ``background_color``: the base color as edited in the UI (the project's
    settings), when it differs from the run's - the export used to describe
    the run's base under an edited one."""
    optimizer = result["optimizer"]
    activate_device(result.get("device"))
    _, disc_height = optimizer.get_discretized_solution(best=True)
    if disc_height is None:
        return {}
    max_layers = int(optimizer.max_layers)
    disc_global, uuids, material_sliders = edited_stack(sliders, max_layers)
    if not uuids:
        return {}
    heights = disc_height.detach().cpu().numpy()

    records = []
    for uuid, s in zip(uuids, material_sliders):
        f = filaments_by_uuid.get(uuid) or {}
        td = float(s.get("td") or f.get("td") or 5.0)
        records.append({
            "Brand": f.get("brand", ""),
            "Name": f.get("name", "") or "Unassigned",
            "Color": f.get("color", "#808080"),
            "Transmissivity": td,
            "Type": f.get("filament_type") or "PLA",
            "Owned": bool(f.get("owned", False)),
            "Uuid": uuid,
        })
    names = [" - ".join(p for p in (r["Brand"], r["Name"]) if p) for r in records]

    args = copy.copy(result["args"])
    # The base filament, re-indexed into this stack's material list (the
    # result's index points into the run's own filament list).
    base_index = None
    edited_base = _normalized_hex(background_color)
    if edited_base and edited_base != _normalized_hex(getattr(args, "background_color", None)):
        # The base was changed in the UI: describe that one. It is a
        # filament of this stack when one has its color.
        args.background_color = edited_base
        args.background_material_name = None
        for i, r in enumerate(records):
            if _normalized_hex(r["Color"]) == edited_base:
                base_index = i
                break
    else:
        run_index = getattr(args, "background_material_index", None)
        run_uuids = result.get("material_uuids") or []
        if run_index is not None and 0 <= int(run_index) < len(run_uuids) and run_uuids[int(run_index)] in uuids:
            base_index = uuids.index(run_uuids[int(run_index)])
    args.background_material_index = base_index
    base_name = names[base_index] if base_index is not None else getattr(args, "background_material_name", None)
    args.max_layers = max_layers

    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, "materials.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    args.csv_file, args.json_file = csv_path, ""

    lh = float(args.layer_height)
    swap_path = os.path.join(out_dir, "swap_instructions.txt")
    lines = generate_swap_instructions(
        disc_global, heights, lh, int(round(float(args.background_height) / lh)),
        float(args.background_height), names, base_name, base_index,
    )
    with open(swap_path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")

    hfp_path = os.path.join(out_dir, "project_file.hfp")
    target = result["output_target"]
    generate_project_file(
        hfp_path, args, disc_global, heights,
        *model_size_mm(target.shape[1], target.shape[0], args.stl_output_size),
        "final_model.stl", csv_path,
    )
    return {"swap_instructions.txt": swap_path, "project_file.hfp": hfp_path}
