"""Export a heightmap print as an OrcaSlicer / Bambu Studio project (.3mf).

The project carries the model, one filament per colour, a filament change at
every swap height, and the complete configuration of the printer the user
picked (see OrcaProfiles), so it opens sliced-and-ready for that printer.

Swaps are written as tool changes between filaments. Printers with an AMS,
MMU or tool changer do them on their own. For every other printer the export
turns on Orca's "manual filament change", which replaces each tool change with
the printer's pause G-code: the print stops and the user swaps the spool.
"""

from __future__ import annotations

import io
import json
import math
import os
import re
import struct
import threading
import uuid as _uuid
import zipfile
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Sequence

import numpy as np

from autoforge.Helper.OrcaProfiles import (
    OrcaProfiles, ProfileError, as_list, load_printer_index, nozzle_list, printer_for_nozzle, shared_profiles,
)

# Matches what OrcaSlicer 2.5.0 writes into its own projects.
ORCA_VERSION = "2.5.0"
BBS_VERSION = "02.08.01.55"

# Process settings every heightmap print needs, whatever the printer.
PROCESS_OVERRIDES = {
    # Light has to pass through solid plastic for the colours to mix.
    "sparse_infill_density": "100%",
    "sparse_infill_pattern": "rectilinear",
    # Purging happens in the nozzle / poop chute; a tower would stand next to the picture.
    "enable_prime_tower": "0",
    "enable_support": "0",
    # A brim would have to be cut off the picture; ironing smears the thin colour layers.
    "brim_type": "no_brim",
    "ironing_type": "no ironing",
    # Pointless at 100 % infill, and with it on OrcaSlicer 2.5 failed to
    # slice heightmaps on the Dremel 3D20/3D45 profiles.
    "detect_narrow_internal_solid_infill": "0",
}
# Optional "AutoForge print settings" for more reliable colour-layer prints,
# based on published presets for colour-layer (filament painting) prints and
# the process settings of existing colour-layer 3MF exports. ("set", v) sets
# a value; ("max", v) / ("min", v) only lower / raise a printer's own value,
# so a printer that is already gentler keeps its setting.
AUTOFORGE_PRINT_SETTINGS: dict[str, tuple[str, Any]] = {
    # One wall, everything else solid infill: no sparse/solid switching
    # between layers, so every colour layer has the same structure.
    "wall_loops": ("set", "1"),
    "bottom_shell_layers": ("set", "999"),
    "bottom_shell_thickness": ("set", "0"),
    "top_shell_layers": ("set", "1"),
    "top_shell_thickness": ("set", "0"),
    "top_surface_pattern": ("set", "monotonic"),
    "bottom_surface_pattern": ("set", "rectilinear"),
    "internal_solid_infill_pattern": ("set", "alignedrectilinear"),
    "ensure_vertical_shell_thickness": ("set", "none"),
    # Keep small colour islands instead of dropping them as gaps.
    "filter_out_gap_fill": ("set", "0"),
    # Classic walls: Arachne crashed OrcaSlicer 2.4 on AutoForge heightmaps.
    "wall_generator": ("set", "classic"),
    # No G2/G3 arcs: firmware without arc support prints them wrong.
    "enable_arc_fitting": ("set", "disabled"),
    "enable_overhang_speed": ("set", "0"),
    # Gentler on the thin layers and the large flat base.
    "initial_layer_speed": ("max", 30),
    "initial_layer_infill_speed": ("max", 50),
    "outer_wall_speed": ("max", 100),
    "top_surface_speed": ("max", 100),
    # Prime the nozzle before the base layer.
    "skirt_loops": ("min", 1),
}

# For the UI: what the AutoForge print settings change, in words.
AUTOFORGE_PRINT_SETTINGS_SUMMARY = [
    "1 wall; every layer solid (bottom shells 999, top shell 1 layer)",
    "Monotonic top surface, rectilinear bottom, aligned rectilinear solid infill",
    "Classic wall generator, no vertical shell thickening, small gaps kept",
    "Arc fitting and overhang slowdown off",
    "Slower first layer (≤30 mm/s, infill ≤50), outer walls and top surface ≤100 mm/s",
    "A skirt loop to prime the nozzle",
]


def _apply_setting(current: Any, mode: str, value: Any) -> Any:
    """``value`` in ``current``'s shape (scalar or per-variant vector)."""
    def one(old: Any) -> Any:
        if mode == "set":
            return str(value)
        try:
            number = float(old)
        except (TypeError, ValueError):
            return old  # "nil" or a percentage: leave the printer's value
        limited = min(number, float(value)) if mode == "max" else max(number, float(value))
        return f"{limited:g}"

    if isinstance(current, list):
        return [one(v) for v in current] if current else [one(None) if mode == "set" else str(value)]
    if current is None and mode != "set":
        return str(value)
    return one(current)


# Keys copied from the printer that are not per-filament settings even though
# they name filaments.
_DROP_KEYS = frozenset({
    "compatible_printers", "compatible_printers_condition", "compatible_prints",
    "compatible_prints_condition", "filament_id", "inherits", "print_settings_id",
    "printer_settings_id", "filament_settings_id",
})

# Per-extruder-variant options (OrcaSlicer 2.5 PrintConfig.cpp, *_options_with_variant).
# A preset holds one value per variant of its kind; Orca pads shorter vectors by
# repeating the first value (extend_default_config_length) and so does this export.
_PRINT_VARIANT_KEYS = frozenset("""
initial_layer_speed initial_layer_infill_speed outer_wall_speed inner_wall_speed small_perimeter_speed
small_perimeter_threshold sparse_infill_speed internal_solid_infill_speed top_surface_speed
enable_overhang_speed overhang_1_4_speed overhang_2_4_speed overhang_3_4_speed overhang_4_4_speed
slowdown_for_curled_perimeters bridge_speed internal_bridge_speed gap_infill_speed support_speed
support_interface_speed travel_speed travel_speed_z initial_layer_travel_speed default_acceleration
bridge_acceleration travel_acceleration initial_layer_travel_acceleration initial_layer_acceleration
outer_wall_acceleration inner_wall_acceleration sparse_infill_acceleration
internal_solid_infill_acceleration top_surface_acceleration default_jerk outer_wall_jerk inner_wall_jerk
infill_jerk top_surface_jerk initial_layer_jerk travel_jerk initial_layer_travel_jerk
default_junction_deviation print_extruder_id print_extruder_variant top_solid_infill_flow_ratio
""".split())
_FILAMENT_VARIANT_KEYS = frozenset("""
filament_flow_ratio filament_max_volumetric_speed filament_ramming_volumetric_speed
filament_pre_cooling_temperature filament_ramming_travel_time filament_ramming_volumetric_speed_nc
filament_pre_cooling_temperature_nc filament_ramming_travel_time_nc filament_retract_length_nc
filament_preheat_temperature_delta filament_extruder_id filament_extruder_variant
filament_retraction_length filament_z_hop filament_z_hop_types filament_retract_lift_above
filament_retract_lift_below filament_retract_lift_enforce filament_retract_restart_extra
filament_retract_length_toolchange filament_retract_restart_extra_toolchange filament_retraction_speed
filament_deretraction_speed filament_retraction_minimum_travel filament_retract_when_changing_layer
filament_wipe filament_wipe_distance filament_retract_before_wipe filament_retract_after_wipe
filament_long_retractions_when_cut filament_retraction_distances_when_cut long_retractions_when_ec
retraction_distances_when_ec nozzle_temperature_initial_layer nozzle_temperature
filament_flush_volumetric_speed filament_flush_temp filament_cooling_before_tower
volumetric_speed_coefficients filament_adaptive_volumetric_speed filament_ironing_flow
filament_ironing_spacing filament_ironing_inset filament_ironing_speed enable_pressure_advance
pressure_advance adaptive_pressure_advance adaptive_pressure_advance_model
adaptive_pressure_advance_overhangs adaptive_pressure_advance_bridges fan_min_speed fan_max_speed
additional_cooling_fan_speed filament_minimal_purge_on_wipe_tower filament_multitool_ramming
filament_multitool_ramming_volume filament_multitool_ramming_flow nozzle_temperature_range_low
nozzle_temperature_range_high activate_air_filtration activate_air_filtration_during_print
activate_air_filtration_on_completion during_print_exhaust_fan_speed complete_print_exhaust_fan_speed
""".split())
_PRINTER_VARIANT_KEYS = frozenset("""
nozzle_volume retraction_length z_hop travel_slope retract_lift_above retract_lift_below
retract_lift_enforce z_hop_types retraction_speed deretraction_speed retraction_minimum_travel
retract_when_changing_layer wipe wipe_distance retract_before_wipe retract_after_wipe
retract_length_toolchange retract_restart_extra retract_restart_extra_toolchange
long_retractions_when_cut retraction_distances_when_cut nozzle_type printer_extruder_id
printer_extruder_variant hotend_cooling_rate hotend_heating_rate nozzle_flush_dataset
""".split())
# Machine limits hold two values (normal, silent) per variant.
_PRINTER_VARIANT_KEYS_2 = frozenset("""
machine_max_acceleration_x machine_max_acceleration_y machine_max_acceleration_z
machine_max_acceleration_e machine_max_acceleration_extruding machine_max_acceleration_retracting
machine_max_acceleration_travel machine_max_speed_x machine_max_speed_y machine_max_speed_z
machine_max_speed_e machine_max_jerk_x machine_max_jerk_y machine_max_jerk_z machine_max_jerk_e
machine_max_junction_deviation
""".split())


def _resize(value: Any, length: int) -> Any:
    """``value`` as a vector of ``length``: padded with its first value, or cut."""
    values = value if isinstance(value, list) else [value]
    if not values or length < 1:
        return values
    return (values + [values[0]] * length)[:length]


def pad_variants(config: dict[str, Any], kind: str) -> dict[str, Any]:
    """Size a resolved preset's per-variant vectors the way Orca does on load."""
    out = dict(config)
    if kind == "filament":
        n = len(_vector(out.get("filament_extruder_variant", [None])))
        groups = [(_FILAMENT_VARIANT_KEYS, n)]
    elif kind == "process":
        n = len(_vector(out.get("print_extruder_variant", [None])))
        groups = [(_PRINT_VARIANT_KEYS, n)]
    else:
        n = len(_vector(out["printer_extruder_variant"])) if "printer_extruder_variant" in out else len(
            as_list(out.get("nozzle_diameter")) or [None])
        groups = [(_PRINTER_VARIANT_KEYS, n), (_PRINTER_VARIANT_KEYS_2, 2 * n)]
    for keys, length in groups:
        for key in keys & out.keys():
            out[key] = _resize(out[key], length)
    return out


# Vendors and model words for printers that change filament on their own.
_AUTO_SWAP_VENDORS = {"BBL"}
_AUTO_SWAP_WORDS = ("MMU", "AMS", "CFS", "Toolchanger", "Tool Changer", "IDEX", "Snapmaker U1", "INDX")


def printer_changes_filament(printer: dict[str, Any]) -> bool:
    """Whether a printer swaps filament without help (AMS, MMU, tool changer)."""
    if printer.get("vendor") in _AUTO_SWAP_VENDORS:
        return True
    # Tool changers; two-extruder IDEX/dual printers still swap by hand when a
    # print has more colours than tools.
    if int(printer.get("extruders") or 1) >= 4:
        return True
    name = f"{printer.get('model', '')} {printer.get('name', '')}".lower()
    return any(word.lower() in name for word in _AUTO_SWAP_WORDS)


# ------------------------------------------------------------------ swap plan

@dataclass
class Slot:
    """One filament of the project."""
    color: str
    name: str = ""
    filament_type: str = "PLA"


@dataclass
class SwapPlan:
    """Filament 1 prints the base; each swap switches to a slot at a height.

    ``swaps`` holds ``(top_z, slot index)``: the first layer whose top is at
    ``top_z`` prints with that slot."""
    slots: list[Slot]
    swaps: list[tuple[float, int]] = field(default_factory=list)


def _hex(color: Any, fallback: str = "#808080") -> str:
    text = str(color or "").strip()
    if not text:
        return fallback
    if not text.startswith("#"):
        text = "#" + text
    if len(text) == 4:
        text = "#" + "".join(c * 2 for c in text[1:])
    try:
        int(text[1:7], 16)
    except ValueError:
        return fallback
    return text[:7].upper()


def _slot(material: dict[str, Any]) -> Slot:
    name = str(material.get("name") or "")
    brand = str(material.get("brand") or "")
    if brand and name and not name.lower().startswith(brand.lower()):
        name = f"{brand} - {name}"
    return Slot(
        color=_hex(material.get("color")),
        name=name,
        filament_type=str(material.get("filament_type") or material.get("filamentType") or "PLA"),
    )


def swap_plan(
    stack: Sequence[int],
    max_layer: int,
    layer_height: float,
    background_height: float,
    materials: Sequence[dict[str, Any]],
    background_material_index: int | None,
    background_color: str = "#000000",
) -> SwapPlan:
    """The swaps of ``generate_swap_instructions``, as filament slots.

    Layer #1 is the base (``background_height`` thick); ``stack[i]`` is the
    material of layer #``i + 2``, whose top is at
    ``background_height + (i + 1) * layer_height``."""
    stack = [int(m) for m in np.asarray(stack).reshape(-1)]
    max_layer = min(int(max_layer), len(stack))
    if background_material_index is not None and 0 <= int(background_material_index) < len(materials):
        base_index = int(background_material_index)
        slots = [_slot(materials[base_index])]
        slot_of = {base_index: 0}
    else:
        base_index = None
        slots = [Slot(color=_hex(background_color, "#000000"), name="Background")]
        slot_of = {}
    plan = SwapPlan(slots=slots)
    current = 0
    for i in range(max_layer):
        material = stack[i]
        if i == 0 and base_index is not None and material == base_index:
            continue
        if i > 0 and material == stack[i - 1]:
            continue
        if material not in slot_of:
            if not 0 <= material < len(materials):
                raise ValueError(f"Layer {i + 2} uses material {material}, but there are only {len(materials)}")
            slot_of[material] = len(plan.slots)
            plan.slots.append(_slot(materials[material]))
        slot = slot_of[material]
        if slot != current:
            top_z = round(float(background_height) + (i + 1) * float(layer_height), 6)
            plan.swaps.append((top_z, slot))
            current = slot
    return plan


# ----------------------------------------------------------------------- mesh

_STL_RECORD = np.dtype([("normal", "<f4", 3), ("v", "<f4", (3, 3)), ("attr", "<u2")])


_mesh_cache: "OrderedDict[tuple, tuple[np.ndarray, np.ndarray]]" = OrderedDict()
_mesh_cache_lock = threading.Lock()
_MESH_CACHE_SIZE = 2


def read_stl(src: str | bytes | bytearray) -> tuple[np.ndarray, np.ndarray]:
    """``(vertices (V, 3) float32, triangles (T, 3) int)`` of an STL file or
    its bytes, with shared vertices merged (3MF meshes are indexed).

    The last files read are kept (by path, size and mtime), so exporting the
    same result for another printer doesn't parse it again."""
    if isinstance(src, (str, os.PathLike)):
        stat = os.stat(src)
        key = (os.path.realpath(src), stat.st_size, stat.st_mtime_ns)
        with _mesh_cache_lock:
            if key in _mesh_cache:
                _mesh_cache.move_to_end(key)
                return _mesh_cache[key]
        mesh = _read_stl(src)
        with _mesh_cache_lock:
            _mesh_cache[key] = mesh
            while len(_mesh_cache) > _MESH_CACHE_SIZE:
                _mesh_cache.popitem(last=False)
        return mesh
    return _read_stl(src)


def _read_stl(src) -> tuple[np.ndarray, np.ndarray]:
    if isinstance(src, (bytes, bytearray)):
        data = bytes(src)
    else:
        with open(src, "rb") as f:
            data = f.read()
    if len(data) >= 84:
        count = struct.unpack("<I", data[80:84])[0]
        if len(data) == 84 + 50 * count:
            corners = np.frombuffer(data, dtype=_STL_RECORD, count=count, offset=84)["v"].reshape(-1, 3)
            return _index_mesh(corners)
    if data.lstrip()[:5].lower() == b"solid":
        import trimesh

        mesh = trimesh.load(io.BytesIO(data), file_type="stl", process=False)
        return _index_mesh(np.asarray(mesh.triangles, dtype=np.float32).reshape(-1, 3))
    raise ValueError("Not a valid STL file")


def _index_mesh(corners: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Merge bit-identical corners into shared vertices.

    Sorting two integer keys (x|y bits, z bits) is ~5x faster than
    np.unique on 12-byte rows, which took 7 s for a 6 M-triangle STL."""
    # + 0.0 turns -0.0 into 0.0, so both merge into one vertex.
    corners = np.ascontiguousarray(corners, dtype=np.float32) + np.float32(0.0)
    if len(corners) == 0:
        return corners.reshape(0, 3), np.zeros((0, 3), dtype=np.int64)
    bits = corners.view(np.uint32)
    xy = (bits[:, 0].astype(np.uint64) << np.uint64(32)) | bits[:, 1].astype(np.uint64)
    z = bits[:, 2]
    order = np.lexsort((z, xy))
    xy_sorted, z_sorted = xy[order], z[order]
    first = np.empty(len(order), dtype=bool)
    first[0] = True
    first[1:] = (xy_sorted[1:] != xy_sorted[:-1]) | (z_sorted[1:] != z_sorted[:-1])
    inverse = np.empty(len(order), dtype=np.int64)
    inverse[order] = np.cumsum(first) - 1
    vertices = corners[order[first]]
    return vertices, inverse.reshape(-1, 3)


def _write_rows(out, fmt: str, rows: np.ndarray, chunk: int = 100_000, done: Callable[[int], None] | None = None) -> None:
    for start in range(0, len(rows), chunk):
        part = rows[start:start + chunk].tolist()
        out.write("".join(fmt % tuple(r) for r in part).encode("ascii"))
        if done is not None:
            done(len(part))


# ------------------------------------------------------------------- settings

def _filament_preset(printer: dict[str, Any], filament_type: str) -> str:
    choices = printer["filaments"]
    wanted = str(filament_type or "PLA").upper().replace(" ", "-")
    if wanted in choices:
        return choices[wanted]
    for known in sorted(choices, key=len, reverse=True):
        if known in wanted:
            return choices[known]
    return choices["PLA"]


def _vector(value: Any) -> list:
    return list(value) if isinstance(value, list) else [value]


def project_settings(
    printer: dict[str, Any],
    slots: Sequence[Slot],
    layer_height: float,
    background_height: float,
    manual_swaps: bool,
    profiles: OrcaProfiles,
    autoforge_settings: bool = False,
    notes: list[str] | None = None,
) -> dict[str, Any]:
    """``Metadata/project_settings.config``: the printer's resolved machine,
    process and filament presets, with the heightmap settings on top (and
    AUTOFORGE_PRINT_SETTINGS when ``autoforge_settings``)."""
    vendor = printer["vendor"]
    defaults = load_printer_index().get("defaults") or {}

    def complete(config: dict[str, Any], kind: str) -> dict[str, Any]:
        # Every option, as in OrcaSlicer's own projects: one the presets
        # leave out would be taken from whatever printer OrcaSlicer has
        # loaded (another printer's bed exclusion area made delta printers
        # refuse to slice).
        for key, value in (defaults.get(kind) or {}).items():
            config.setdefault(key, value)
        return pad_variants(config, kind)

    machine = complete(profiles.resolve(vendor, "machine", printer["name"]), "machine")
    process = complete(profiles.resolve(vendor, "process", printer["process"]), "process")
    preset_names = [_filament_preset(printer, s.filament_type) for s in slots]
    filaments = [complete(profiles.resolve(vendor, "filament", n), "filament") for n in preset_names]

    config: dict[str, Any] = {}
    config.update({k: v for k, v in machine.items() if k not in _DROP_KEYS})
    config.update({k: v for k, v in process.items() if k not in _DROP_KEYS})

    filament_keys: list[str] = []
    for f in filaments:
        filament_keys.extend(k for k in f if k not in _DROP_KEYS and k not in filament_keys)
    for key in filament_keys:
        values = []
        for f in filaments:
            value = f.get(key)
            if value is None:  # take it from a filament that has it
                value = next(g[key] for g in filaments if key in g)
                if key in _FILAMENT_VARIANT_KEYS:
                    value = _resize(value, len(_vector(f.get("filament_extruder_variant", [None]))))
            values.extend(_vector(value))
        config[key] = values
    # Which filament each per-variant entry belongs to (Orca's full_config).
    config["filament_self_index"] = [
        str(i + 1) for i, f in enumerate(filaments) for _ in _vector(f.get("filament_extruder_variant", [None]))
    ]

    process_changes: dict[str, Any] = {}
    if autoforge_settings:
        variants = len(_vector(process.get("print_extruder_variant", [None])))
        for key, (mode, value) in AUTOFORGE_PRINT_SETTINGS.items():
            current = config.get(key)
            if current is None and key in _PRINT_VARIANT_KEYS:
                # Not in the profile (Orca's default): one value per variant.
                changed: Any = [str(value)] * variants
            else:
                changed = _apply_setting(current, mode, value)
            if changed != current:
                process_changes[key] = changed
    process_changes.update(PROCESS_OVERRIDES)
    first_layer = first_layer_height(background_height, layer_height, machine.get("nozzle_diameter"))
    if notes is not None and abs(first_layer - float(background_height)) > 1e-9:
        notes.append(
            f"The {float(background_height):g} mm base is thicker than this printer's nozzle can print in one layer: "
            f"it is printed as a {first_layer:g} mm first layer plus {float(layer_height):g} mm layers. "
            "The colour changes stay at the same heights."
        )
    process_changes["layer_height"] = f"{float(layer_height):g}"
    process_changes["initial_layer_print_height"] = f"{first_layer:g}"

    printer_changes: dict[str, Any] = {}
    min_heights = as_list(machine.get("min_layer_height"))
    if min_heights and any(_float(h, 0.0) > float(layer_height) for h in min_heights):
        printer_changes["min_layer_height"] = [f"{float(layer_height):g}"] * len(min_heights)
    if manual_swaps:
        pause = str(machine.get("machine_pause_gcode") or "").strip() or "M600"
        printer_changes["manual_filament_change"] = "1"
        printer_changes["change_filament_gcode"] = f"; AutoForge: swap the filament by hand\n{pause}\n"
        printer_changes["single_extruder_multi_material"] = "1"
    config.update(process_changes)
    config.update(printer_changes)
    # The prime tower is off, but OrcaSlicer still checks where it would
    # stand: its default spot (15, 220) is off the bed of every printer less
    # than 220 mm deep, and such a project refuses to slice.
    bx0, by0 = printer["bed"][0], printer["bed"][1]
    config["wipe_tower_x"] = [f"{bx0 + 5:g}"]
    config["wipe_tower_y"] = [f"{by0 + 5:g}"]
    make_orca_consistent(config, printer["vendor"] == "BBL")
    index = load_printer_index()
    fill_nil_values(config, set(index.get("nullable") or []), defaults, set(index.get("int_options") or []))
    clamp_to_orca_ranges(config, index.get("option_ranges") or {})
    config["flush_volumes_matrix"] = flush_matrix(config, [s.color for s in slots])
    config["flush_volumes_vector"] = ["140"] * (2 * len(slots))

    n = len(slots)
    config.update({
        "name": "project_settings",
        "from": "project",
        "version": BBS_VERSION,
        "printer_settings_id": printer["name"],
        "print_settings_id": printer["process"],
        "filament_settings_id": preset_names,
        "filament_colour": [s.color for s in slots],
        "filament_multi_colour": [s.color for s in slots],
        "filament_ids": [str(f.get("filament_id") or "") for f in filaments],
        "print_compatible_printers": as_list(process.get("compatible_printers")),
        "inherits_group": [""] * (n + 2),
        # Empty on purpose: for a section with a list here, OrcaSlicer resets
        # every unlisted option to its own resolution of the system preset
        # (Preset::load_external_config). Some vendor profiles resolve to
        # values it then refuses to slice (and every correction made above
        # would have to be listed); empty, it uses this complete config as is.
        "different_settings_to_system": [""] * (n + 2),
    })
    return config


# ------------------------------------------------------------- flush volumes
# OrcaSlicer's colour-based purge volumes (FlushVolCalc.cpp, the HSV formula
# the GUI falls back on), so AMS / tool-changer prints purge as much as Orca
# itself would plan. The matrix has to be there with one row and column per
# filament: without it OrcaSlicer 2.5 aborts on multi-extruder printers (it
# indexes the one-entry default). It is a single n x n block - one block per
# nozzle made the same printers fail to slice, even with all-zero volumes.

def _rgb2hsv(r: float, g: float, b: float) -> tuple[float, float, float]:
    cmax, cmin = max(r, g, b), min(r, g, b)
    delta = cmax - cmin
    if abs(delta) < 0.001:
        h = 0.0
    elif cmax == r:
        h = 60.0 * math.fmod((g - b) / delta, 6.0)
    elif cmax == g:
        h = 60.0 * ((b - r) / delta + 2)
    else:
        h = 60.0 * ((r - g) / delta + 4)
    s = 0.0 if abs(cmax) < 0.001 else delta / cmax
    return h, s, cmax


def _luminance(r: float, g: float, b: float) -> float:
    return r * 0.3 + g * 0.59 + b * 0.11


def flush_volume_rgb(src: str, dst: str) -> int:
    """Purge volume (mm³) from colour ``src`` to ``dst``, before the nozzle minimum."""
    a = [int(src[i:i + 2], 16) / 255.0 for i in (1, 3, 5)]
    b = [int(dst[i:i + 2], 16) / 255.0 for i in (1, 3, 5)]
    h1, s1, v1 = _rgb2hsv(*a)
    h2, s2, v2 = _rgb2hsv(*b)
    dx = math.cos(math.radians(h1)) * s1 * v1 - math.cos(math.radians(h2)) * s2 * v2
    dy = math.sin(math.radians(h1)) * s1 * v1 - math.sin(math.radians(h2)) * s2 * v2
    hs_dist = min(1.2, math.hypot(dx, dy))
    from_lumi, to_lumi = _luminance(*a), _luminance(*b)
    if to_lumi >= from_lumi:
        lumi_flush = (to_lumi - from_lumi) ** 0.7 * 560.0
    else:
        lumi_flush = (from_lumi - to_lumi) * 80.0
        hs_dist = min(0.67 * v2 + 0.33 * v1, hs_dist)
    hs_flush = 230.0 * hs_dist
    volume = math.sqrt(hs_flush ** 2 + lumi_flush ** 2 - 2 * hs_flush * lumi_flush * math.cos(math.radians(120.0)))
    return int(max(volume, 60.0))


def _at(values: Any, index: int, default: Any = None) -> Any:
    values = _vector(values) if values is not None else []
    if not values:
        return default
    return values[index] if index < len(values) else values[-1]


def _min_flush_volume(config: dict[str, Any], nozzle: int, filament: int) -> int:
    """Orca's get_min_flush_volumes: the nozzle volume, less the filament a
    long retraction on cut already pulls out."""
    nozzle_volume = int(_float(_at(config.get("nozzle_volume"), nozzle, 0), 0.0))
    level = int(_float(config.get("enable_long_retraction_when_cut"), 0.0))
    machine_on = str(_at(config.get("long_retractions_when_cut"), nozzle, "0")) == "1"
    retract = _float(_at(config.get("retraction_distances_when_cut"), nozzle, 18), 18.0) if level and machine_on else 0.0
    filament_on = str(_at(config.get("filament_long_retractions_when_cut"), filament, "0"))
    if filament_on == "0":
        retract = 0.0
    elif filament_on == "1" and level == 2:
        own = _at(config.get("filament_retraction_distances_when_cut"), filament, "nil")
        retract = float(int(_float(own, retract))) if own != "nil" else _float(
            _at(config.get("retraction_distances_when_cut"), nozzle, 18), 18.0)
    return int(nozzle_volume - math.pi * 1.75 * 1.75 / 4 * retract)


def flush_matrix(config: dict[str, Any], colors: Sequence[str]) -> list[str]:
    """``flush_volumes_matrix``: n x n purge volumes, row = from, column = to."""
    n = len(colors)
    out: list[str] = []
    for i in range(n):
        minimum = _min_flush_volume(config, 0, i)
        for j in range(n):
            volume = 0 if i == j else min(flush_volume_rgb(colors[i], colors[j]) + minimum, _MAX_FLUSH_VOLUME)
            out.append(str(volume))
    return out


_MAX_FLUSH_VOLUME = 20000


def first_layer_height(background_height: float, layer_height: float, nozzle_diameter: Any) -> float:
    """The first layer for a ``background_height`` base on this printer.

    OrcaSlicer refuses layers thicker than the smallest nozzle ("Layer height
    cannot exceed nozzle diameter"), so a 0.24 mm base on a 0.2 mm nozzle
    becomes a 0.16 mm first layer plus one 0.08 mm layer: whole layers below
    the base's top, so every colour change still lands on a layer top.
    ValueError when even the layer height is too thick for the nozzle."""
    nozzles = [n for n in (_float(v, 0.0) for v in nozzle_list(nozzle_diameter)) if n > 0]
    base, layer = float(background_height), float(layer_height)
    if not nozzles:
        return base
    smallest = min(nozzles)
    if layer > smallest + 1e-9:
        raise ValueError(
            f"{layer:g} mm layers are thicker than the printer's {smallest:g} mm nozzle can print; "
            "use a smaller layer height or a printer with a bigger nozzle"
        )
    if base <= smallest + 1e-9:
        return base
    first = base - math.ceil((base - smallest) / layer - 1e-9) * layer
    return round(first, 6)


def _clamp_one(value: Any, low: float | None, high: float | None) -> Any:
    text = str(value).strip()
    if text == "nil":
        return value
    percent = text.endswith("%")
    try:
        number = float(text[:-1] if percent else text)
    except ValueError:
        return value
    clamped = number
    if low is not None and clamped < low:
        clamped = low
    if high is not None and clamped > high:
        clamped = high
    if clamped == number:
        return value
    return f"{clamped:g}%" if percent else f"{clamped:g}"


_G92_ANY = re.compile(r"^[ \t]*[gG]92[ \t]*[eE](0(\.0*)?|\.0+)[ \t]*(;.*)?$", re.M)
_G92_EXACT = re.compile(r"^[ \t]*G92[ \t]*E(0(\.0*)?|\.0+)[ \t]*(;.*)?$", re.M)
# OrcaSlicer's plate types and the filament option holding each one's bed temperature.
_BED_TYPES = [
    ("Textured PEI Plate", "textured_plate_temp"), ("High Temp Plate", "hot_plate_temp"),
    ("Cool Plate", "cool_plate_temp"), ("Engineering Plate", "eng_plate_temp"),
    ("Textured Cool Plate", "textured_cool_plate_temp"), ("Supertack Plate", "supertack_plate_temp"),
]


_WALL_SEQUENCE = {
    "inner wall/outer wall/infill": "inner wall/outer wall",
    "infill/inner wall/outer wall": "inner wall/outer wall",
    "outer wall/inner wall/infill": "outer wall/inner wall",
    "infill/outer wall/inner wall": "outer wall/inner wall",
    "inner-outer-inner wall/infill": "inner-outer-inner wall",
}


def convert_legacy_options(config: dict[str, Any]) -> list[str]:
    """Rewrite the legacy options OrcaSlicer converts on load (ConfigBase::
    load_from_json + handle_legacy). For these two it also adds the new keys
    to different_settings_to_system, which makes it reset every other option
    of the project to the system preset (Prusa MK3S's wall_infill_order
    turned a whole project back into the stock 0.2 mm profile)."""
    changed = []
    order = config.pop("wall_infill_order", None)
    if order is not None:
        order = str(order)
        if order in _WALL_SEQUENCE:
            config["wall_sequence"] = _WALL_SEQUENCE[order]
        config["is_infill_first"] = "1" if order.startswith("infill/") else "0"
        changed += ["wall_infill_order", "wall_sequence", "is_infill_first"]
    if str(config.get("support_type")) == "hybrid(auto)":
        config["support_type"] = "tree(auto)"
        config["support_style"] = "tree_hybrid"
        changed += ["support_type", "support_style"]
    return changed


def make_orca_consistent(config: dict[str, Any], is_bbl: bool) -> list[str]:
    """Fix the contradictions in some vendor profiles that OrcaSlicer refuses
    to slice (Print::validate / PrintConfig validate), the way its messages
    ask. Returns the keys changed."""
    changed = convert_legacy_options(config)
    relative = str(_vector(config.get("use_relative_e_distances", "1"))[0]) == "1"
    # Some profiles write single G-code options as a one-element list.
    def text(key: str) -> str:
        value = config.get(key)
        return str((value[0] if value else "") if isinstance(value, list) else (value or ""))

    before = text("before_layer_change_gcode")
    layer = text("layer_change_gcode")
    if relative:
        # Lower-case "g92 e0" is refused; Marlin needs a reset every layer.
        def upper(m: "re.Match[str]") -> str:  # only the letters change
            return re.sub(r"^([ \t]*)[gG]92([ \t]*)[eE]", r"\1G92\2E", m.group(0))

        fixed_before = _G92_ANY.sub(upper, before)
        fixed_layer = _G92_ANY.sub(upper, layer)
        # OrcaSlicer's source limits this to Marlin, but OrcaSlicer 2.5.0 as
        # shipped also refused a Klipper profile without it (Anycubic Kobra
        # S1 Max). G92 E0 is a plain extruder reset on every firmware.
        if (not is_bbl
                and not _G92_EXACT.search(fixed_before) and not _G92_EXACT.search(fixed_layer)):
            fixed_layer = (fixed_layer.rstrip("\n") + "\n" if fixed_layer.strip() else "") + "G92 E0 ; reset extruder (relative E)\n"
    else:
        # Absolute E: OrcaSlicer tracks E itself, a reset is refused.
        fixed_before = _G92_ANY.sub("", before)
        fixed_layer = _G92_ANY.sub("", layer)
    for key, old, new in (("before_layer_change_gcode", before, fixed_before), ("layer_change_gcode", layer, fixed_layer)):
        if new != old:
            config[key] = [new] if isinstance(config.get(key), list) else new
            changed.append(key)

    # Firmware retraction can't be combined with wipe. Always listed as
    # changed: OrcaSlicer's own resolution of such a profile can have wipe
    # on even where ours has it off, and it uses its own for unlisted keys.
    if str(config.get("use_firmware_retraction", "0")) == "1":
        wipe = config.get("wipe")
        config["wipe"] = ["0"] * max(1, len(wipe) if isinstance(wipe, list) else 1)
        changed.append("wipe")
        value = config.get("filament_wipe")
        if isinstance(value, list) and any(str(v) == "1" for v in value):
            config["filament_wipe"] = ["0" if str(v) == "1" else v for v in value]
            changed.append("filament_wipe")

    # A bridge line may not be wider than the smallest nozzle.
    nozzles = [n for n in (_float(v, 0.0) for v in nozzle_list(config.get("nozzle_diameter"))) if n > 0]
    if nozzles and config.get("bridge_line_width") is not None:
        raw = str(_vector(config["bridge_line_width"])[0])
        width = _float(raw.rstrip("%"), 0.0) * (max(nozzles) / 100 if raw.endswith("%") else 1)
        if width > min(nozzles) + 1e-9:
            config["bridge_line_width"] = f"{min(nozzles):g}"
            changed.append("bridge_line_width")

    # [""] is no script, but the OrcaSlicer CLI refuses any non-empty list.
    scripts = config.get("post_process")
    if isinstance(scripts, list) and scripts and not any(str(v).strip() for v in scripts):
        config["post_process"] = []
        changed.append("post_process")

    # A plate type every filament has a bed temperature for (0 = unsupported).
    def supported(temp_key: str) -> bool:
        temps = config.get(temp_key)
        return temps is None or all(_float(t, 0.0) > 0 for t in _vector(temps))

    preferred = str(config.get("default_bed_type") or config.get("curr_bed_type") or "")
    keys = dict(_BED_TYPES)
    if not (preferred in keys and supported(keys[preferred])):
        preferred = next((name for name, key in _BED_TYPES if supported(key)), "Textured PEI Plate")
    if config.get("curr_bed_type") != preferred:
        config["curr_bed_type"] = preferred
        changed.append("curr_bed_type")
    return changed


def fill_nil_values(config: dict[str, Any], nullable: set[str], defaults: dict[str, Any],
                    int_options: set[str] | None = None) -> list[str]:
    """Replace "nil" ("use the printer's value") where OrcaSlicer refuses it:
    options that aren't nullable, and integer ones, whose nil fails its range
    check (Anycubic Kobra S1 Max's filament_flush_temp). A filament override
    gets the printer's own value, anything else OrcaSlicer's default."""
    int_options = int_options or set()
    if not nullable and not int_options:
        return []
    flat = {}
    for section in defaults.values():
        flat.update(section)
    changed = []
    for key, value in list(config.items()):
        if not isinstance(value, list) or "nil" not in value:
            continue
        if key in nullable and key not in int_options:
            continue
        base = config.get(key[len("filament_"):]) if key.startswith("filament_") else None
        fallback = _vector(base)[0] if base not in (None, [], "nil", ["nil"]) else _vector(flat.get(key, ["0"]))[0]
        if str(fallback) == "nil":
            fallback = "0"
        config[key] = [fallback if v == "nil" else v for v in value]
        changed.append(key)
    return changed


def clamp_to_orca_ranges(config: dict[str, Any], ranges: dict[str, Any]) -> list[str]:
    """Bring values into OrcaSlicer's allowed range for each option.

    Some vendor profiles carry values OrcaSlicer itself refuses a project
    for ("Param values in 3mf/config error: ... not in range", e.g. an
    exhaust fan at 255 where 0..100 is allowed). Returns the keys changed."""
    changed = []
    # Filament overrides (filament_z_hop, ...) are checked against their
    # printer option's range once OrcaSlicer applies them.
    ranges = dict(ranges)
    for key in list(config):
        if key.startswith("filament_") and key not in ranges and key[len("filament_"):] in ranges:
            ranges[key] = ranges[key[len("filament_"):]]
    for key, (low, high) in ranges.items():
        if key not in config:
            continue
        value = config[key]
        fixed = [_clamp_one(v, low, high) for v in value] if isinstance(value, list) else _clamp_one(value, low, high)
        if fixed != value:
            config[key] = fixed
            changed.append(key)
    return changed


def _float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


# ---------------------------------------------------------------------- 3mf

_CONTENT_TYPES = """<?xml version="1.0" encoding="UTF-8"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
 <Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
 <Default Extension="model" ContentType="application/vnd.ms-package.3dmanufacturing-3dmodel+xml"/>
 <Default Extension="png" ContentType="image/png"/>
 <Default Extension="json" ContentType="application/json"/>
 <Default Extension="config" ContentType="text/plain"/>
 <Default Extension="xml" ContentType="application/xml"/>
</Types>"""

_ROOT_RELS = """<?xml version="1.0" encoding="UTF-8"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
 <Relationship Target="/3D/3dmodel.model" Id="rel-1" Type="http://schemas.microsoft.com/3dmanufacturing/2013/01/3dmodel"/>
</Relationships>"""

_MODEL_RELS = """<?xml version="1.0" encoding="UTF-8"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
 <Relationship Target="/3D/Objects/object_1.model" Id="rel-1" Type="http://schemas.microsoft.com/3dmanufacturing/2013/01/3dmodel"/>
</Relationships>"""

_MODEL_NS = (
    'unit="millimeter" xml:lang="en-US" '
    'xmlns="http://schemas.microsoft.com/3dmanufacturing/core/2015/02" '
    'xmlns:BambuStudio="http://schemas.bambulab.com/package/2021" '
    'xmlns:p="http://schemas.microsoft.com/3dmanufacturing/production/2015/06" requiredextensions="p"'
)


def _xml_attr(text: str) -> str:
    return (str(text).replace("&", "&amp;").replace('"', "&quot;")
            .replace("<", "&lt;").replace(">", "&gt;"))


def _root_model(title: str, offset: Iterable[float]) -> str:
    x, y, z = offset
    uid = lambda: str(_uuid.uuid4())  # noqa: E731
    return f"""<?xml version="1.0" encoding="UTF-8"?>
<model {_MODEL_NS}>
 <metadata name="Application">BambuStudio-{BBS_VERSION}</metadata>
 <metadata name="OrcaSlicer">{ORCA_VERSION}</metadata>
 <metadata name="BambuStudio:3mfVersion">1</metadata>
 <metadata name="Title">{_xml_attr(title)}</metadata>
 <resources>
  <object id="2" p:UUID="{uid()}" type="model">
   <components>
    <component p:path="/3D/Objects/object_1.model" objectid="1" p:UUID="{uid()}" transform="1 0 0 0 1 0 0 0 1 0 0 0"/>
   </components>
  </object>
 </resources>
 <build p:UUID="{uid()}">
  <item objectid="2" p:UUID="{uid()}" transform="1 0 0 0 1 0 0 0 1 {x:.4f} {y:.4f} {z:.4f}" printable="1"/>
 </build>
</model>"""


def _model_settings(title: str) -> str:
    name = _xml_attr(title)
    return f"""<?xml version="1.0" encoding="UTF-8"?>
<config>
  <object id="2">
    <metadata key="name" value="{name}"/>
    <metadata key="extruder" value="1"/>
    <part id="1" subtype="normal_part">
      <metadata key="name" value="{name}"/>
      <metadata key="matrix" value="1 0 0 0 0 1 0 0 0 0 1 0 0 0 0 1"/>
    </part>
  </object>
  <plate>
    <metadata key="plater_id" value="1"/>
    <metadata key="plater_name" value=""/>
    <metadata key="locked" value="false"/>
    <model_instance>
      <metadata key="object_id" value="2"/>
      <metadata key="instance_id" value="0"/>
    </model_instance>
  </plate>
</config>"""


def _custom_gcodes(plan: SwapPlan) -> str:
    rows = "".join(
        f'  <layer top_z="{z:.4f}" type="2" extruder="{slot + 1}" color="{plan.slots[slot].color}" '
        f'extra="" gcode="tool_change"/>\n'
        for z, slot in plan.swaps
    )
    return (
        '<?xml version="1.0" encoding="UTF-8"?>\n<custom_gcodes_per_layer>\n <plate>\n'
        '  <plate_info id="1"/>\n' + rows + '  <mode value="MultiAsSingle"/>\n </plate>\n</custom_gcodes_per_layer>\n'
    )


Progress = Callable[[float, str], None]


class Cancelled(Exception):
    """The progress callback asked to stop."""


def export_3mf(
    out: str | io.IOBase,
    stl: str | bytes,
    plan: SwapPlan,
    printer_name: str,
    layer_height: float,
    background_height: float,
    manual_swaps: bool | None = None,
    title: str = "AutoForge",
    profiles: OrcaProfiles | None = None,
    progress: Progress | None = None,
    nozzle: float | None = None,
    autoforge_settings: bool = False,
) -> dict[str, Any]:
    """Write the project for ``printer_name`` (a name from the printer index).

    ``autoforge_settings``: also apply AUTOFORGE_PRINT_SETTINGS (more reliable
    colour-layer prints).
    ``nozzle``: the nozzle (mm) the result was made for. The printer's preset
    for that nozzle is used; without one the picked preset stays and the
    result carries a warning. Returns ``{"printer": preset used, "notes":
    [...], "warnings": [...]}``.

    ``manual_swaps``: pause for a hand swap at each change instead of a tool
    change; ``None`` picks by printer (see ``printer_changes_filament``).
    ``progress(fraction, stage)`` is called along the way; it may raise to
    stop the export. Raises ``ProfileError`` for an unknown printer or a
    profile that can't be fetched, and ``ValueError`` for a model that
    doesn't fit the bed."""
    report = progress or (lambda fraction, stage: None)
    printer, note, warning = printer_for_nozzle(printer_name, nozzle)
    if manual_swaps is None:
        manual_swaps = not printer_changes_filament(printer)
    profiles = profiles or shared_profiles()

    notes: list[str] = []
    report(0.02, "Loading the printer settings")
    with ThreadPoolExecutor(max_workers=1) as pool:
        # The presets (maybe a download) and the STL load side by side.
        settings_job = pool.submit(
            project_settings, printer, plan.slots, layer_height, background_height, manual_swaps, profiles,
            autoforge_settings, notes,
        )
        report(0.05, "Reading the model")
        vertices, triangles = read_stl(stl)
        if len(triangles) == 0:
            raise ValueError("The STL has no triangles")
        if not settings_job.done():
            report(0.15, "Loading the printer settings")
        settings = settings_job.result()

    lo, hi = vertices.min(axis=0).astype(np.float64), vertices.max(axis=0).astype(np.float64)
    size = hi - lo
    bx0, by0, bx1, by1 = printer["bed"]
    if size[0] > bx1 - bx0 + 1e-6 or size[1] > by1 - by0 + 1e-6:
        raise ValueError(
            f"The model is {size[0]:.0f} x {size[1]:.0f} mm, but the {printer['model']} bed is only "
            f"{bx1 - bx0:.0f} x {by1 - by0:.0f} mm"
        )
    height = float(printer.get("height") or 0)
    if height and size[2] > height + 1e-6:
        raise ValueError(f"The model is {size[2]:.1f} mm tall, but the {printer['model']} prints at most {height:.0f} mm")
    # Centre on the bed, resting on it.
    offset = ((bx0 + bx1) / 2 - (lo[0] + hi[0]) / 2, (by0 + by1) / 2 - (lo[1] + hi[1]) / 2, -lo[2])

    total_rows = len(vertices) + len(triangles)
    written = [0]

    def rows_done(n: int) -> None:
        written[0] += n
        report(0.2 + 0.78 * written[0] / total_rows, "Writing the 3mf")

    report(0.2, "Writing the 3mf")
    # Level 1: about 4x faster than the default for a ~25 % bigger file.
    with zipfile.ZipFile(out, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=1) as zf:
        zf.writestr("[Content_Types].xml", _CONTENT_TYPES)
        zf.writestr("_rels/.rels", _ROOT_RELS)
        zf.writestr("3D/3dmodel.model", _root_model(title, offset))
        zf.writestr("3D/_rels/3dmodel.model.rels", _MODEL_RELS)
        with zf.open("3D/Objects/object_1.model", "w", force_zip64=True) as f:
            f.write((
                f'<?xml version="1.0" encoding="UTF-8"?>\n<model {_MODEL_NS}>\n'
                ' <metadata name="BambuStudio:3mfVersion">1</metadata>\n <resources>\n'
                f'  <object id="1" p:UUID="{_uuid.uuid4()}" type="model">\n   <mesh>\n    <vertices>\n'
            ).encode("ascii"))
            _write_rows(f, '     <vertex x="%.9g" y="%.9g" z="%.9g"/>\n', vertices, done=rows_done)
            f.write(b"    </vertices>\n    <triangles>\n")
            _write_rows(f, '     <triangle v1="%d" v2="%d" v3="%d"/>\n', triangles, done=rows_done)
            f.write(b"    </triangles>\n   </mesh>\n  </object>\n </resources>\n <build/>\n</model>\n")
        zf.writestr("Metadata/model_settings.config", _model_settings(title))
        zf.writestr("Metadata/project_settings.config", json.dumps(settings, indent=4, ensure_ascii=False))
        zf.writestr("Metadata/custom_gcode_per_layer.xml", _custom_gcodes(plan))
        zf.writestr(
            "Metadata/slice_info.config",
            '<?xml version="1.0" encoding="UTF-8"?>\n<config>\n  <header>\n'
            '    <header_item key="X-BBL-Client-Type" value="slicer"/>\n'
            f'    <header_item key="X-BBL-Client-Version" value="{BBS_VERSION}"/>\n'
            f'    <header_item key="OrcaSlicer-Version" value="{ORCA_VERSION}"/>\n'
            "  </header>\n</config>\n",
        )
    report(1.0, "Done")
    return {"printer": printer["name"], "notes": ([note] if note else []) + notes, "warnings": [warning] if warning else []}


__all__ = [
    "Cancelled", "ProfileError", "Slot", "SwapPlan", "export_3mf", "printer_changes_filament",
    "project_settings", "read_stl", "swap_plan",
]
