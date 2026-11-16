"""Slice AutoForge slicer projects with the real OrcaSlicer CLI and read the G-code.

Shared by test_orca_slicer_integration.py (a few printers) and
test_orca_printer_sweep.py (every printer).
"""

import json
import os
import re
import shutil
import subprocess

import numpy as np

from autoforge.Helper.Export3MF import swap_plan
from autoforge.Helper.OutputHelper import generate_stl

MATERIALS = [
    {"color": "#111111", "name": "Black", "filament_type": "PLA"},
    {"color": "#FF0000", "name": "Red", "filament_type": "PLA"},
    {"color": "#00FF00", "name": "Green", "filament_type": "PETG"},
    {"color": "#FFFFFF", "name": "White", "filament_type": "PLA"},
]
# Black base, then red, green, white and red again: four swaps, one of them
# back to a filament already used.
STACK = [0, 0, 1, 1, 2, 2, 3, 1]
SWAP_MARKER = "; AutoForge: swap the filament by hand"


def orca_command(work_dir: str) -> list[str] | None:
    """The OrcaSlicer CLI: ``orca-slicer`` on PATH or the flatpak (which gets
    access to ``work_dir``). None when neither is installed."""
    if shutil.which("orca-slicer"):
        return ["orca-slicer"]
    if shutil.which("flatpak"):
        listed = subprocess.run(["flatpak", "list", "--app", "--columns=application"], capture_output=True, text=True)
        if "com.orcaslicer.OrcaSlicer" in listed.stdout:
            return ["flatpak", "run", f"--filesystem={work_dir}", "--command=orca-slicer", "com.orcaslicer.OrcaSlicer"]
    return None


def make_model(path: str, layer: float, base: float, size_mm: float = 30):
    """A 30 mm heightmap STL (rings of 0..8 layers) and its swap plan."""
    yy, xx = np.mgrid[0:60, 0:60]
    heights = np.clip(8 - (np.hypot(yy - 30, xx - 30) // 4).astype(int), 0, 8)
    generate_stl(heights.astype(np.float32) * layer, path, base, maximum_x_y_size=size_mm)
    return swap_plan(STACK, 8, layer, base, MATERIALS, 0)


def slice_project(orca: list[str], project: str, out_dir: str, timeout: int = 900) -> tuple[dict, str | None]:
    """Slice plate 1 of ``project`` into ``out_dir``: (result.json, G-code path or None)."""
    os.makedirs(out_dir, exist_ok=True)
    try:
        subprocess.run(orca + ["--slice", "0", "--outputdir", out_dir, project],
                       capture_output=True, timeout=timeout, cwd=out_dir)
    except subprocess.TimeoutExpired:
        return {"return_code": "timeout", "error_string": f"OrcaSlicer took more than {timeout} s"}, None
    result_path = os.path.join(out_dir, "result.json")
    result = json.load(open(result_path)) if os.path.exists(result_path) else {
        "return_code": "no result", "error_string": "OrcaSlicer wrote no result.json (crashed?)"}
    gcode = next((os.path.join(out_dir, f) for f in sorted(os.listdir(out_dir)) if f.endswith(".gcode")), None)
    return result, gcode


def read_gcode(gcode: str, manual: bool, base: float) -> tuple[list[float], list]:
    """(layer print_z values, swaps): for hand swaps the height of each pause
    (our marker), else (height, tool) of each tool change above the base.
    Bambu writes "; Z_HEIGHT:", others ";Z:"; some put the T command indented."""
    layers, swaps, z = [], [], None
    with open(gcode, encoding="utf-8", errors="replace") as f:
        for line in f:
            m = re.match(r";\s*Z_HEIGHT:\s*([\d.]+)", line) or re.match(r";Z:([\d.]+)", line)
            if m:
                z = round(float(m.group(1)), 2)
                layers.append(z)
            elif manual:
                if line.startswith(SWAP_MARKER):
                    swaps.append(z)
            elif (t := re.match(r"^[ \t]*T([0-9]+)\b", line)) and z is not None and z >= base:
                swaps.append((z, int(t.group(1))))
    return layers, swaps


def check_slice(result: dict, gcode: str | None, plan, manual: bool, layer: float, base: float) -> str | None:
    """What's wrong with a slice, or None when it is right."""
    if result.get("return_code") != 0:
        return f"OrcaSlicer refused it ({result.get('return_code')}): {result.get('error_string')}"
    if gcode is None:
        return "OrcaSlicer wrote no G-code"
    layers, swaps = read_gcode(gcode, manual, base)
    # The base top is a layer top (the first one, or a later one when a base
    # too thick for the nozzle is split into layers), and the colour layers
    # above it are one layer height each.
    top = round(base, 2)
    if top not in layers or layers[0] > top:
        return f"base top {top} is no layer top (first layers {layers[:4]})"
    i = layers.index(top)
    if i + 1 >= len(layers) or abs(layers[i + 1] - round(base + layer, 2)) > 1e-6:
        return f"layer after the base at {layers[i + 1:i + 2]}, expected {round(base + layer, 2)}"
    expected = [round(z, 2) for z, _ in plan.swaps]
    if manual:
        if swaps != expected:
            return f"pauses at {swaps}, expected {expected}"
    else:
        found = [z for z, _ in swaps]
        missing = [z for z in expected if z not in found]
        if missing:
            return f"no tool change at {missing} (tool changes at {swaps[:12]})"
    return None
