"""Slice exported projects with the real OrcaSlicer CLI.

Opt-in: needs ``AUTOFORGE_ORCA_PROFILES`` pointing at OrcaSlicer's
``resources/profiles`` (tag ORCA_TAG) and OrcaSlicer itself, as ``orca-slicer``
on PATH or the ``com.orcaslicer.OrcaSlicer`` flatpak. Checks that the project
loads and slices, and that every swap lands on its layer in the G-code."""

import json
import os
import re
import shutil
import subprocess

import numpy as np
import pytest

from autoforge.Helper.Export3MF import export_3mf, swap_plan
from autoforge.Helper.OutputHelper import generate_stl


def _orca_command(work):
    if shutil.which("orca-slicer"):
        return ["orca-slicer"]
    if shutil.which("flatpak"):
        listed = subprocess.run(["flatpak", "list", "--app", "--columns=application"], capture_output=True, text=True)
        if "com.orcaslicer.OrcaSlicer" in listed.stdout:
            return ["flatpak", "run", f"--filesystem={work}", "--command=orca-slicer", "com.orcaslicer.OrcaSlicer"]
    return None


pytestmark = pytest.mark.skipif(
    not os.environ.get("AUTOFORGE_ORCA_PROFILES"), reason="set AUTOFORGE_ORCA_PROFILES to OrcaSlicer's resources/profiles"
)

LH, BG = 0.08, 0.24
MATERIALS = [
    {"color": "#111111", "name": "Black", "filament_type": "PLA"},
    {"color": "#FF0000", "name": "Red", "filament_type": "PLA"},
    {"color": "#00FF00", "name": "Green", "filament_type": "PETG"},
    {"color": "#FFFFFF", "name": "White", "filament_type": "PLA"},
]
STACK = [0, 0, 1, 1, 2, 2, 3, 1]


@pytest.mark.parametrize("printer, manual, layer, base", [
    ("Bambu Lab X1 Carbon 0.4 nozzle", False, LH, BG),  # AMS: tool changes
    ("Prusa MK4 0.4 nozzle", True, LH, BG),  # pause for a hand swap
    ("Snapmaker U1 (0.4 nozzle)", False, LH, BG),  # tool changer, four extruders
    # Hand swaps on a Bambu printer: OrcaSlicer skipped the first one when it
    # was done through the change-filament G-code.
    ("Bambu Lab X1 Carbon 0.4 nozzle", True, 0.12, 0.36),
    ("Prusa MK4 0.4 nozzle", True, 0.04, 0.16),
])
def test_orca_slices_the_project(tmp_path, printer, manual, layer, base):
    work = str(tmp_path)
    orca = _orca_command(work)
    if orca is None:
        pytest.skip("OrcaSlicer is not installed")
    yy, xx = np.mgrid[0:60, 0:60]
    heights = np.clip(8 - (np.hypot(yy - 30, xx - 30) // 4).astype(int), 0, 8)
    stl = os.path.join(work, "model.stl")
    generate_stl(heights.astype(np.float32) * layer, stl, base, maximum_x_y_size=30)
    plan = swap_plan(STACK, 8, layer, base, MATERIALS, 0)
    project = os.path.join(work, "project.3mf")
    export_3mf(project, stl, plan, printer, layer, base, manual_swaps=manual)

    subprocess.run(orca + ["--slice", "0", "--outputdir", work, project], capture_output=True, timeout=900, cwd=work)
    result = json.load(open(os.path.join(work, "result.json")))
    assert result["return_code"] == 0, result["error_string"]
    gcode = next(os.path.join(work, f) for f in os.listdir(work) if f.endswith(".gcode"))

    changes, layers, z = [], [], None
    for line in open(gcode, encoding="utf-8", errors="replace"):
        m = re.match(r";\s*Z_HEIGHT:\s*([\d.]+)", line) or re.match(r";Z:([\d.]+)", line)
        if m:
            z = round(float(m.group(1)), 2)
            layers.append(z)
        elif manual and line.startswith("; AutoForge: swap the filament by hand"):
            changes.append(z)
        elif not manual and (t := re.match(r"^[ \t]*T([0-9])\b", line)) and z is not None and z >= base:
            changes.append((z, int(t.group(1))))
    # The project's layer and base heights are what OrcaSlicer prints.
    assert layers[0] == pytest.approx(base) and layers[1] == pytest.approx(base + layer)
    if manual:
        assert changes == [round(z, 2) for z, _ in plan.swaps]
    else:
        assert changes == [(round(z, 2), slot) for z, slot in plan.swaps]
