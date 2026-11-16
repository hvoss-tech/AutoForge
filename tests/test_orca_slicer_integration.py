"""Slice exported projects with the real OrcaSlicer CLI.

Opt-in: needs ``AUTOFORGE_ORCA_PROFILES`` pointing at OrcaSlicer's
``resources/profiles`` (tag ORCA_TAG) and OrcaSlicer itself, as ``orca-slicer``
on PATH or the ``com.orcaslicer.OrcaSlicer`` flatpak. Checks that the project
loads and slices, that the layer and base heights are what OrcaSlicer prints,
and that every swap lands on its layer. For every printer at once see
test_orca_printer_sweep.py."""

import os

import pytest

from autoforge.Helper.Export3MF import export_3mf
from orca_slicing import check_slice, make_model, orca_command, slice_project

pytestmark = pytest.mark.skipif(
    not os.environ.get("AUTOFORGE_ORCA_PROFILES"), reason="set AUTOFORGE_ORCA_PROFILES to OrcaSlicer's resources/profiles"
)

LH, BG = 0.08, 0.24


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
    orca = orca_command(work)
    if orca is None:
        pytest.skip("OrcaSlicer is not installed")
    stl = os.path.join(work, "model.stl")
    plan = make_model(stl, layer, base)
    project = os.path.join(work, "project.3mf")
    export_3mf(project, stl, plan, printer, layer, base, manual_swaps=manual)
    result, gcode = slice_project(orca, project, os.path.join(work, "out"))
    assert check_slice(result, gcode, plan, manual, layer, base) is None
