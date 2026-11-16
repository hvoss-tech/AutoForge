"""Every printer in the bundled index, sliced with the real OrcaSlicer.

NOT part of the normal test run: it slices over a thousand projects and takes
20-60 minutes. Skipped unless switched on:

    AUTOFORGE_ORCA_SWEEP=default pytest tests/test_orca_printer_sweep.py -q

AUTOFORGE_ORCA_SWEEP picks the modes, comma separated, or ``all``:

  default            every printer as exported by default (AutoForge print
                     settings on, hand swaps unless the printer changes
                     filament itself)
  no-print-settings  every printer with the AutoForge print settings off
  hand-swaps         the printers that change filament themselves (AMS,
                     MMU, tool changer), with hand swaps instead

Other settings:

  AUTOFORGE_ORCA_PROFILES        OrcaSlicer's resources/profiles at ORCA_TAG
                                 (else the profiles are downloaded from GitHub)
  AUTOFORGE_ORCA_SWEEP_WORKERS   OrcaSlicer processes at once (default 4)
  AUTOFORGE_ORCA_SWEEP_DIR       keep projects and G-code there (default: a
                                 temporary directory, removed afterwards)
  AUTOFORGE_ORCA_SWEEP_LAYER / _BASE   layer and base height (0.08 / 0.24)

Needs OrcaSlicer: ``orca-slicer`` on PATH or the com.orcaslicer.OrcaSlicer
flatpak. There is one test per printer and mode, so ``-k "Prusa"`` or
``--lf`` rerun a part. All selected projects are sliced together first (in
parallel); whatever fails is sliced again on its own, as OrcaSlicer processes
running side by side occasionally fail at random.

Each test checks that OrcaSlicer loads and slices the project without error,
that the base and the first colour layer have the right heights, and that
every colour change is at its height (a pause for hand swaps, else a tool
change).
"""

import os
import re
import tempfile
from concurrent.futures import ThreadPoolExecutor

import pytest

from autoforge.Helper.Export3MF import export_3mf, printer_changes_filament
from autoforge.Helper.OrcaProfiles import load_printer_index, shared_profiles
from orca_slicing import check_slice, make_model, orca_command, slice_project

MODES = ("default", "no-print-settings", "hand-swaps")
_setting = os.environ.get("AUTOFORGE_ORCA_SWEEP", "").strip()
SELECTED = MODES if _setting == "all" else tuple(m.strip() for m in _setting.split(",") if m.strip())

pytestmark = pytest.mark.skipif(
    not SELECTED, reason="the printer sweep runs only with AUTOFORGE_ORCA_SWEEP set (see the module docstring)"
)

LAYER = float(os.environ.get("AUTOFORGE_ORCA_SWEEP_LAYER", "0.08"))
BASE = float(os.environ.get("AUTOFORGE_ORCA_SWEEP_BASE", "0.24"))
WORKERS = int(os.environ.get("AUTOFORGE_ORCA_SWEEP_WORKERS", "4"))


def _cases():
    unknown = set(SELECTED) - set(MODES)
    if unknown:
        raise ValueError(f"AUTOFORGE_ORCA_SWEEP: unknown mode(s) {sorted(unknown)}; use {', '.join(MODES)} or all")
    cases = []
    for mode in SELECTED:
        for printer in load_printer_index()["printers"]:
            if mode == "hand-swaps" and not printer_changes_filament(printer):
                continue
            cases.append(pytest.param(mode, printer["name"], id=f"{mode}::{printer['name']}"))
    return cases


def _settings(mode: str, printer: dict) -> tuple[bool, bool]:
    """(manual swaps, AutoForge print settings) for a mode."""
    manual = True if mode == "hand-swaps" else not printer_changes_filament(printer)
    return manual, mode != "no-print-settings"


def _slug(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", text).strip("_")


@pytest.fixture(scope="session")
def sweep(request):
    """Slice every selected (mode, printer) of this run; {(mode, name): problem or None}."""
    keep = os.environ.get("AUTOFORGE_ORCA_SWEEP_DIR")
    holder = None if keep else tempfile.TemporaryDirectory(prefix="orca-sweep-")
    root = os.path.abspath(keep or holder.name)
    os.makedirs(root, exist_ok=True)
    orca = orca_command(root)
    if orca is None:
        pytest.skip("OrcaSlicer is not installed (orca-slicer on PATH or the flatpak)")

    wanted = [
        tuple(item.callspec.params[k] for k in ("mode", "printer"))
        for item in request.session.items
        if item.originalname == "test_printer_slices" and hasattr(item, "callspec")
    ]
    printers = {p["name"]: p for p in load_printer_index()["printers"]}
    stl = os.path.join(root, "model.stl")
    plan = make_model(stl, LAYER, BASE)
    profiles = shared_profiles()

    def run(case):
        mode, name = case
        printer = printers[name]
        manual, print_settings = _settings(mode, printer)
        work = os.path.join(root, mode, _slug(name))
        os.makedirs(work, exist_ok=True)
        project = os.path.join(work, "project.3mf")
        try:
            export_3mf(project, stl, plan, name, LAYER, BASE, manual_swaps=manual, profiles=profiles,
                       autoforge_settings=print_settings)
        except Exception as e:  # noqa: BLE001 - reported as the test's failure
            return case, f"export failed: {e}"
        result, gcode = slice_project(orca, project, os.path.join(work, "slice0"))
        problem = check_slice(result, gcode, plan, manual, LAYER, BASE)
        # A failure is sliced again on its own below.
        return case, None if problem is None else ("retry", problem)

    results = {}
    with ThreadPoolExecutor(max_workers=max(1, WORKERS)) as pool:
        for case, outcome in pool.map(run, wanted):
            results[case] = outcome
    # OrcaSlicer processes side by side occasionally fail at random: slice
    # every failure again, one at a time.
    for case, outcome in list(results.items()):
        if isinstance(outcome, tuple):
            mode, name = case
            printer = printers[name]
            manual, _ = _settings(mode, printer)
            work = os.path.join(root, mode, _slug(name))
            result, gcode = slice_project(orca, os.path.join(work, "project.3mf"), os.path.join(work, "slice1"))
            results[case] = check_slice(result, gcode, plan, manual, LAYER, BASE)
    yield results
    if holder is not None:
        holder.cleanup()


@pytest.mark.parametrize("mode, printer", _cases() if SELECTED else [])
def test_printer_slices(sweep, mode, printer):
    problem = sweep[(mode, printer)]
    assert problem is None, f"{printer} ({mode}): {problem}"
