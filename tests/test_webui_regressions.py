"""Regression tests for webui bugs found and fixed during the 2026-09-16 audit.

Bug 1: ``run_pipeline()`` always blanks ``args.csv_file``/``args.json_file``
       (materials for a webui run come from the ``active_filaments`` list,
       not a CSV on disk), but ``export_results()`` used
       ``has_material_file = bool(args.csv_file) or bool(args.json_file)``
       to decide whether to write ``project_file.hfp`` — which was
       therefore always ``False`` for every webui-run job.
       ``GET /api/outputs/project/{job_id}`` (the "Download Project"
       button) 404'd for every completed webui run, silently. Fixed by
       writing a materials CSV from ``active_filaments`` at export time and
       pointing ``args.csv_file`` at it before calling
       ``generate_project_file`` (src/autoforge/webui/helpers/pipeline_runner.py).

Bug 2: ``api/outputs.py`` built filesystem paths directly from an
       unvalidated ``job_id`` URL segment, unlike
       ``services/image_service.py`` which guards against path traversal.
       Fixed by adding the same realpath-prefix check to
       ``_job_path()``/``export_project()``.

Bug 3: job failures (including a CUDA OOM) surfaced ``str(exc)`` verbatim —
       for an OOM that's a multi-paragraph allocator dump with no
       actionable takeaway. ``helpers.pipeline_runner.friendly_error_message``
       now leads with a plain-language summary + concrete next step for an
       OOM specifically, keeping the raw text available after a blank line.

Bug 4: the webui version shown in the frontend (and returned by
       ``/api/system/version``) was either a hand-edited literal in the
       frontend (already fixed separately) or ``importlib.metadata``'s
       installed-package snapshot, which goes stale the moment
       pyproject.toml's version is bumped without reinstalling.
       ``api.system._current_version`` now reads pyproject.toml directly
       when running from a source checkout (this project's normal
       deployment), falling back to installed package metadata otherwise.

Bug 5: the untracked webui test suite (tests/webui/*.py,
       tests/test_webui_backend.py) had no isolation from the real,
       CWD-relative checkpoints/uploads/filament_library directories a
       real, interactively-run webui session reads from — confirmed in
       practice (filament_library/library.json and active.json had
       accumulated dozens of "Test"/"OptGuardFilament"/"VerifyTest" junk
       entries from past test runs, which then showed up as real Active
       Filaments in a real session). Fixed with tests/webui/conftest.py and
       an equivalent fixture in test_webui_backend.py that redirect all
       webui storage to a throwaway temp directory for the test session.
"""

import argparse
import csv
import json
import os

import numpy as np
import pytest
import torch

from autoforge.Modules.Optimizer import FilamentOptimizer
from autoforge.webui.helpers.pipeline_runner import export_results, friendly_error_message
from autoforge.webui.api.outputs import _job_path
from autoforge.webui.api.system import _current_version, _version_from_pyproject, _PYPROJECT_PATH


def _args(**overrides):
    base = dict(
        max_layers=8,
        layer_height=0.2,
        learning_rate=2e-2,
        init_tau=1.0,
        final_tau=0.02,
        iterations=80,
        warmup_fraction=0.2,
        learning_rate_warmup_fraction=0.2,
        visualize=False,
        tensorboard=False,
        run_name="",
        disable_visualization_for_gradio=1,
        output_folder="/tmp",
        background_height=0.4,
        background_color="#000000",
        spike_threshold_layers=1,
        spike_removal_passes=4,
        stl_output_size=20,
        cap_layers=0,
        flatforge=False,
        perform_pruning=False,
        csv_file="",
        json_file="",
    )
    base.update(overrides)
    ns = argparse.Namespace()
    for k, v in base.items():
        setattr(ns, k, v)
    return ns


def _active_filaments():
    return [
        {
            "color": "#ff0000", "td": 1.5, "name": "Acme - Red",
            "brand": "Acme", "short_name": "Red", "owned": True,
            "uuid": "u-red", "filament_type": "PLA",
        },
        {
            "color": "#00ff00", "td": 2.5, "name": "Acme - Green",
            "brand": "Acme", "short_name": "Green", "owned": False,
            "uuid": "u-green", "filament_type": "PETG",
        },
    ]


def _make_result(tmp_path, active_filaments, H=16, W=16, seed=0):
    """Build a minimal-but-real state dict matching what run_pipeline()
    hands to export_results(), with just enough training (a handful of
    step(record_best=True) calls, same pattern as test_optimizer_training.py)
    to populate best_params so export doesn't hit a None dereference."""
    M = max(1, len(active_filaments))
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device("cpu")
    target = torch.rand(H, W, 3) * 255
    pixel_height_logits_init = np.zeros((H, W), dtype=np.float32)
    pixel_height_labels = np.zeros((H, W), dtype=np.int32)
    global_logits_init = np.random.randn(8, M).astype(np.float32)
    material_colors = torch.rand(M, 3)
    material_TDs = torch.rand(M) * 4.0 + 1.0
    background = torch.zeros(3)

    args = _args(output_folder=str(tmp_path))
    optimizer = FilamentOptimizer(
        args,
        target,
        pixel_height_logits_init,
        pixel_height_labels,
        global_logits_init,
        material_colors,
        material_TDs,
        background,
        device,
        perception_loss_module=None,
    )
    for _ in range(5):
        optimizer.step(record_best=True)

    return {
        "optimizer": optimizer,
        "args": args,
        "device": device,
        "material_colors_np": material_colors.numpy(),
        "material_TDs_np": material_TDs.numpy(),
        "material_names": [f["name"] for f in active_filaments],
        "active_filaments": active_filaments,
        "alpha": None,
        "output_target": target,
        "focus_map_full": None,
        "focus_map_proc": None,
        "pixel_height_logits_init": pixel_height_logits_init,
        "pixel_height_labels": pixel_height_labels,
    }


def test_export_results_writes_project_file_from_active_filaments(tmp_path):
    """Bug 1: project_file.hfp must be written for webui runs even though
    args.csv_file/json_file are always blank (materials come from
    active_filaments, not a CSV)."""
    result = _make_result(tmp_path, _active_filaments())
    out = export_results(result)

    assert out["project_file"] is not None
    assert os.path.exists(out["project_file"])

    with open(out["project_file"]) as f:
        project_data = json.load(f)

    names = {entry["Name"] for entry in project_data["filament_set"]}
    assert names & {"Red", "Green"}


def test_export_results_materials_csv_matches_active_filaments(tmp_path):
    """The materials.csv written for generate_project_file() must carry the
    real brand/name/color/td/type of each active filament, not the combined
    'Brand - Name' display string, so the project file's filament_set is
    correct instead of merely non-empty."""
    filaments = _active_filaments()
    result = _make_result(tmp_path, filaments)
    export_results(result)

    csv_path = os.path.join(str(tmp_path), "materials.csv")
    assert os.path.exists(csv_path)
    with open(csv_path, newline="") as f:
        rows = list(csv.DictReader(f))

    assert len(rows) == len(filaments)
    assert rows[0]["Brand"] == "Acme"
    assert rows[0]["Name"] == "Red"
    assert rows[0]["Color"] == "#ff0000"
    assert float(rows[0]["Transmissivity"]) == 1.5
    assert rows[0]["Type"] == "PLA"
    assert rows[1]["Name"] == "Green"


def test_job_path_rejects_traversal_outside_checkpoints(tmp_path, monkeypatch):
    """Bug 2: a job_id of '..' (or anything resolving outside
    checkpoints_path) must be rejected rather than silently resolved."""
    from autoforge.webui import config as config_module

    checkpoints_dir = tmp_path / "checkpoints"
    os.makedirs(checkpoints_dir)
    monkeypatch.setattr(config_module.config, "checkpoints_dir", str(checkpoints_dir))

    secret_dir = tmp_path / "secret"
    os.makedirs(secret_dir)
    with open(secret_dir / "final_model.stl", "w") as f:
        f.write("not yours")

    assert _job_path("../secret", "final_model.stl") is None


def test_job_path_resolves_legitimate_job(tmp_path, monkeypatch):
    from autoforge.webui import config as config_module

    checkpoints_dir = tmp_path / "checkpoints"
    job_dir = checkpoints_dir / "job-123"
    os.makedirs(job_dir)
    monkeypatch.setattr(config_module.config, "checkpoints_dir", str(checkpoints_dir))
    with open(job_dir / "final_model.stl", "w") as f:
        f.write("stl data")

    resolved = _job_path("job-123", "final_model.stl")
    assert resolved is not None
    assert os.path.realpath(resolved) == os.path.realpath(str(job_dir / "final_model.stl"))


def test_stl_download_route_blocks_traversal_end_to_end(tmp_path, monkeypatch):
    """Same as test_job_path_rejects_traversal_outside_checkpoints but
    through the real HTTP route, proving the download endpoint itself is
    blocked (not just the helper in isolation) and that a file which *does*
    exist at the un-guarded location isn't served."""
    from fastapi.testclient import TestClient
    from autoforge.webui import create_app
    from autoforge.webui import config as config_module

    checkpoints_dir = tmp_path / "checkpoints"
    os.makedirs(checkpoints_dir)
    with open(tmp_path / "final_model.stl", "w") as f:
        f.write("SECRET STL CONTENTS")
    monkeypatch.setattr(config_module.config, "checkpoints_dir", str(checkpoints_dir))

    client = TestClient(create_app())
    # ".." as a single URL path segment, percent-encoded so the HTTP client
    # doesn't collapse it away before the request is even sent (a literal
    # "/api/outputs/stl/.." gets normalised client-side to "/api/outputs",
    # which wouldn't exercise the route at all).
    resp = client.get("/api/outputs/stl/%2e%2e")
    assert resp.status_code == 404
    assert b"SECRET" not in resp.content


# ---------------------------------------------------------------------------
# Bug 3: friendly_error_message
# ---------------------------------------------------------------------------


def test_friendly_error_message_passes_through_ordinary_exceptions():
    exc = RuntimeError("input image not found: whatever.png")
    assert friendly_error_message(exc) == str(exc)


def test_friendly_error_message_detects_oom_by_exception_type():
    exc = torch.OutOfMemoryError("CUDA out of memory. Tried to allocate 2.00 GiB")
    msg = friendly_error_message(exc)
    assert msg.startswith("Out of GPU memory.")
    assert "Max Layers" in msg
    # Raw text is preserved after a blank-line separator for anyone who
    # wants it (Preview3DPanel puts it behind a "Show details" toggle).
    assert "\n\nCUDA out of memory. Tried to allocate 2.00 GiB" in msg


def test_friendly_error_message_detects_oom_by_message_text():
    # Some out-of-memory failures surface as a plain RuntimeError rather
    # than torch.OutOfMemoryError (e.g. raised from third-party code) —
    # detection must not depend solely on the exception type.
    exc = RuntimeError("CUDA out of memory. Tried to allocate 512.00 MiB")
    msg = friendly_error_message(exc)
    assert msg.startswith("Out of GPU memory.")


# ---------------------------------------------------------------------------
# Bug 4: version read from pyproject.toml, not stale installed metadata
# ---------------------------------------------------------------------------


def test_version_from_pyproject_matches_the_real_file():
    assert os.path.exists(_PYPROJECT_PATH)
    with open(_PYPROJECT_PATH) as f:
        text = f.read()
    import re
    match = re.search(r'(?m)^version\s*=\s*"([^"]+)"', text)
    assert match is not None
    assert _version_from_pyproject() == match.group(1)


def test_current_version_prefers_pyproject_over_installed_metadata():
    # Regression for the exact drift this bug caused: at the time this test
    # was written, the installed package metadata reported 1.9.5 while
    # pyproject.toml already said 1.9.7 (and the frontend's now-removed
    # hardcoded literal said 1.9.4) — three different answers to "what
    # version is this?" depending which one you asked.
    assert _current_version() == _version_from_pyproject()


def test_version_from_pyproject_returns_none_for_a_missing_file(monkeypatch):
    import autoforge.webui.api.system as system_module

    monkeypatch.setattr(system_module, "_PYPROJECT_PATH", "/nonexistent/pyproject.toml")
    assert system_module._version_from_pyproject() is None
    # _current_version() must still resolve to *something* via the
    # importlib.metadata fallback rather than raising.
    assert system_module._current_version()


# ---------------------------------------------------------------------------
# Bug 6: get_optimization_service() ignored config.checkpoints_path
# ---------------------------------------------------------------------------


def test_optimization_service_honors_checkpoints_path_override(tmp_path, monkeypatch):
    """get_optimization_service() used to hardcode
    OptimizationService(checkpoints_dir="checkpoints") — always relative to
    the process's CWD, never config.checkpoints_path — so nothing that
    overrode config (e.g. tests/webui/conftest.py's storage-isolation
    fixture, or AUTOFORGE_WEBUI_CHECKPOINTS_DIR) actually redirected job
    history. In practice that meant every /api/optimize/start call made by
    the (supposedly isolated) webui test suite still wrote real job
    records into the real, shared checkpoints/history.json — confirmed: it
    had accumulated 28 fake "Input image not found: whatever.png" entries
    from automated test runs, which the frontend's loadCurrentJob() then
    surfaced as if they'd just happened."""
    import autoforge.webui.services.optimization_service as opt_service_module
    from autoforge.webui import config as config_module

    isolated_checkpoints = tmp_path / "isolated_checkpoints"
    monkeypatch.setattr(config_module.config, "checkpoints_dir", str(isolated_checkpoints))
    opt_service_module.reset_service()
    try:
        svc = opt_service_module.get_optimization_service()
        svc.create_job({"iterations": 1}, job_id="probe-job")
        svc.update_status("probe-job", "completed")

        assert os.path.exists(isolated_checkpoints / "history.json")
    finally:
        opt_service_module.reset_service()
