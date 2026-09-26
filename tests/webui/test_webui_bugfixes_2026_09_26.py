"""WebUI regressions for the 2026-09-26 audit. Each fails on the pre-fix code."""

import argparse
import time

import pytest
import torch
from fastapi.testclient import TestClient

from autoforge.Modules.Optimizer import FilamentOptimizer
from autoforge.webui.server import create_app


@pytest.fixture
def client():
    return TestClient(create_app())


def _active_filament(client, uuid="bf26-fil-1"):
    body = {"brand": "T", "name": f"n-{uuid}", "color": "#123456", "td": 5.0,
            "filament_type": "PLA", "uuid": uuid}
    client.post("/api/filaments", json=body)
    client.post("/api/filaments/active", json=body)


def _wait_for_status(client, job_id, status, timeout=10.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        r = client.get(f"/api/optimize/status/{job_id}")
        if r.is_success and r.json().get("status") == status:
            return r.json()
        time.sleep(0.05)
    raise AssertionError(f"job {job_id} never reached {status!r}")


def _wait_until(predicate, timeout=10.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        if predicate():
            return
        time.sleep(0.05)
    raise AssertionError("condition never became true")


# ---------------------------------------------------------------------------
# A cancelled or failed prune gives the result back, unpruned
# ---------------------------------------------------------------------------


class _Optimizer:
    """Just enough optimizer for the prune endpoint, snapshotting with the
    real FilamentOptimizer methods."""

    solution_snapshot = FilamentOptimizer.solution_snapshot
    restore_solution_snapshot = FilamentOptimizer.restore_solution_snapshot

    def __init__(self):
        self.best_params = {
            "global_logits": torch.zeros(8, 3),
            "pixel_height_logits": torch.zeros(4, 4),
            "height_offsets": torch.zeros(2),
        }
        self.pixel_height_logits = torch.zeros(4, 4)
        self.max_layers = 8
        self.best_seed = 11
        self._prune_runs = 0
        self.preview_callback = None

    def get_discretized_solution(self, best=True):
        return torch.zeros(self.max_layers, dtype=torch.long), torch.ones(4, 4)

    def get_best_discretized_image(self):
        return torch.zeros(4, 4, 3)


def _half_prune(optimizer):
    """What an interrupted prune leaves behind: fewer layers, new params."""
    optimizer._prune_runs += 1
    optimizer.best_params = {
        "global_logits": torch.ones(5, 3),
        "pixel_height_logits": torch.ones(4, 4),
        "height_offsets": torch.ones(2),
    }
    optimizer.pixel_height_logits = torch.ones(4, 4)
    optimizer.max_layers = 5
    optimizer.best_seed = 99


@pytest.fixture
def prune_setup(client, monkeypatch):
    from autoforge.webui.services.optimization_service import get_optimization_service
    import autoforge.Helper.PruningHelper as pruning_helper
    import autoforge.webui.api.ws as ws
    import autoforge.webui.helpers.sliders as sliders

    svc = get_optimization_service()
    optimizer = _Optimizer()
    done = svc.create_job({"iterations": 10, "input_image": "a.png"})
    svc.update_status(done.job_id, "completed")
    svc.set_pipeline_result(done.job_id, {
        "optimizer": optimizer, "args": argparse.Namespace(spike_removal=False),
    })
    monkeypatch.setattr(pruning_helper, "_compute_loss_for_heightmap", lambda *_a, **_k: 1.0)

    broadcasts = []
    monkeypatch.setattr(
        sliders, "derive_sliders_from_result",
        lambda result: {"sliders": [{"layers": int(result["optimizer"].max_layers)}],
                        "min_layer": 0, "max_layer": int(result["optimizer"].max_layers)},
    )
    monkeypatch.setattr(
        ws, "broadcast_preview",
        lambda image, job_id, sliders=None, **kw: broadcasts.append((job_id, sliders)),
    )
    return svc, done.job_id, optimizer, broadcasts


def _assert_restored(svc, job_id, optimizer):
    assert optimizer.max_layers == 8
    assert optimizer.best_seed == 11
    assert torch.equal(optimizer.best_params["global_logits"], torch.zeros(8, 3))
    assert torch.equal(optimizer.pixel_height_logits, torch.zeros(4, 4))
    assert optimizer._prune_runs == 0
    # The job the frontend is still showing owns its result again: it can be
    # re-colored and pruned instead of answering "pruned since".
    result = svc.get_pipeline_result(job_id)
    assert result is not None and result["optimizer"] is optimizer


def test_cancelled_prune_restores_the_result_and_hands_it_back(client, prune_setup, monkeypatch):
    import autoforge.webui.helpers.pipeline_runner as pipeline_runner

    svc, job_id, optimizer, broadcasts = prune_setup

    def cancelled_export(result, cancel_event=None, pause_event=None, apply_spike_removal=True):
        _half_prune(result["optimizer"])
        return {"pruning_completed": False}

    monkeypatch.setattr(pipeline_runner, "export_results", cancelled_export)
    r = client.post("/api/pruning/start", json={"job_id": job_id})
    assert r.status_code == 200, r.text
    _wait_for_status(client, r.json()["job_id"], "cancelled")
    _wait_until(lambda: svc.get_pipeline_result(job_id) is not None)
    _assert_restored(svc, job_id, optimizer)
    # The restored (8-layer) stack is pushed so the frontend's sliders follow.
    assert broadcasts and broadcasts[-1] == (job_id, [{"layers": 8}])


def test_failed_prune_restores_the_result_and_hands_it_back(client, prune_setup, monkeypatch):
    import autoforge.webui.helpers.pipeline_runner as pipeline_runner

    svc, job_id, optimizer, _broadcasts = prune_setup

    def failing_export(result, cancel_event=None, pause_event=None, apply_spike_removal=True):
        _half_prune(result["optimizer"])
        raise RuntimeError("CUDA out of memory")

    monkeypatch.setattr(pipeline_runner, "export_results", failing_export)
    r = client.post("/api/pruning/start", json={"job_id": job_id})
    assert r.status_code == 200, r.text
    _wait_for_status(client, r.json()["job_id"], "failed")
    _wait_until(lambda: svc.get_pipeline_result(job_id) is not None)
    _assert_restored(svc, job_id, optimizer)


def test_completed_prune_still_keeps_its_result(client, prune_setup, monkeypatch):
    import autoforge.webui.helpers.pipeline_runner as pipeline_runner

    svc, job_id, optimizer, _broadcasts = prune_setup

    def completed_export(result, cancel_event=None, pause_event=None, apply_spike_removal=True):
        _half_prune(result["optimizer"])
        return {"pruning_completed": True}

    monkeypatch.setattr(pipeline_runner, "export_results", completed_export)
    r = client.post("/api/pruning/start", json={"job_id": job_id})
    prune_id = r.json()["job_id"]
    _wait_for_status(client, prune_id, "completed")
    assert optimizer.max_layers == 5
    assert svc.get_pipeline_result(prune_id)["optimizer"] is optimizer
    assert svc.get_pipeline_result(job_id) is None


def test_return_pipeline_result_only_from_the_current_owner():
    from autoforge.webui.services.optimization_service import OptimizationService

    svc = OptimizationService(checkpoints_dir="/nonexistent-bf26")
    svc.set_pipeline_result("job", {"optimizer": object()})
    svc.claim_pipeline_result("prune-a", "job")
    assert svc.return_pipeline_result("job", "prune-b") is False
    assert svc.get_pipeline_result("job") is None
    assert svc.return_pipeline_result("job", "prune-a") is True
    assert svc.get_pipeline_result("job") is not None


# ---------------------------------------------------------------------------
# No optimization or pruning while the auto-preview is being built
# ---------------------------------------------------------------------------


@pytest.fixture
def init_building(monkeypatch):
    import autoforge.webui.api.init as init_api

    monkeypatch.setattr(init_api, "_state", {"status": "initializing", "preview_image": None, "error": None})


def test_optimization_refused_while_the_preview_builds(client, init_building):
    from autoforge.webui.services.optimization_service import get_optimization_service

    _active_filament(client)
    r = client.post("/api/optimize/start", json={"input_image": "a.png", "iterations": 10})
    assert r.status_code == 409, r.text
    assert "preview" in r.json()["detail"].lower()
    assert get_optimization_service().get_any_active_job() is None


def test_pruning_refused_while_the_preview_builds(client, init_building):
    from autoforge.webui.services.optimization_service import get_optimization_service

    svc = get_optimization_service()
    done = svc.create_job({"iterations": 10, "input_image": "a.png"})
    svc.update_status(done.job_id, "completed")
    svc.set_pipeline_result(done.job_id, {"optimizer": object()})
    r = client.post("/api/pruning/start", json={"job_id": done.job_id})
    assert r.status_code == 409, r.text
    assert svc.get_pipeline_result(done.job_id) is not None


# ---------------------------------------------------------------------------
# The webui's project file: strict JSON, millimetre sizes
# ---------------------------------------------------------------------------


def test_webui_export_writes_a_strict_json_project_in_millimetres(tmp_path):
    """End to end through export_results with a filament that has no brand
    (the webui's materials.csv then has an empty cell)."""
    import json

    import cv2
    import numpy as np

    from autoforge.webui.helpers.pipeline_runner import build_init_preview, export_results

    img = np.zeros((40, 60, 3), np.uint8)
    img[:, :30] = 255
    cv2.imwrite(str(tmp_path / "in.png"), img)
    filaments = [
        {"color": "#000000", "td": 0.6, "name": "Black", "brand": "", "short_name": "Black", "uuid": ""},
        {"color": "#ffffff", "td": 5.0, "name": "White", "brand": "", "short_name": "White", "uuid": ""},
    ]
    state = build_init_preview(
        str(tmp_path / "in.png"), filaments, str(tmp_path / "out"),
        {"stl_output_size": 10, "max_layers": 8, "num_init_rounds": 1,
         "num_init_threads": 1, "random_seed": 3, "device": "cpu"},
        device=torch.device("cpu"),
    )
    outputs = export_results(state)

    def reject(constant):
        raise ValueError(constant)

    with open(outputs["project_file"]) as f:
        data = json.load(f, parse_constant=reject)
    assert max(data["width_in_mm"], data["height_in_mm"]) == pytest.approx(10.0)
    for entry in data["filament_set"]:
        assert isinstance(entry["Brand"], str) and isinstance(entry["uuid"], str)
