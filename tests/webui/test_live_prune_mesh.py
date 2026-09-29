"""The 3D preview follows each pruning step (api/pruning.py's
_on_prune_step), not only the finished prune."""

import argparse
import os
import time

import pytest
import torch
from fastapi.testclient import TestClient

from autoforge.webui.server import create_app


@pytest.fixture
def client():
    return TestClient(create_app())


def _wait_for_status(client, job_id, status, timeout=10.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        r = client.get(f"/api/optimize/status/{job_id}")
        if r.is_success and r.json().get("status") == status:
            return r.json()
        time.sleep(0.05)
    raise AssertionError(f"job {job_id} never reached {status!r}")


class _Optimizer:
    def __init__(self):
        self.best_params = {"global_logits": torch.zeros(4, 3)}
        self.max_layers = 4
        self.heights = torch.ones(6, 8)
        self._prune_runs = 0
        self.preview_callback = None
        self.prune_step_callback = None

    def get_discretized_solution(self, best=True):
        return torch.zeros(self.max_layers, dtype=torch.long), self.heights.clone()

    def get_best_discretized_image(self):
        return torch.full((6, 8, 3), 200.0)


def test_each_pruning_step_publishes_a_live_mesh(client, monkeypatch):
    from autoforge.webui.config import config
    from autoforge.webui.services.optimization_service import get_optimization_service
    import autoforge.Helper.PruningHelper as pruning_helper
    import autoforge.webui.api.ws as ws
    import autoforge.webui.helpers.pipeline_runner as pipeline_runner
    import autoforge.webui.helpers.sliders as sliders

    svc = get_optimization_service()
    optimizer = _Optimizer()
    done = svc.create_job({"iterations": 10, "input_image": "a.png"})
    svc.update_status(done.job_id, "completed")
    svc.set_pipeline_result(done.job_id, {
        "optimizer": optimizer,
        "alpha": None,
        "args": argparse.Namespace(
            spike_removal=False, layer_height=0.04, background_height=0.4, stl_output_size=50.0,
        ),
    })
    monkeypatch.setattr(pruning_helper, "_compute_loss_for_heightmap", lambda *_a, **_k: 1.0)
    monkeypatch.setattr(
        sliders, "derive_sliders_from_result",
        lambda result: {"sliders": [{"layer": 4}], "min_layer": 0, "max_layer": 4},
    )
    broadcasts = []
    monkeypatch.setattr(
        ws, "broadcast_preview",
        lambda image, job_id, sliders=None, live_mesh=None, **kw: broadcasts.append((job_id, live_mesh)),
    )

    seen_live = []

    def export(result, cancel_event=None, pause_event=None, apply_spike_removal=True, **_kwargs):
        opt = result["optimizer"]
        opt.prune_step_callback(opt, "Reducing colors")
        opt.prune_step_callback(opt, "Reducing swaps")  # nothing changed: no new mesh
        opt.heights = opt.heights * 2
        opt.prune_step_callback(opt, "Reducing layers")
        seen_live.extend(live for _job, live in broadcasts if live)
        return {"pruning_completed": True}

    monkeypatch.setattr(pipeline_runner, "export_results", export)
    r = client.post("/api/pruning/start", json={"job_id": done.job_id})
    assert r.status_code == 200, r.text
    prune_id = r.json()["job_id"]
    _wait_for_status(client, prune_id, "completed")

    # One live mesh per step that changed the solution, sent under the job
    # the frontend is still showing.
    assert seen_live == [{"prune_job_id": prune_id, "url": f"/api/outputs/live-ply/{prune_id}"}] * 2
    assert all(job == done.job_id for job, live in broadcasts if live)

    mesh = client.get(f"/api/outputs/live-ply/{prune_id}")
    assert mesh.status_code == 200
    assert mesh.content.startswith(b"ply")
    # Never written over the result pruning started from.
    assert not os.path.exists(os.path.join(config.checkpoints_path, done.job_id, "live_model_colored.ply"))
    assert optimizer.prune_step_callback is None


def test_live_mesh_404_before_any_step(client):
    assert client.get("/api/outputs/live-ply/prune-none").status_code == 404
