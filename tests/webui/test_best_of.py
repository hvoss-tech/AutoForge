"""Best of N runs in the webui (OptimizationSettings.best_of), like the CLI's
--best_of: the optimization runs N times and the lowest-loss run is kept."""

import os
import threading
import time

import numpy as np
import pytest
import torch
from fastapi.testclient import TestClient

from autoforge.webui.helpers.pipeline_runner import run_best_of
from autoforge.webui.server import create_app
from autoforge.webui.services.optimization_service import get_optimization_service


class _Opt:
    h = 0.04
    base_material = None

    def __init__(self, loss):
        self.best_discrete_loss = loss
        self.stack = torch.tensor([0, 1], dtype=torch.long)
        self.material_colors = torch.zeros((2, 3))
        self.background = torch.zeros(3)

    def get_discretized_solution(self, best=True):
        return self.stack, torch.full((4, 4), 2.0)

    def get_best_discretized_image(self):
        return np.zeros((4, 4, 3), np.uint8)


def _fake_pipeline(losses, seen_seeds, fail_at=None, cancel_at=None, cancel_event=None):
    states = []

    def pipeline(**kwargs):
        i = len(seen_seeds)
        seen_seeds.append(kwargs["settings"].get("random_seed"))
        if fail_at == i:
            raise RuntimeError("out of memory")
        if cancel_at == i:
            cancel_event.set()
            state = {"cancelled": True, "optimizer": _Opt(0.0)}
        else:
            state = {"cancelled": False, "optimizer": _Opt(losses[i]), "run": i}
        states.append(state)
        return state

    return pipeline, states


def test_keeps_the_lowest_loss_run_and_releases_the_others():
    seeds = []
    pipeline, states = _fake_pipeline([3.0, 1.0, 2.0], seeds)
    best = run_best_of("in.png", [], "out", {"random_seed": 42}, best_of=3, pipeline=pipeline)
    assert best["run"] == 1
    assert best["best_of"] == {"runs": 3, "best_run": 1, "losses": [3.0, 1.0, 2.0]}
    # The losing runs gave their tensors back.
    assert states[0] == {} and states[2] == {}
    # A fixed seed counts up from the first run's.
    assert seeds == [42, 43, 44]


def test_seed_zero_draws_a_fresh_seed_every_run():
    seeds = []
    pipeline, _ = _fake_pipeline([1.0, 2.0], seeds)
    run_best_of("in.png", [], "out", {"random_seed": 0}, best_of=2, pipeline=pipeline)
    assert seeds == [0, 0]


def test_reports_each_run_with_the_losses_finished_before_it():
    calls = []
    pipeline, _ = _fake_pipeline([3.0, 1.0, 2.0], [])
    run_best_of(
        "in.png", [], "out", {}, best_of=3, pipeline=pipeline,
        run_callback=lambda i, n, losses: calls.append((i, n, losses)),
    )
    assert calls == [(0, 3, []), (1, 3, [3.0]), (2, 3, [3.0, 1.0])]


def test_cancelling_stops_the_series_and_keeps_nothing():
    cancel = threading.Event()
    seeds = []
    pipeline, states = _fake_pipeline([1.0, 2.0, 3.0], seeds, cancel_at=1, cancel_event=cancel)
    result = run_best_of("in.png", [], "out", {}, best_of=3, pipeline=pipeline, cancel_event=cancel)
    assert result["cancelled"] is True
    assert len(seeds) == 2
    assert states[0] == {} and states[1] == {}


def test_a_later_failed_run_keeps_the_best_finished_one():
    pipeline, _ = _fake_pipeline([2.0, 1.0, 0.5], [], fail_at=2)
    best = run_best_of("in.png", [], "out", {}, best_of=3, pipeline=pipeline)
    assert best["run"] == 1
    assert best["best_of"]["losses"] == [2.0, 1.0]


def test_a_failed_first_run_fails_the_series():
    pipeline, _ = _fake_pipeline([1.0], [], fail_at=0)
    with pytest.raises(RuntimeError):
        run_best_of("in.png", [], "out", {}, best_of=3, pipeline=pipeline)


def test_webui_run_with_best_of_exports_the_best_run(monkeypatch):
    from autoforge.webui.api import optimization as opt_api
    from autoforge.webui.config import config

    client = TestClient(create_app())
    monkeypatch.setattr(opt_api, "broadcast_preview", lambda *a, **k: None)
    svc = get_optimization_service()
    original_update = svc.update_status
    updates = []

    def recording_update(job_id, status, **kwargs):
        updates.append(kwargs)
        return original_update(job_id, status, **kwargs)

    monkeypatch.setattr(svc, "update_status", recording_update)

    losses = iter([3.0, 1.0, 2.0])

    def fake_pipeline(**kwargs):
        loss = next(losses)
        kwargs["phase_callback"]("Estimating heights", 1.0)
        kwargs["progress_callback"](_Opt(loss), 10)
        return {"cancelled": False, "optimizer": _Opt(loss), "material_TDs_np": np.array([1.0, 2.0])}

    exported = []
    monkeypatch.setattr(opt_api, "_run_pipeline", fake_pipeline)
    monkeypatch.setattr(
        "autoforge.webui.helpers.pipeline_runner.export_results",
        lambda result, **_kwargs: exported.append(result["optimizer"].best_discrete_loss),
    )

    os.makedirs(config.uploads_path, exist_ok=True)
    with open(os.path.join(config.uploads_path, "bestof.png"), "wb") as f:
        f.write(b"x")
    client.put("/api/filaments/active", json=[{"uuid": "a", "name": "A", "td": 1.0}, {"uuid": "b", "name": "B", "td": 2.0}])
    job_id = client.post(
        "/api/optimize/start", json={"input_image": "bestof.png", "iterations": 10, "best_of": 3}
    ).json()["job_id"]

    deadline = time.time() + 10
    while time.time() < deadline and client.get(f"/api/optimize/status/{job_id}").json()["status"] != "completed":
        time.sleep(0.05)
    status = client.get(f"/api/optimize/status/{job_id}").json()
    assert status["status"] == "completed"
    assert status["best_of_run"] == 1
    assert status["best_of_losses"] == [3.0, 1.0, 2.0]
    assert exported == [1.0]

    phases = [u["phase"] for u in updates if u.get("phase")]
    assert "Run 1/3 · Optimizing" in phases and "Run 3/3 · Optimizing" in phases
    # One bar over the whole series.
    progress = [u["progress"] for u in updates if "progress" in u]
    assert progress == sorted(progress)
    first_run = [u["progress"] for u in updates if "progress" in u and (u.get("phase") or "").startswith("Run 1/")]
    assert first_run and max(first_run) <= 95.0 / 3
