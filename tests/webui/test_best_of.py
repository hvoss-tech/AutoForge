"""Best of N runs in the webui (OptimizationSettings.best_of): the
optimization runs N times, every finished run is saved, and the user picks
the one to keep from a gallery (the lowest-loss run is the one still live)."""

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


def test_a_run_that_fails_to_save_ends_the_series_with_the_finished_ones():
    pipeline, states = _fake_pipeline([2.0, 1.0, 0.5], [])

    def save(i, state, loss):
        if i == 1:
            raise RuntimeError("out of memory while pruning")

    best = run_best_of("in.png", [], "out", {}, best_of=3, pipeline=pipeline, candidate_callback=save)
    assert best["run"] == 0
    assert best["best_of"]["losses"] == [2.0]
    assert states[1] == {}


def test_a_run_changed_while_saving_is_compared_by_its_new_loss():
    pipeline, _ = _fake_pipeline([2.0, 1.0], [])
    best = run_best_of(
        "in.png", [], "out", {}, best_of=2, pipeline=pipeline,
        candidate_callback=lambda i, state, loss: [0.5, 1.5][i],
    )
    assert best["run"] == 0
    assert best["best_of"]["losses"] == [0.5, 1.5]


def test_a_failed_first_run_fails_the_series():
    pipeline, _ = _fake_pipeline([1.0], [], fail_at=0)
    with pytest.raises(RuntimeError):
        run_best_of("in.png", [], "out", {}, best_of=3, pipeline=pipeline)


def test_choose_keeps_the_live_run_when_it_is_picked():
    restored = []
    pipeline, states = _fake_pipeline([3.0, 1.0, 2.0], [])
    chosen = []
    best = run_best_of(
        "in.png", [], "out", {}, best_of=3, pipeline=pipeline,
        choose=lambda losses, best_index: chosen.append((losses, best_index)) or best_index,
        restore=lambda i: restored.append(i),
    )
    assert chosen == [([3.0, 1.0, 2.0], 1)]
    assert best["run"] == 1 and best["best_of"]["best_run"] == 1
    assert restored == []


def test_picking_another_run_restores_it_and_releases_the_live_one():
    pipeline, states = _fake_pipeline([3.0, 1.0, 2.0], [])
    seen = []
    best = run_best_of(
        "in.png", [], "out", {}, best_of=3, pipeline=pipeline,
        candidate_callback=lambda i, state, loss: seen.append((i, state["run"], loss)),
        choose=lambda losses, best_index: 2,
        restore=lambda i: {"cancelled": False, "optimizer": _Opt(2.0), "restored": i},
    )
    # Every finished run was handed over before it was released.
    assert seen == [(0, 0, 3.0), (1, 1, 1.0), (2, 2, 2.0)]
    assert best["restored"] == 2
    assert best["best_of"] == {"runs": 3, "best_run": 2, "losses": [3.0, 1.0, 2.0]}
    # The run that was live (the lowest loss) gave its tensors back.
    assert states[1] == {}


def test_not_choosing_cancels_and_keeps_nothing():
    pipeline, states = _fake_pipeline([3.0, 1.0], [])
    result = run_best_of(
        "in.png", [], "out", {}, best_of=2, pipeline=pipeline,
        choose=lambda losses, best_index: None, restore=lambda i: pytest.fail("nothing to restore"),
    )
    assert result == {"cancelled": True}
    assert states[1] == {}


def test_a_single_finished_run_needs_no_choice():
    pipeline, _ = _fake_pipeline([2.0, 1.0], [], fail_at=1)
    best = run_best_of(
        "in.png", [], "out", {}, best_of=2, pipeline=pipeline,
        choose=lambda losses, best_index: pytest.fail("one run, nothing to choose"),
    )
    assert best["run"] == 0


@pytest.fixture(params=[{}])
def fake_run(monkeypatch, request):
    """The webui's best-of job with a fake pipeline: saving, rebuilding and
    exporting a run are recorded instead of done."""
    from autoforge.webui.api import optimization as opt_api
    from autoforge.webui.config import config
    from autoforge.webui.helpers import pipeline_runner, result_store

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

    def fake_save(job_dir, result):
        os.makedirs(job_dir, exist_ok=True)
        np.savez(os.path.join(job_dir, "solution.npz"), loss=result["optimizer"].best_discrete_loss)
        return True

    def fake_load(job_dir):
        path = os.path.join(job_dir, "solution.npz")
        return {"loss": float(np.load(path)["loss"])} if os.path.exists(path) else None

    rebuilt = []

    def fake_rebuild(solution, image_path, output_dir, device=None):
        rebuilt.append(solution["loss"])
        return {"cancelled": False, "optimizer": _Opt(solution["loss"]), "material_TDs_np": np.array([1.0, 2.0])}

    exported = []
    monkeypatch.setattr(opt_api, "_run_pipeline", fake_pipeline)
    monkeypatch.setattr(pipeline_runner, "to_output_resolution", lambda result: None)
    # The pruning pass every run of a series gets: lowers the loss by 0.5.
    pruned = []

    def fake_prune(result, seed_search_count=200, **_k):
        pruned.append(seed_search_count)
        result["optimizer"].best_discrete_loss -= 0.5
        return True

    monkeypatch.setattr(pipeline_runner, "initial_prune", fake_prune)
    monkeypatch.setattr(pipeline_runner, "solution_loss", lambda result: result["optimizer"].best_discrete_loss)
    monkeypatch.setattr(pipeline_runner, "export_results", lambda result, **_k: exported.append(result["optimizer"].best_discrete_loss))
    monkeypatch.setattr(result_store, "save_solution", fake_save)
    monkeypatch.setattr(result_store, "load_solution", fake_load)
    monkeypatch.setattr(result_store, "rebuild_pipeline_result", fake_rebuild)

    os.makedirs(config.uploads_path, exist_ok=True)
    with open(os.path.join(config.uploads_path, "bestof.png"), "wb") as f:
        f.write(b"x")
    client.put("/api/filaments/active", json=[{"uuid": "a", "name": "A", "td": 1.0}, {"uuid": "b", "name": "B", "td": 2.0}])
    job_id = client.post(
        "/api/optimize/start",
        json={"input_image": "bestof.png", "iterations": 10, "best_of": 3, "initial_prune_seed_search_count": 77,
              **request.param},
    ).json()["job_id"]
    job_dir = os.path.join(config.checkpoints_path, job_id)
    yield {"client": client, "job_id": job_id, "job_dir": job_dir, "updates": updates,
           "exported": exported, "rebuilt": rebuilt, "pruned": pruned}
    client.post(f"/api/optimize/cancel/{job_id}")


def _wait_for(client, job_id, check, timeout=10.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        status = client.get(f"/api/optimize/status/{job_id}").json()
        if check(status):
            return status
        time.sleep(0.05)
    raise AssertionError(f"timed out, last status: {status}")


def test_webui_series_waits_for_the_user_to_pick_a_run(fake_run):
    client, job_id = fake_run["client"], fake_run["job_id"]
    status = _wait_for(client, job_id, lambda s: s.get("best_of_choosing"))
    assert status["status"] == "running"
    assert status["phase"] == "Choose a result"
    # Every run was pruned (as a single run is after it finished) before it
    # is shown, and is compared by its loss after pruning.
    assert fake_run["pruned"] == [77, 77, 77]
    assert status["best_of_losses"] == [2.5, 0.5, 1.5]
    # The lowest loss is suggested.
    assert status["best_of_run"] == 1
    # Nothing is exported until a run was picked.
    time.sleep(0.3)
    assert fake_run["exported"] == []
    assert client.get(f"/api/optimize/status/{job_id}").json()["status"] == "running"

    # Every run's image is there to look at.
    for run in range(3):
        r = client.get(f"/api/optimize/best-of/{job_id}/image/{run}")
        assert r.status_code == 200 and r.headers["content-type"] == "image/png"
    assert client.get(f"/api/optimize/best-of/{job_id}/image/3").status_code == 404

    # Nothing left to pause.
    client.post(f"/api/optimize/pause/{job_id}")
    assert client.get(f"/api/optimize/status/{job_id}").json()["status"] == "running"

    phases = [u["phase"] for u in fake_run["updates"] if u.get("phase")]
    assert "Run 1/3 · Optimizing" in phases and "Run 3/3 · Optimizing" in phases
    # One bar over the whole series.
    progress = [u["progress"] for u in fake_run["updates"] if "progress" in u]
    assert progress == sorted(progress)
    first_run = [u["progress"] for u in fake_run["updates"] if "progress" in u and (u.get("phase") or "").startswith("Run 1/")]
    assert first_run and max(first_run) <= 95.0 / 3


def test_webui_picking_the_live_run_exports_it_as_is(fake_run):
    client, job_id = fake_run["client"], fake_run["job_id"]
    _wait_for(client, job_id, lambda s: s.get("best_of_choosing"))
    assert client.post(f"/api/optimize/best-of/{job_id}/pick", json={"run": 1}).status_code == 200
    status = _wait_for(client, job_id, lambda s: s["status"] == "completed")
    assert status["best_of_run"] == 1
    assert status["loss"] == 0.5
    assert not status["best_of_choosing"]
    # Pruned already: the frontend must not prune it again.
    assert status["best_of_pruned"] is True
    assert fake_run["exported"] == [0.5]
    assert fake_run["rebuilt"] == []
    # The saved runs are gone once one was picked.
    assert not os.path.exists(os.path.join(fake_run["job_dir"], "best_of_runs"))


def test_webui_picking_another_run_exports_that_run(fake_run):
    client, job_id = fake_run["client"], fake_run["job_id"]
    _wait_for(client, job_id, lambda s: s.get("best_of_choosing"))
    assert client.post(f"/api/optimize/best-of/{job_id}/pick", json={"run": 5}).status_code == 400
    assert client.post(f"/api/optimize/best-of/{job_id}/pick", json={"run": 2}).status_code == 200
    # Only one choice per series.
    assert client.post(f"/api/optimize/best-of/{job_id}/pick", json={"run": 0}).status_code == 409
    status = _wait_for(client, job_id, lambda s: s["status"] == "completed")
    assert status["best_of_run"] == 2
    # The job reports the picked run's loss, not the last run's.
    assert status["loss"] == 1.5
    assert fake_run["rebuilt"] == [1.5]
    assert fake_run["exported"] == [1.5]


@pytest.mark.parametrize("fake_run", [{"auto_initial_prune": False}], indirect=True)
def test_webui_series_without_automatic_pruning_leaves_runs_unpruned(fake_run):
    client, job_id = fake_run["client"], fake_run["job_id"]
    status = _wait_for(client, job_id, lambda s: s.get("best_of_choosing"))
    assert status["best_of_losses"] == [3.0, 1.0, 2.0]
    assert fake_run["pruned"] == []
    client.post(f"/api/optimize/best-of/{job_id}/pick", json={"run": 1})
    status = _wait_for(client, job_id, lambda s: s["status"] == "completed")
    assert status["best_of_pruned"] is False


def test_webui_cancelling_the_choice_keeps_nothing(fake_run):
    client, job_id = fake_run["client"], fake_run["job_id"]
    _wait_for(client, job_id, lambda s: s.get("best_of_choosing"))
    client.post(f"/api/optimize/cancel/{job_id}")
    _wait_for(client, job_id, lambda s: s["status"] == "cancelled")
    deadline = time.time() + 5
    while time.time() < deadline and os.path.exists(os.path.join(fake_run["job_dir"], "best_of_runs")):
        time.sleep(0.05)
    assert not os.path.exists(os.path.join(fake_run["job_dir"], "best_of_runs"))
    assert fake_run["exported"] == []
    assert client.post(f"/api/optimize/best-of/{job_id}/pick", json={"run": 0}).status_code == 409


def test_webui_a_run_that_was_not_saved_cannot_be_picked(fake_run):
    client, job_id = fake_run["client"], fake_run["job_id"]
    _wait_for(client, job_id, lambda s: s.get("best_of_choosing"))
    os.remove(os.path.join(fake_run["job_dir"], "best_of_runs", "run_0", "solution.npz"))
    r = client.post(f"/api/optimize/best-of/{job_id}/pick", json={"run": 0})
    assert r.status_code == 409
    assert "Run 1" in r.json()["detail"]
    # Still waiting; another run can be picked.
    assert client.post(f"/api/optimize/best-of/{job_id}/pick", json={"run": 1}).status_code == 200
    _wait_for(client, job_id, lambda s: s["status"] == "completed")


def test_real_series_finishes_with_the_picked_run():
    """A real (tiny) series: the run picked from the gallery that wasn't the
    live one comes back from its saved (pruned) solution, and the finished
    job holds exactly that run's layers and heights."""
    import io

    from PIL import Image

    from autoforge.webui.config import config
    from autoforge.webui.helpers.result_store import load_solution

    with TestClient(create_app()) as client:
        for name, color, td in [("Black", "#111111", 0.6), ("White", "#F0F0F0", 2.5), ("Red", "#D02020", 1.5), ("Blue", "#2040C0", 1.2)]:
            r = client.post("/api/filaments", json={"brand": "T", "name": name, "color": color, "td": td})
            assert client.post("/api/filaments/active", json=r.json()).status_code == 200
        y, x = np.mgrid[0:48, 0:48]
        img = np.zeros((48, 48, 3), np.uint8)
        img[..., 0] = (x * 5).astype(np.uint8)
        img[..., 2] = (y * 5).astype(np.uint8)
        img[16:32, 16:32] = (240, 240, 240)
        buf = io.BytesIO()
        Image.fromarray(img).save(buf, format="PNG")
        image = client.post("/api/images/upload", files={"file": ("pic.png", buf.getvalue(), "image/png")}).json()["filename"]

        job_id = client.post("/api/optimize/start", json={
            "input_image": image, "iterations": 60, "max_layers": 10, "stl_output_size": 16,
            "num_init_rounds": 1, "random_seed": 7, "best_of": 2,
        }).json()["job_id"]
        status = _wait_for(client, job_id, lambda s: s.get("best_of_choosing") or s["status"] in ("failed", "cancelled"), timeout=300)
        assert status["status"] == "running", status
        other = 1 - status["best_of_run"]
        picked = load_solution(os.path.join(config.checkpoints_path, job_id, "best_of_runs", f"run_{other}"))
        assert picked is not None
        assert client.get(f"/api/optimize/best-of/{job_id}/image/{other}").status_code == 200

        assert client.post(f"/api/optimize/best-of/{job_id}/pick", json={"run": other}).status_code == 200
        status = _wait_for(client, job_id, lambda s: s["status"] in ("completed", "failed", "cancelled"), timeout=300)
        assert status["status"] == "completed", status
        assert status["best_of_run"] == other
        assert status["loss"] == status["best_of_losses"][other]
        # Each run had its pruning pass before it was shown, and the export
        # wrote the run exactly as it was shown (no second prune).
        assert status["best_of_pruned"] is True
        final = load_solution(os.path.join(config.checkpoints_path, job_id))
        assert np.array_equal(final["stack"], picked["stack"])
        assert np.array_equal(final["heights"], picked["heights"])
        assert os.path.exists(os.path.join(config.checkpoints_path, job_id, "final_model.stl"))
