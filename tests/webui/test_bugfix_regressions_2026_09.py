"""Regressions for webui bugs found in the 2026-09-17 audit."""

import threading
import time

import pytest
from fastapi.testclient import TestClient

from autoforge.webui import create_app
from autoforge.webui.services.optimization_service import get_optimization_service


@pytest.fixture(autouse=True)
def _fresh_storage(tmp_path, monkeypatch):
    """The session-wide temp dirs from conftest persist between tests, and
    snapshots are reloaded from disk — give every test here its own."""
    from autoforge.webui.config import config

    monkeypatch.setattr(config, "checkpoints_dir", str(tmp_path / "checkpoints"))
    monkeypatch.setattr(config, "uploads_dir", str(tmp_path / "uploads"))
    monkeypatch.setattr(config, "library_dir", str(tmp_path / "filament_library"))


@pytest.fixture
def client():
    return TestClient(create_app())


# --- Job control on a job that already finished -----------------------------


@pytest.mark.parametrize("action", ["pause", "resume", "cancel"])
def test_control_on_completed_job_does_not_change_status(client, action):
    """Clicking Cancel/Pause just as a job completes used to overwrite the
    completed job with "cancelled" (persisted to history, disabling
    prune/export) or "paused"/"running" (with no thread left to finish it)."""
    svc = get_optimization_service()
    svc.create_job({"iterations": 1}, job_id="done-job")
    svc.update_status("done-job", "completed")

    resp = client.post(f"/api/optimize/{action}/done-job")

    assert resp.status_code == 200
    assert resp.json() == {"status": "completed"}
    assert svc.get_job("done-job").status == "completed"
    assert svc.get_result("done-job").status == "completed"


def test_control_endpoints_report_real_status_for_live_job(client):
    svc = get_optimization_service()
    svc.create_job({"iterations": 1}, job_id="live-job")
    svc.update_status("live-job", "running")

    assert client.post("/api/optimize/pause/live-job").json() == {"status": "paused"}
    assert client.post("/api/optimize/resume/live-job").json() == {"status": "running"}
    assert client.post("/api/optimize/cancel/live-job").json() == {"status": "cancelled"}
    assert client.post("/api/optimize/cancel/missing").status_code == 404


@pytest.mark.parametrize("late_status", ["failed", "completed", "running"])
def test_cancelled_job_is_not_overwritten_by_worker_thread(late_status):
    """The worker thread can still report after a cancel (an exception, or
    the export finishing) — that used to turn "cancelled" into
    "failed"/"completed"."""
    svc = get_optimization_service()
    svc.create_job({"iterations": 1}, job_id="c-job")
    svc.update_status("c-job", "running")
    svc.cancel("c-job")

    svc.update_status("c-job", late_status, error="boom")

    job = svc.get_job("c-job")
    assert job.status == "cancelled"
    assert job.error is None


# --- Snapshot history / restore ----------------------------------------------


def test_snapshot_history_keeps_nested_fields_snake_case(client):
    """After a page reload the undo stack is hydrated from /history. Nested
    fields came back camelCased (settings.layerHeight, filamentUuid, ...), so
    an undo wiped the settings/slider/filament fields the frontend reads."""
    snap = {
        "timestamp": 1726570000.5,
        "label": "edit",
        "activeFilaments": [{"uuid": "f1", "name": "Red", "filament_type": "PLA"}],
        "colorSliders": [{"layer": 3, "depth_mm": 0.12, "filament_uuid": "f1", "enabled": True}],
        "settings": {"input_image": "img.png", "layer_height": 0.08},
        "inputImage": "/uploads/img.png",
        "currentJobId": "job-1",
    }
    assert client.post("/api/state/snapshot", json=snap).status_code == 200

    [entry] = client.get("/api/state/history").json()

    assert entry["timestamp"] == 1726570000.5
    assert entry["inputImage"] == "/uploads/img.png"
    assert entry["currentJobId"] == "job-1"
    assert entry["activeFilaments"][0]["filament_type"] == "PLA"
    assert entry["colorSliders"][0]["filament_uuid"] == "f1"
    assert entry["colorSliders"][0]["depth_mm"] == 0.12
    assert entry["settings"]["input_image"] == "img.png"
    assert entry["settings"]["layer_height"] == 0.08
    assert "layerHeight" not in entry["settings"]


def test_restore_snapshot_with_whole_second_timestamp(client):
    """Date.now()/1000 on a whole second is sent as a JSON integer; it used
    to be string-compared against the stored float ("…00" vs "…00.0")."""
    client.post("/api/state/snapshot", json={"timestamp": 1726570000.0, "label": "whole"})

    resp = client.post("/api/state/restore", json={"timestamp": 1726570000})

    assert resp.status_code == 200
    assert resp.json()["label"] == "whole"
    assert resp.json()["colorSliders"] == []


def test_restore_unknown_timestamp_404(client):
    assert client.post("/api/state/restore", json={"timestamp": 1.25}).status_code == 404


# --- Active filament sync -----------------------------------------------------


def test_put_active_replaces_whole_list(client):
    """Undo/redo and project loading need to set the backend's active list
    (what /api/optimize/start reads) to exactly the restored one."""
    client.post("/api/filaments/active", json={"uuid": "a", "name": "A"})
    client.post("/api/filaments/active", json={"uuid": "b", "name": "B"})

    resp = client.put("/api/filaments/active", json=[{"uuid": "b", "name": "B"}, {"uuid": "c", "name": "C"}])

    assert resp.status_code == 200
    assert sorted(f["uuid"] for f in client.get("/api/filaments/active").json()) == ["b", "c"]


# --- Preview broadcast thread safety -------------------------------------------


def test_broadcast_survives_concurrent_connection_changes(monkeypatch):
    """broadcast_preview runs on the optimizer thread while the event loop
    adds/removes preview sockets; iterating the live set raised
    "Set changed size during iteration"."""
    import asyncio
    from autoforge.webui.api import ws as ws_module

    loop = asyncio.new_event_loop()
    monkeypatch.setattr(ws_module, "_main_loop", loop)
    monkeypatch.setattr(ws_module, "_preview_connections", set())

    class FakeSocket:
        async def send_text(self, msg):
            pass

    def fake_run_coroutine_threadsafe(coro, _loop):
        coro.close()
        # Simulate another client connecting mid-broadcast.
        ws_module._preview_connections.add(FakeSocket())

    monkeypatch.setattr(asyncio, "run_coroutine_threadsafe", fake_run_coroutine_threadsafe)
    ws_module._preview_connections.add(FakeSocket())
    try:
        ws_module.broadcast_preview("img", job_id="j")
    finally:
        loop.close()


# --- Undo/redo history persistence (round 2) ------------------------------------


def _snap(client, ts, label, query=""):
    resp = client.post(f"/api/state/snapshot{query}", json={"timestamp": ts, "label": label})
    assert resp.status_code == 200


def test_new_edit_after_undo_discards_redo_branch_on_server(client):
    """After undoing to step 2 of 4 and making a new edit, steps 3-4 must be
    gone server-side too — they used to come back interleaved by time with
    the new branch when the page hydrated its history on reload."""
    for ts, label in [(100.0, "a"), (200.0, "b"), (300.0, "c"), (400.0, "d")]:
        _snap(client, ts, label)

    _snap(client, 500.0, "e", query="?discard_after=200.0")

    labels = [s["label"] for s in client.get("/api/state/history").json()]
    assert sorted(labels) == ["a", "b", "e"]


def test_snapshot_history_is_capped(client, tmp_path):
    from autoforge.webui.services.project_service import MAX_SNAPSHOTS, get_project_service

    for i in range(MAX_SNAPSHOTS + 5):
        _snap(client, float(i + 1), f"s{i}")

    history = client.get("/api/state/history").json()
    assert len(history) == MAX_SNAPSHOTS
    assert min(s["timestamp"] for s in history) == 6.0
    snapshot_dir = get_project_service()._snapshot_dir
    import os
    assert len([f for f in os.listdir(snapshot_dir) if f.endswith(".json")]) == MAX_SNAPSHOTS


def test_restore_does_not_cancel_running_job(client):
    """Undo used to cancel whatever job the target snapshot referenced."""
    svc = get_optimization_service()
    svc.create_job({"iterations": 1}, job_id="busy")
    svc.update_status("busy", "running")
    client.post("/api/state/snapshot", json={"timestamp": 42.0, "label": "x", "currentJobId": "busy"})

    assert client.post("/api/state/restore", json={"timestamp": 42.0}).status_code == 200
    assert svc.get_job("busy").status == "running"


def test_snapshot_keeps_slider_layer_range(client):
    client.post("/api/state/snapshot", json={"timestamp": 7.0, "label": "r", "sliderLayerRange": {"min": 3, "max": 41}})
    [entry] = client.get("/api/state/history").json()
    assert entry["sliderLayerRange"] == {"min": 3, "max": 41}


# --- Slider-edit renders -----------------------------------------------------------


@pytest.fixture
def fake_render(monkeypatch):
    from autoforge.webui.api import preview as preview_api

    calls = []

    def _fake(pipeline_result, sliders, lookup, output_dir, **names):
        calls.append({"output_dir": output_dir, **names})
        return {"image_b64": None}

    monkeypatch.setattr(preview_api, "render_with_sliders", _fake)
    return calls


def _completed_job(job_id, with_pipeline=True):
    svc = get_optimization_service()
    svc.create_job({"iterations": 1}, job_id=job_id)
    svc.update_status(job_id, "completed")
    if with_pipeline:
        svc.set_pipeline_result(job_id, {"optimizer": object()})


def test_render_for_unrenderable_job_does_not_fall_back_to_another_job(client, fake_render):
    """A render for a job that isn't completed used to silently render into
    the latest *other* completed job's folder, overwriting its files."""
    _completed_job("other")
    svc = get_optimization_service()
    svc.create_job({"iterations": 1}, job_id="mine")

    resp = client.post("/api/preview/render-with-sliders", json={"sliders": [], "job_id": "mine"})

    assert resp.status_code == 400
    assert fake_render == []


def test_render_after_restart_explains_instead_of_silently_failing(client, fake_render):
    _completed_job("old", with_pipeline=False)
    resp = client.post("/api/preview/render-with-sliders", json={"sliders": [], "job_id": "old"})
    assert resp.status_code == 409
    assert "restarted" in resp.json()["detail"]


def test_slider_edit_render_does_not_overwrite_optimizer_outputs(client, fake_render):
    """Edits go to edited_model_* so the optimizer's final_model_* (what the
    export zip, STL and swap instructions correspond to) stay intact."""
    import os
    from autoforge.webui.config import config

    _completed_job("res")
    resp = client.post("/api/preview/render-with-sliders", json={"sliders": [], "job_id": "res"})

    assert resp.status_code == 200
    [call] = fake_render
    assert call["output_dir"] == os.path.join(config.checkpoints_path, "res")
    assert call["png_name"] == "edited_model.png"
    assert call["ply_name"] == "edited_model_colored.ply"


def test_colored_ply_prefers_slider_edited_mesh(client):
    import os
    from autoforge.webui.config import config
    from autoforge.webui.api.outputs import discard_slider_edits

    job_dir = os.path.join(config.checkpoints_path, "plyjob")
    os.makedirs(job_dir, exist_ok=True)
    with open(os.path.join(job_dir, "final_model_colored.ply"), "w") as f:
        f.write("original")
    assert client.get("/api/outputs/colored-ply/plyjob").text == "original"

    with open(os.path.join(job_dir, "edited_model_colored.ply"), "w") as f:
        f.write("edited")
    assert client.get("/api/outputs/colored-ply/plyjob").text == "edited"

    discard_slider_edits("plyjob")
    assert client.get("/api/outputs/colored-ply/plyjob").text == "original"


def test_slider_td_overrides_library_td():
    """The TD box above each slider column had no effect on the render."""
    from autoforge.webui.helpers.slider_render import _resolve_slider_materials

    lookup = {"f": {"color": "#ff0000", "td": 4.0}}
    _, tds = _resolve_slider_materials(
        [{"filament_uuid": "f", "td": 1.5}, {"filament_uuid": "f", "td": 0}], lookup
    )
    assert list(tds) == [1.5, 4.0]


def test_init_status_reports_layer_range(client, monkeypatch):
    from autoforge.webui.api import init as init_api

    monkeypatch.setattr(init_api, "_state", {"status": "ready", "preview_image": None, "error": None, "range": {"min_layer": 2, "max_layer": 30}})
    assert client.get("/api/init/status").json() == {
        "status": "ready",
        "error": None,
        # No base recorded in this hand-built state; the client falls back to
        # its own settings (see test_init_status_reports_resolved_base).
        "base": None,
        "min_layer": 2,
        "max_layer": 30,
    }



# --- Settings validation --------------------------------------------------------


@pytest.mark.parametrize(
    "bad",
    [
        {"processing_reduction_factor": 0},  # was: job failed with "division by zero"
        {"max_layers": 0},  # was: job failed with a scikit-learn KMeans error
        {"stl_output_size": 0},  # was: job failed with an OpenCV assertion
        {"layer_height": 0},
        {"iterations": 0},
        {"init_heightmap_method": "no-such-method"},  # was: silently ran as kmeans
    ],
)
def test_start_rejects_invalid_settings_before_creating_a_job(client, bad):
    client.put("/api/filaments/active", json=[{"uuid": "f", "name": "F", "td": 1.0}])
    resp = client.post("/api/optimize/start", json={"input_image": "x.png", **bad})
    assert resp.status_code == 422
    assert client.get("/api/optimize/history").json() == []


def test_invalid_persisted_settings_do_not_break_loading(client):
    import json
    import os
    from autoforge.webui.config import config

    os.makedirs(config.checkpoints_path, exist_ok=True)
    with open(os.path.join(config.checkpoints_path, "project_state.json"), "w") as f:
        json.dump({"settings": {"max_layers": 0}}, f)
    resp = client.get("/api/project/state")
    assert resp.status_code == 200
    assert resp.json()["settings"]["max_layers"] == 75


# --- Usability rework ---------------------------------------------------------------


def test_current_preview_prefers_the_slider_edited_image(client):
    import os
    from autoforge.webui.config import config

    job_dir = os.path.join(config.checkpoints_path, "imgjob")
    os.makedirs(job_dir, exist_ok=True)
    with open(os.path.join(job_dir, "final_model.png"), "wb") as f:
        f.write(b"original")
    assert client.get("/api/outputs/current-preview/imgjob").content == b"original"

    with open(os.path.join(job_dir, "edited_model.png"), "wb") as f:
        f.write(b"edited")
    assert client.get("/api/outputs/current-preview/imgjob").content == b"edited"
    # The download endpoint keeps serving the optimizer's own file.
    assert client.get("/api/outputs/preview/imgjob").content == b"original"
    assert client.get("/api/outputs/current-preview/../etc").status_code == 404


def test_optimization_reports_phases(client, monkeypatch):
    """The top bar shows "Preparing" / "Optimizing" / "Exporting results"
    instead of a bare 0.0% that looked stuck."""
    import time
    from autoforge.webui.api import optimization as opt_api

    seen = []
    svc = get_optimization_service()
    original_update = svc.update_status

    def recording_update(job_id, status, **kwargs):
        if "phase" in kwargs:
            seen.append(kwargs["phase"])
        return original_update(job_id, status, **kwargs)

    monkeypatch.setattr(svc, "update_status", recording_update)

    def fake_pipeline(**kwargs):
        class Opt:
            best_discrete_loss = 1.0
        kwargs["progress_callback"](Opt(), 5)
        return {"cancelled": False}

    monkeypatch.setattr(opt_api, "_run_pipeline", fake_pipeline)
    monkeypatch.setattr("autoforge.webui.helpers.pipeline_runner.export_results", lambda result: None)

    import os
    from autoforge.webui.config import config
    os.makedirs(config.uploads_path, exist_ok=True)
    with open(os.path.join(config.uploads_path, "p.png"), "wb") as f:
        f.write(b"x")
    client.put("/api/filaments/active", json=[{"uuid": "f", "name": "F", "td": 1.0}])
    job_id = client.post("/api/optimize/start", json={"input_image": "p.png", "iterations": 10}).json()["job_id"]

    deadline = time.time() + 10
    while time.time() < deadline and client.get(f"/api/optimize/status/{job_id}").json()["status"] != "completed":
        time.sleep(0.05)
    status = client.get(f"/api/optimize/status/{job_id}").json()
    assert status["status"] == "completed"
    assert status["phase"] is None
    assert seen[:3] == ["Preparing", "Optimizing", "Exporting results"]


def test_slider_render_replaces_files_atomically(tmp_path, monkeypatch):
    """Readers must never see a half-written edited mesh/image: the files
    are written next to the target and renamed into place."""
    import numpy as np
    import torch
    from autoforge.webui.helpers import slider_render

    replaced = []
    real_replace = slider_render.os.replace
    monkeypatch.setattr(slider_render.os, "replace", lambda src, dst: (replaced.append((src, dst)), real_replace(src, dst)))

    class Opt:
        max_layers = 4

        def get_discretized_solution(self, best=True):
            return torch.zeros(4, dtype=torch.long), torch.full((6, 8), 3.0)

    class Args:
        layer_height = 0.04
        background_height = 0.24
        stl_output_size = 20

    result = slider_render.render_with_sliders(
        {"optimizer": Opt(), "args": Args(), "alpha": None, "background": torch.zeros(3)},
        [{"enabled": True, "layer": 4, "td": 2.0, "filament_uuid": "f"}],
        {"f": {"color": "#ff0000", "td": 2.0}},
        str(tmp_path),
        png_name="edited_model.png",
        ply_name="edited_model_colored.ply",
    )

    assert result is not None
    assert {dst.split("/")[-1] for _, dst in replaced} == {"edited_model.png", "edited_model_colored.ply"}
    assert sorted(p.name for p in tmp_path.iterdir()) == ["edited_model.png", "edited_model_colored.ply"]
    assert np.fromfile(tmp_path / "edited_model.png", dtype=np.uint8)[:4].tobytes() == b"\x89PNG"
