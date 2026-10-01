"""Webui regression tests for the 2026-10-01 bug report (W-*, F-* IDs)."""

import argparse
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
        self._prune_runs = 0
        self.preview_callback = None
        self.prune_step_callback = None

    def get_discretized_solution(self, best=True):
        return torch.zeros(self.max_layers, dtype=torch.long), torch.ones(6, 8)

    def get_best_discretized_image(self):
        return torch.full((6, 8, 3), 200.0)


def _completed_job_with_result(svc):
    done = svc.create_job({"iterations": 10, "input_image": "a.png"})
    svc.update_status(done.job_id, "completed")
    svc.set_pipeline_result(done.job_id, {
        "optimizer": _Optimizer(),
        "alpha": None,
        "args": argparse.Namespace(spike_removal=False, layer_height=0.04, background_height=0.4, stl_output_size=50.0),
    })
    return done


def test_w5_cancel_while_saving_returns_the_result_to_its_job(client, monkeypatch):
    """W-5: a cancel that lands during "Saving the result" of a single-pass
    prune leaves the job cancelled and the original job owning its result."""
    import autoforge.Helper.PruningHelper as pruning_helper
    import autoforge.webui.helpers.pipeline_runner as pipeline_runner
    from autoforge.webui.services.optimization_service import get_optimization_service

    svc = get_optimization_service()
    done = _completed_job_with_result(svc)
    monkeypatch.setattr(pruning_helper, "_compute_loss_for_heightmap", lambda *_a, **_k: 1.0)
    prune_ids = []

    def export(result, cancel_event=None, **_kwargs):
        # Every phase finished; the user cancels while files are written.
        svc.cancel(prune_ids[0])
        return {"pruning_completed": True}

    monkeypatch.setattr(pipeline_runner, "export_results", export)
    real_create = svc.create_job
    monkeypatch.setattr(svc, "create_job", lambda s, job_id=None: (prune_ids.append(job_id), real_create(s, job_id=job_id))[1])
    r = client.post("/api/pruning/start", json={"job_id": done.job_id})
    assert r.status_code == 200, r.text
    _wait_for_status(client, prune_ids[0], "cancelled")
    deadline = time.time() + 5
    while time.time() < deadline and svc.get_pipeline_result(done.job_id) is None:
        time.sleep(0.05)
    assert svc.get_pipeline_result(done.job_id) is not None
    assert svc.get_pipeline_result(prune_ids[0]) is None


def test_w6_cross_origin_requests_refused_and_localhost_default(client):
    """W-6: no wildcard CORS; a browser request from another site is refused
    (HTTP and websocket), same-origin and Origin-less requests work, and
    the server binds to this machine by default."""
    from starlette.websockets import WebSocketDisconnect

    from autoforge.webui.config import WebUIConfig

    assert WebUIConfig().host == "127.0.0.1"

    evil = {"Origin": "https://evil.example"}
    r = client.post("/api/optimize/cancel/whatever", headers=evil)
    assert r.status_code == 403
    assert "access-control-allow-origin" not in r.headers
    assert client.options("/api/filaments", headers={**evil, "Access-Control-Request-Method": "DELETE"}).status_code == 403
    assert client.get("/api/system/health").status_code == 200
    assert client.get("/api/system/health", headers={"Origin": "http://testserver"}).status_code == 200
    assert client.get("/api/system/health", headers={"Origin": "null"}).status_code == 403

    with pytest.raises(WebSocketDisconnect):
        with client.websocket_connect("/ws/preview", headers=evil) as ws:
            ws.receive_text()
    with client.websocket_connect("/ws/preview", headers={"Origin": "http://testserver"}) as ws:
        ws.send_text("connected")


def test_w7_telemetry_docs_say_paths_can_be_sent():
    """W-7: the telemetry docs no longer promise that no file paths are sent
    - exception messages and stack traces carry them."""
    import pathlib

    from autoforge.webui.helpers import telemetry

    doc = telemetry.__doc__
    assert "file paths are sent" not in doc
    assert "file paths" in doc and "traceback" in doc
    readme = (pathlib.Path(__file__).resolve().parents[2] / "README.md").read_text()
    assert "can contain file paths" in readme


def test_w10_upload_cap_is_enforced_while_reading(client, monkeypatch):
    """W-10: an oversized upload is rejected without reading it whole."""
    import cv2
    import numpy as np

    import autoforge.webui.api.images as images
    from starlette.datastructures import UploadFile

    monkeypatch.setattr(images, "_MAX_UPLOAD_BYTES", 3000)
    monkeypatch.setattr(images, "_CHUNK", 1000)
    reads = []
    real_read = UploadFile.read

    async def counting_read(self, size=-1):
        reads.append(size)
        return await real_read(self, size)

    monkeypatch.setattr(UploadFile, "read", counting_read)
    big = np.random.default_rng(0).integers(0, 255, (100, 100, 3), dtype=np.uint8)
    ok, png = cv2.imencode(".png", big)
    r = client.post("/api/images/upload", files={"file": ("big.png", png.tobytes(), "image/png")})
    assert r.status_code == 413
    assert -1 not in reads  # never "read everything"
    assert len(reads) <= 4

    small = cv2.imencode(".png", np.zeros((4, 4, 3), np.uint8))[1].tobytes()
    r = client.post("/api/images/upload", files={"file": ("s.png", small, "image/png")})
    assert r.status_code == 200, r.text


def test_w11_only_hashed_assets_are_cached_forever(tmp_path):
    """W-11: immutable caching only for 200 responses under /assets/."""
    import asyncio

    from autoforge.webui.server import SPAStaticFiles

    (tmp_path / "assets").mkdir()
    (tmp_path / "assets" / "index-abc123.js").write_text("x")
    (tmp_path / "favicon.ico").write_text("i")
    (tmp_path / "index.html").write_text("<html></html>")
    static = SPAStaticFiles(directory=str(tmp_path), html=True)

    def get(path):
        scope = {"type": "http", "path": path, "method": "GET", "headers": []}
        try:
            return asyncio.run(static.get_response(path.lstrip("/"), scope))
        except Exception as exc:  # StaticFiles raises HTTPException for 404
            return exc

    assert "immutable" in get("/assets/index-abc123.js").headers["cache-control"]
    assert get("/favicon.ico").headers["cache-control"] == "no-cache"
    assert get("/").headers["cache-control"] == "no-cache"
    missing = get("/assets/missing.js")
    assert "immutable" not in str(getattr(missing, "headers", {}) or {})


def test_w12_history_eviction_deletes_the_output_folder(tmp_path, monkeypatch):
    """W-12: evicted history jobs take their checkpoints/<job_id> folder
    with them, except while a live result still uses it."""
    import os

    import autoforge.webui.services.optimization_service as osvc

    monkeypatch.setattr(osvc, "MAX_HISTORY_JOBS", 2)
    svc = osvc.OptimizationService(checkpoints_dir=str(tmp_path))
    ids = []
    for i in range(4):
        js = svc.create_job({"iterations": 1}, job_id=f"job{i}")
        js.started_at = f"2026-10-01T00:00:0{i}"
        os.makedirs(tmp_path / js.job_id)
        (tmp_path / js.job_id / "final_model.stl").write_text("x")
        ids.append(js.job_id)
    svc._pipeline_results["job1"] = {"optimizer": None}
    for jid in ids:
        svc.update_status(jid, "completed")
    assert not os.path.exists(tmp_path / "job0")
    assert os.path.exists(tmp_path / "job1")  # still live
    assert os.path.exists(tmp_path / "job2") and os.path.exists(tmp_path / "job3")


def test_w12_render_maps_drop_dead_targets(client, monkeypatch):
    """W-12: per-job render bookkeeping doesn't grow for jobs that can no
    longer be rendered."""
    import autoforge.webui.api.preview as preview
    from autoforge.webui.services.optimization_service import get_optimization_service

    for k in range(5):
        preview._latest_seq[f"dead{k}"] = 1
        preview._render_locks[f"dead{k}"]
    svc = get_optimization_service()
    done = _completed_job_with_result(svc)
    svc.get_pipeline_result(done.job_id)["background"] = torch.zeros(3)
    r = client.post("/api/preview/render-with-sliders", json={"job_id": done.job_id, "sliders": []})
    assert r.status_code == 200, r.text
    assert not any(k.startswith("dead") for k in preview._latest_seq)
    assert not any(k.startswith("dead") for k in preview._render_locks)


def _active_filament(client, uuid="br-fil-1"):
    fil = {"brand": "BR", "name": f"br-{uuid}", "color": "#224466", "td": 3.0, "filament_type": "PLA", "uuid": uuid}
    client.post("/api/filaments", json=fil)
    client.post("/api/filaments/active", json=fil)


def _upload_png(client, name="picture.png"):
    import cv2
    import numpy as np

    ok, png = cv2.imencode(".png", np.full((20, 40, 3), 128, np.uint8))
    r = client.post("/api/images/upload", files={"file": (name, png.tobytes(), "image/png")})
    assert r.status_code == 200, r.text
    return r.json()["filename"]


def test_w13_failed_export_releases_the_result(client, monkeypatch):
    """W-13: a run whose export fails doesn't keep its optimizer resident."""
    import autoforge.webui.api.optimization as opt_api
    import autoforge.webui.helpers.pipeline_runner as pipeline_runner
    from autoforge.webui.services.optimization_service import get_optimization_service

    _active_filament(client)
    image = _upload_png(client)
    monkeypatch.setattr(opt_api, "_run_pipeline", lambda **kw: {"cancelled": False, "device": None, "optimizer": _Optimizer()})

    def failing_export(result, **kw):
        raise RuntimeError("disk full")

    monkeypatch.setattr(pipeline_runner, "export_results", failing_export)
    r = client.post("/api/optimize/start", json={"input_image": image, "iterations": 10})
    assert r.status_code == 200, r.text
    job_id = r.json()["job_id"]
    status = _wait_for_status(client, job_id, "failed")
    assert "disk full" in status["error"]
    assert job_id not in get_optimization_service().pipeline_result_job_ids()


def test_w15_update_check_versions_and_error_cache(monkeypatch):
    """W-15: pre-release tags aren't offered as updates, versions compare
    numerically, and a failed check is retried after a few minutes."""
    import asyncio

    import autoforge.webui.api.system as system

    assert system._is_newer("2.2.0", "2.1.1")
    assert system._is_newer("2.10.0", "2.9.9")
    assert not system._is_newer("2.2.0-beta.1", "2.2.0")
    assert not system._is_newer("2.3.0rc1", "2.2.0")
    assert system._is_newer("2.2.0", "2.2.0rc1")
    assert not system._is_newer("2.1.1", "2.1.1")
    assert not system._is_newer("2.1", "2.1.0")

    calls = []
    clock = [1000.0]
    monkeypatch.setattr(system.time, "time", lambda: clock[0])
    monkeypatch.setattr(system, "_check_latest_release", lambda: (calls.append(1), {"error": "offline"})[1])
    monkeypatch.setattr(system, "_update_cache", {"checked_at": 0.0, "data": None})
    asyncio.run(system.check_for_update())
    clock[0] += 60
    asyncio.run(system.check_for_update())
    assert len(calls) == 1  # still cached
    clock[0] += system._UPDATE_ERROR_TTL
    asyncio.run(system.check_for_update())
    assert len(calls) == 2  # retried well before the hour
