"""Regressions for the 2026-09-23 WebUI bug review.

Each test here was written against the buggy behavior first: it fails on the
pre-fix code and passes after the fix. The review also confirmed that the
existing suite (all green) did not cover any of these paths.
"""

import json
import os
import threading
import time

import pytest
from fastapi.testclient import TestClient

from autoforge.webui.server import create_app


@pytest.fixture
def client():
    return TestClient(create_app())


def _active_filament(client, uuid="rev-fil-1", color="#123456"):
    """One active filament, so /api/optimize/start passes its guards."""
    client.post("/api/filaments", json={
        "brand": "ReviewTest", "name": f"rev-{uuid}", "color": color,
        "td": 5.0, "filament_type": "PLA", "uuid": uuid,
    })
    client.post("/api/filaments/active", json={
        "brand": "ReviewTest", "name": f"rev-{uuid}", "color": color,
        "td": 5.0, "filament_type": "PLA", "uuid": uuid,
    })
    return uuid


def _wait_for_status(client, job_id, status, timeout=10.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        r = client.get(f"/api/optimize/status/{job_id}")
        if r.is_success and r.json().get("status") == status:
            return r.json()
        time.sleep(0.05)
    raise AssertionError(f"job {job_id} never reached {status!r}")


# ---------------------------------------------------------------------------
# 1. release_pipeline_result must release a *real* optimizer's CUDA graph
# ---------------------------------------------------------------------------


def test_release_pipeline_result_releases_a_real_optimizer_graph():
    """Regression: the helper probed for `_release_graph`/`release_graph`,
    but FilamentOptimizer's actual method is `release_cuda_graph`
    (Modules/Optimizer.py:599), so on a real result the probe was always a
    silent no-op and the graph's private memory pool stayed pinned through
    export and every prune pass. The existing test used a fake that
    *happened* to have `_release_graph`, which is why this went unnoticed."""
    from autoforge.webui.helpers.gpu_memory import release_pipeline_result

    released = []

    class RealLikeOptimizer:
        device = None

        def release_cuda_graph(self):
            released.append(True)

    release_pipeline_result({"optimizer": RealLikeOptimizer()})
    assert released == [True]


# ---------------------------------------------------------------------------
# 2. One GPU job at a time: pruning must not overlap an optimization (or
#    another pruning)
# ---------------------------------------------------------------------------


def test_pruning_rejected_while_an_optimization_is_running(client):
    from autoforge.webui.services.optimization_service import get_optimization_service

    _active_filament(client)
    svc = get_optimization_service()
    done = svc.create_job({"iterations": 10, "input_image": "a.png"})
    svc.update_status(done.job_id, "completed")
    svc.set_pipeline_result(done.job_id, {"optimizer": object()})
    running = svc.create_job({"iterations": 10, "input_image": "b.png"})
    svc.update_status(running.job_id, "running")

    r = client.post("/api/pruning/start", json={"job_id": done.job_id})
    assert r.status_code == 409, r.text
    # The completed job's result must be intact and prunable afterwards.
    assert svc.get_pipeline_result(done.job_id) is not None


def test_optimization_rejected_while_a_prune_is_running(client):
    _active_filament(client)
    from autoforge.webui.services.optimization_service import get_optimization_service

    svc = get_optimization_service()
    prune = svc.create_job({"iterations": 1}, job_id="prune-review1")
    svc.update_status(prune.job_id, "running")

    r = client.post("/api/optimize/start", json={"input_image": "a.png", "iterations": 10})
    assert r.status_code == 409, r.text


def test_second_prune_rejected_while_one_is_running(client):
    _active_filament(client)
    from autoforge.webui.services.optimization_service import get_optimization_service

    svc = get_optimization_service()
    done = svc.create_job({"iterations": 10, "input_image": "a.png"})
    svc.update_status(done.job_id, "completed")
    svc.set_pipeline_result(done.job_id, {"optimizer": object()})
    prune = svc.create_job({"iterations": 1}, job_id="prune-review2")
    svc.update_status(prune.job_id, "running")

    r = client.post("/api/pruning/start", json={"job_id": done.job_id})
    assert r.status_code == 409, r.text


# ---------------------------------------------------------------------------
# 3. Cancelling a new run must not destroy the previous run's result
# ---------------------------------------------------------------------------


def test_job_cancelled_before_its_thread_runs_keeps_previous_result(client, monkeypatch):
    """Regression: start_optimization's worker called
    `clear_all_pipeline_results()` before checking the cancel event, so a
    cancelled run (even one cancelled before it did any work) destroyed the
    previous job's pipeline result — pruning and slider re-rendering of it
    then failed with 409 'No optimization pipeline result found'."""
    import autoforge.webui.api.optimization as opt_api
    from autoforge.webui.config import config
    from autoforge.webui.services.optimization_service import get_optimization_service

    svc = get_optimization_service()
    _active_filament(client)
    previous = svc.create_job({"iterations": 10, "input_image": "a.png"})
    svc.update_status(previous.job_id, "completed")
    result_a = {"optimizer": object(), "device": None}
    svc.set_pipeline_result(previous.job_id, result_a)

    pipeline_calls = []

    def fake_run_pipeline(**kwargs):
        pipeline_calls.append(True)
        return {"cancelled": True, "device": None, "optimizer": object()}

    monkeypatch.setattr(opt_api, "_run_pipeline", fake_run_pipeline)

    # Start the job's worker thread manually, only after the cancel has
    # landed — the endpoint normally fires the thread immediately. Patch the
    # `threading` reference inside the optimization module only (a global
    # threading.Thread patch would also capture the TestClient's own portal
    # thread and deadlock it).
    captured = {}
    RealThread = threading.Thread

    class DeferredThread(RealThread):
        def start(self):
            # First start() call (from the endpoint) only captures the
            # thread; the test's second call actually runs it.
            if getattr(self, "_deferred", False):
                RealThread.start(self)
                return
            self._deferred = True
            captured["thread"] = self

    class _ThreadProxy:
        Thread = DeferredThread

        def __getattr__(self, name):
            return getattr(threading, name)

    monkeypatch.setattr(opt_api, "threading", _ThreadProxy())

    r = client.post("/api/optimize/start", json={"input_image": "b.png", "iterations": 10})
    assert r.status_code == 200, r.text
    job_id = r.json()["job_id"]
    client.post(f"/api/optimize/cancel/{job_id}")
    captured["thread"].start()
    time.sleep(0.5)  # let the (cancelled) worker run to completion

    assert svc.get_job(job_id).status == "cancelled"
    assert pipeline_calls == [], "a cancelled run must not start the pipeline"
    assert svc.get_pipeline_result(previous.job_id) is result_a
    assert result_a != {}, "previous run's result was released by the cancelled run"


def test_successful_run_still_releases_previous_results_before_it_runs(client, monkeypatch):
    """Guard for the OOM fix this review must not regress: a new run releases
    every retained result *before* its own tensors are built, so two
    optimizers never coexist on the GPU. (Cancel-before-start is the only
    case where the previous result is kept.)"""
    import autoforge.webui.api.optimization as opt_api
    from autoforge.webui.services.optimization_service import get_optimization_service

    _active_filament(client)
    svc = get_optimization_service()
    previous = svc.create_job({"iterations": 10, "input_image": "a.png"})
    svc.update_status(previous.job_id, "completed")
    result_a = {"optimizer": object(), "device": None}
    svc.set_pipeline_result(previous.job_id, result_a)

    from autoforge.webui.config import config
    os.makedirs(config.uploads_dir, exist_ok=True)
    with open(os.path.join(config.uploads_dir, "b.png"), "wb") as f:
        f.write(b"fake")

    def fake_run_pipeline(**kwargs):
        # By the time the pipeline starts, the previous result must already
        # be gone (released) or this run OOMs on small GPUs.
        assert svc.pipeline_result_job_ids() == []
        return {"cancelled": True, "device": None, "optimizer": object()}

    monkeypatch.setattr(opt_api, "_run_pipeline", fake_run_pipeline)

    r = client.post("/api/optimize/start", json={"input_image": "b.png", "iterations": 10})
    assert r.status_code == 200, r.text
    job_id = r.json()["job_id"]
    _wait_for_status(client, job_id, "cancelled")


# ---------------------------------------------------------------------------
# 4. Auto-preview init: reset must wait out an in-flight build, and a
#    superseded build must not overwrite the state of the new one
# ---------------------------------------------------------------------------


def _init_fake_result(tag):
    return {
        "result": {"optimizer": object(), "tag": tag},
        "preview_b64": f"png-{tag}",
        "range": {"min_layer": 0, "max_layer": 30},
        "base": {"color": "#111111", "height_mm": 0.24, "layers": 6, "filament_uuid": "", "auto": True},
    }


def test_init_reset_waits_for_in_flight_build_and_discards_it(client, monkeypatch):
    """Regression: /api/init/reset flipped the status to `idle` even while a
    worker thread was still building, so a subsequent /api/init/run started a
    second, concurrent build; when the older build finished it overwrote the
    state/mesh with data for the image that was just replaced."""
    import asyncio

    import httpx

    import autoforge.webui.api.init as init_api
    from autoforge.webui.config import config

    gates = []

    def fake_run_init_sync(input_image_path, filament_dicts, settings_dict):
        gate = threading.Event()
        gates.append(gate)
        gate.wait(30)
        return _init_fake_result(os.path.basename(input_image_path))

    monkeypatch.setattr(init_api, "_run_init_sync", fake_run_init_sync)
    _active_filament(client)

    img_a = "rev_a.png"
    img_b = "rev_b.png"
    os.makedirs(config.uploads_dir, exist_ok=True)
    for name in (img_a, img_b):
        with open(os.path.join(config.uploads_dir, name), "wb") as f:
            f.write(b"fake")

    async def scenario():
        transport = httpx.ASGITransport(app=client.app)
        async with httpx.AsyncClient(transport=transport, base_url="http://t") as http:
            req_a = asyncio.ensure_future(
                http.post("/api/init/run", json={"input_image": img_a, "num_init_rounds": 1})
            )
            while not gates:
                await asyncio.sleep(0.01)
            # Let the in-flight build finish; reset must not return before it
            # does (no second, concurrent build may start on the GPU).
            reset_task = asyncio.ensure_future(http.post("/api/init/reset"))
            await asyncio.sleep(0.2)
            assert not reset_task.done(), "reset returned while the build was still in flight"
            gates[0].set()
            reset = await reset_task
            assert reset.status_code == 200
            status = (await http.get("/api/init/status")).json()
            assert status["status"] == "idle"
            # The superseded build's response must not report success, and
            # its result must not be the stored one.
            resp_a = await req_a
            assert resp_a.status_code == 409, (
                f"a superseded init must not report success, got {resp_a.status_code}: {resp_a.text[:200]}"
            )
            # Now the new image's build may start and becomes the state.
            req_b_task = asyncio.ensure_future(
                http.post("/api/init/run", json={"input_image": img_b, "num_init_rounds": 1})
            )
            while len(gates) < 2:
                await asyncio.sleep(0.01)
            gates[1].set()
            req_b = await req_b_task
            assert req_b.status_code == 200, req_b.text
            status_b = (await http.get("/api/init/status")).json()
            assert status_b["status"] == "ready"
            assert status_b["min_layer"] == 0
            assert init_api.get_init_pipeline_result()["tag"] == img_b

    asyncio.run(asyncio.wait_for(scenario(), timeout=60))


# ---------------------------------------------------------------------------
# 5. Job records carry the image they ran on (so a reload can tell which
#    result belongs to the image currently on screen)
# ---------------------------------------------------------------------------


def test_job_status_records_its_input_image(client):
    _active_filament(client)
    from autoforge.webui.services.optimization_service import get_optimization_service

    svc = get_optimization_service()
    job = svc.create_job({"iterations": 10, "input_image": "some_img.png"})
    payload = client.get("/api/optimize/status/" + job.job_id).json()
    assert payload.get("input_image") == "some_img.png"

    latest = client.get("/api/optimize/latest").json()
    assert latest.get("input_image") == "some_img.png"


# ---------------------------------------------------------------------------
# 6. FlatForge jobs must be exportable from the webui
# ---------------------------------------------------------------------------


def test_flatforge_job_zip_contains_the_per_material_stls(client):
    """Regression: flatforge runs write `{material}_{hex}.stl` files (no
    final_model.stl), and the export endpoints only knew the fixed
    traditional file list — so the "Everything (.zip)" bundle contained no
    STL at all and the user could not download the print files."""
    from autoforge.webui.config import config
    from autoforge.webui.services.optimization_service import get_optimization_service

    svc = get_optimization_service()
    job = svc.create_job({"iterations": 10, "input_image": "a.png", "flatforge": True})
    svc.update_status(job.job_id, "completed")
    job_dir = os.path.join(config.checkpoints_dir, job.job_id)
    os.makedirs(job_dir, exist_ok=True)
    for name in ("Red_#ff0000.stl", "Blue_#0000ff.stl"):
        with open(os.path.join(job_dir, name), "wb") as f:
            f.write(b"solid test\nendsolid test\n")
    with open(os.path.join(job_dir, "final_model.png"), "wb") as f:
        f.write(b"png")
    with open(os.path.join(job_dir, "final_model_colored.ply"), "wb") as f:
        f.write(b"ply")

    r = client.get(f"/api/outputs/export/{job.job_id}")
    assert r.status_code == 200, r.text
    import zipfile
    import io
    names = zipfile.ZipFile(io.BytesIO(r.content)).namelist()
    assert "Red_#ff0000.stl" in names
    assert "Blue_#0000ff.stl" in names
    assert "final_model_colored.ply" in names

    # The single-STL endpoint has nothing to serve for a flatforge job.
    assert client.get(f"/api/outputs/stl/{job.job_id}").status_code == 404


def test_traditional_job_zip_keeps_working(client):
    from autoforge.webui.config import config
    from autoforge.webui.services.optimization_service import get_optimization_service

    svc = get_optimization_service()
    job = svc.create_job({"iterations": 10, "input_image": "a.png"})
    svc.update_status(job.job_id, "completed")
    job_dir = os.path.join(config.checkpoints_dir, job.job_id)
    os.makedirs(job_dir, exist_ok=True)
    for name in ("final_model.stl", "final_model_colored.ply", "final_model.png",
                 "swap_instructions.txt", "final_loss.txt", "project_file.hfp",
                 "materials.csv", "edited_model.ply"):
        with open(os.path.join(job_dir, name), "wb") as f:
            f.write(b"x")

    r = client.get(f"/api/outputs/export/{job.job_id}")
    assert r.status_code == 200, r.text
    import zipfile
    import io
    names = set(zipfile.ZipFile(io.BytesIO(r.content)).namelist())
    assert "final_model.stl" in names
    assert "project_file.hfp" in names
    # Internal/intermediate files are not part of the download bundle.
    assert "materials.csv" not in names
    assert "edited_model.ply" not in names


# ---------------------------------------------------------------------------
# 7. The base64 preview image must not ride on status payloads / history
# ---------------------------------------------------------------------------


def test_preview_image_not_exposed_in_status_or_history(client, monkeypatch):
    """Regression: JobStatus.preview_image (a full base64 PNG) was sent on
    every 0.5 s WS poll, every REST status response, and persisted into
    history.json for every terminal transition — unbounded bandwidth, RAM and
    disk growth."""
    from autoforge.webui.services.optimization_service import get_optimization_service

    svc = get_optimization_service()
    job = svc.create_job({"iterations": 10})
    svc.update_status(job.job_id, "running", preview_image="AAAA-BASE64-PNG-AAA=")

    for url in (f"/api/optimize/status/{job.job_id}", "/api/optimize/latest", "/api/optimize/history"):
        r = client.get(url)
        assert r.status_code == 200
        blob = r.text
        assert "AAAA-BASE64-PNG-AAA=" not in blob, f"preview image leaked into {url}"

    svc.update_status(job.job_id, "completed")
    history = json.loads(open(svc._history_file()).read())
    assert "preview_image" not in json.dumps(history), "preview image persisted into history.json"


def test_preview_image_ws_not_exposed(client):
    # A terminal job: the server's WS handler sends the status once and
    # exits, so the TestClient's portal can shut down cleanly (a
    # non-terminal job would keep the handler's send loop alive after the
    # test client disconnects, which starlette's test harness cannot
    # cancel).
    from autoforge.webui.services.optimization_service import get_optimization_service

    svc = get_optimization_service()
    job = svc.create_job({"iterations": 10})
    svc.update_status(job.job_id, "running", preview_image="BBBB-BASE64-PNG-BBB=")
    svc.update_status(job.job_id, "cancelled")

    with client.websocket_connect(f"/ws/optimize/{job.job_id}") as ws:
        msg = ws.receive_json()
        assert msg["status"] == "cancelled"
        assert "BBBB-BASE64-PNG-BBB=" not in json.dumps(msg)


# ---------------------------------------------------------------------------
# 8. Corrupted/old-format history files must not take the whole API down
# ---------------------------------------------------------------------------


def test_history_with_bad_records_does_not_crash_the_service():
    """Regression: _load_history only caught (JSONDecodeError, IOError,
    KeyError); a pydantic ValidationError (e.g. `progress: "abc"` from an
    older version) or an AttributeError (top-level JSON array) propagated
    out of OptimizationService.__init__, making *every* API route 500 until
    the file was deleted by hand."""
    from autoforge.webui.config import config
    from autoforge.webui.services import optimization_service

    optimization_service.reset_service()
    os.makedirs(config.checkpoints_dir, exist_ok=True)
    history_path = os.path.join(config.checkpoints_dir, "history.json")

    with open(history_path, "w") as f:
        json.dump({
            "job-bad-type": {"status": {"job_id": "job-bad-type", "status": "completed", "progress": "not-a-number"}},
            "job-bad-shape": {"status": "just-a-string"},
            "job-good": {"status": {"job_id": "job-good", "status": "completed"}},
        }, f)

    svc = optimization_service.get_optimization_service()  # must not raise
    assert svc.get_job("job-good") is not None
    # ...and the app's endpoints keep answering.
    client = TestClient(create_app())
    r = client.get("/api/optimize/history")
    assert r.status_code == 200
    assert any(j["job_id"] == "job-good" for j in r.json())


def test_history_file_that_is_not_an_object_is_ignored():
    from autoforge.webui.config import config
    from autoforge.webui.services import optimization_service

    optimization_service.reset_service()
    os.makedirs(config.checkpoints_dir, exist_ok=True)
    with open(os.path.join(config.checkpoints_dir, "history.json"), "w") as f:
        f.write('[{"job_id": "x"}]')  # top-level array: .items() used to raise AttributeError

    svc = optimization_service.get_optimization_service()  # must not raise
    assert svc.get_history() == []


def test_project_state_file_that_is_not_an_object_is_ignored(client):
    from autoforge.webui.config import config

    with open(os.path.join(config.checkpoints_dir, "project_state.json"), "w") as f:
        f.write("[1, 2, 3]")  # ProjectState(**data) used to raise TypeError -> 500
    r = client.get("/api/project/state")
    assert r.status_code == 200
    assert r.json().get("color_sliders") == []


# ---------------------------------------------------------------------------
# 9. /api/system/update-check must not block the event loop
# ---------------------------------------------------------------------------


def test_update_check_does_not_block_the_event_loop(client, monkeypatch):
    """Regression: the handler ran `urllib.request.urlopen(timeout=5)`
    synchronously inside an async def, stalling every other request and
    websocket (including live job progress) for up to ~5 s on the first page
    load."""
    import autoforge.webui.api.system as system_api

    system_api._update_cache["data"] = None  # force a real (faked) check

    def fake_urlopen(req, timeout=None):
        time.sleep(1.5)  # simulate a slow GitHub API
        raise OSError("simulated network failure")

    monkeypatch.setattr(system_api.urllib.request, "urlopen", fake_urlopen)

    import asyncio
    import httpx

    async def scenario():
        transport = httpx.ASGITransport(app=client.app)
        async with httpx.AsyncClient(transport=transport, base_url="http://t") as http:
            update_task = asyncio.ensure_future(http.get("/api/system/update-check"))
            await asyncio.sleep(0.2)  # update-check is inside its fake sleep now
            t0 = time.monotonic()
            health = await http.get("/api/system/health")
            elapsed = time.monotonic() - t0
            assert health.status_code == 200
            # If the loop were blocked, the health check would wait out the
            # whole 1.5 s fake sleep.
            assert elapsed < 1.0, f"event loop blocked: health took {elapsed:.2f}s"
            upd = await update_task
            assert upd.status_code == 200
            assert upd.json()["error"] is not None  # failure surfaced, not raised

    asyncio.run(scenario())
    system_api._update_cache["data"] = None


# ---------------------------------------------------------------------------
# 10. /api/filaments/brands honours the filament_type filter
# ---------------------------------------------------------------------------


def test_brands_endpoint_filters_by_filament_type(client):
    for brand, ftype in (("BrandPLA", "PLA"), ("BrandABS", "ABS")):
        client.post("/api/filaments", json={
            "brand": brand, "name": "f", "color": "#ffffff", "td": 5.0,
            "filament_type": ftype, "uuid": f"rev-{brand}",
        })
    pla = client.get("/api/filaments/brands", params={"filament_type": "PLA"}).json()
    assert "BrandPLA" in pla
    assert "BrandABS" not in pla
    all_brands = client.get("/api/filaments/brands").json()
    assert "BrandPLA" in all_brands and "BrandABS" in all_brands


# ---------------------------------------------------------------------------
# 11. Deleting/updating a library filament must propagate to the active list
# ---------------------------------------------------------------------------


def test_deleting_a_filament_removes_it_from_the_active_list(client):
    uuid = "rev-active-del"
    client.post("/api/filaments", json={
        "brand": "ReviewTest", "name": "del", "color": "#ffffff", "td": 5.0,
        "filament_type": "PLA", "uuid": uuid,
    })
    client.post("/api/filaments/active", json={
        "brand": "ReviewTest", "name": "del", "color": "#ffffff", "td": 5.0,
        "filament_type": "PLA", "uuid": uuid,
    })
    assert len(client.get("/api/filaments/active").json()) == 1

    r = client.delete(f"/api/filaments/{uuid}")
    assert r.status_code == 200
    assert client.get("/api/filaments/active").json() == []


def test_updating_a_filament_updates_the_active_copy(client):
    uuid = "rev-active-upd"
    client.post("/api/filaments", json={
        "brand": "ReviewTest", "name": "upd", "color": "#ffffff", "td": 5.0,
        "filament_type": "PLA", "uuid": uuid,
    })
    client.post("/api/filaments/active", json={
        "brand": "ReviewTest", "name": "upd", "color": "#ffffff", "td": 5.0,
        "filament_type": "PLA", "uuid": uuid,
    })
    r = client.put(f"/api/filaments/{uuid}", json={
        "brand": "ReviewTest", "name": "upd-renamed", "color": "#000000", "td": 2.0,
        "filament_type": "PLA", "uuid": uuid,
    })
    assert r.status_code == 200
    active = client.get("/api/filaments/active").json()
    assert active[0]["td"] == 2.0, "active copy still held the old TD"
    assert active[0]["name"] == "upd-renamed"


# ---------------------------------------------------------------------------
# 14. /api/sliders/base answers for the job being looked at, not the newest
# ---------------------------------------------------------------------------


def test_base_color_endpoint_answers_for_a_specific_job(client):
    from autoforge.webui.services.optimization_service import get_optimization_service

    svc = get_optimization_service()
    older = svc.create_job({"iterations": 10, "input_image": "a.png"})
    svc.update_status(older.job_id, "completed")
    svc.set_pipeline_result(older.job_id, {"optimizer": object(), "source_tag": "older"})
    # A newer completed job whose result is not retained (e.g. computed in a
    # previous server session). Without the job_id pin, the endpoint used to
    # answer "the newest completed job" — this one — and the UI showed its
    # base under the older result the user was actually looking at.
    newer = svc.create_job({"iterations": 10, "input_image": "b.png"})
    svc.update_status(newer.job_id, "completed")

    # Pinned to the job the UI is looking at:
    r = client.get("/api/sliders/base", params={"job_id": older.job_id})
    assert r.status_code == 200
    assert r.json()["source"] == older.job_id
    # A job without a retained result yields no base (the caller falls back
    # to its own settings) instead of some *other* job's base:
    r2 = client.get("/api/sliders/base", params={"job_id": newer.job_id})
    assert r2.status_code == 200
    assert r2.json()["base"] is None
    r3 = client.get("/api/sliders/base", params={"job_id": "nope"})
    assert r3.status_code == 200
    assert r3.json()["base"] is None


# ---------------------------------------------------------------------------
# 15 (minor). An unrecognized pruning phase name buckets at the end of the
# progress bar, not the start.
# ---------------------------------------------------------------------------


def test_prune_progress_falls_back_to_the_last_bucket_for_an_unknown_phase():
    """Regression: `phases.index(phase_name) if phase_name in phases else 0`
    put an unrecognized phase name (e.g. "Removing spikes", which
    Optimizer.prune() reports but pruning.py's `phases` list did not
    include) in the *first* bucket, making the progress bar visibly jump
    backward to 0% right as pruning was finishing instead of climbing
    toward 100%. `post_remove_spikes` itself never fires a progress
    callback (it is one atomic tensor op), so this is exercised via source
    inspection — mirroring `test_pruning_forwards_the_polish_settings_to_
    the_optimizer` above for the same reason: the real callback is a
    closure nested inside a background thread inside a route handler, not
    something callable in isolation."""
    import inspect

    from autoforge.webui.api import pruning

    source = inspect.getsource(pruning.start_pruning)
    assert '"Removing spikes"' in source
    assert "phases.index(phase_name) if phase_name in phases else len(phases) - 1" in source
    # The old, wrong fallback must be gone, not just superseded textually.
    assert "phases.index(phase_name) if phase_name in phases else 0" not in source


# ---------------------------------------------------------------------------
# 16. Auto-repeat pruning must run spike removal once, on the settled pass —
# not on every intermediate pass.
# ---------------------------------------------------------------------------


def test_auto_repeat_pruning_applies_spike_removal_only_on_the_final_pass(client, monkeypatch):
    """Regression: the auto-repeat pruning loop called `export_results()` ->
    `optimizer.prune()` once per pass with spike removal on every time.
    Spike removal is a printability trade-off, not part of the color/swap/
    layer search (see Optimizer.post_remove_spikes) — its first application
    is explicitly allowed to *raise* the loss, so paying that cost on every
    intermediate pass, only for the next pass to prune the result further
    anyway, wasted both compute and accuracy. It must run exactly once, on
    whichever pass the loop actually settles on (converged or hit the pass
    limit), never on the passes before it."""
    import argparse

    from autoforge.webui.services.optimization_service import get_optimization_service
    import autoforge.Helper.PruningHelper as pruning_helper
    import autoforge.webui.helpers.pipeline_runner as pipeline_runner

    _active_filament(client)
    svc = get_optimization_service()

    class FakeOptimizer:
        def get_discretized_solution(self, best=True):
            return (1, 1)

    args = argparse.Namespace(spike_removal=True)
    done = svc.create_job({"iterations": 10, "input_image": "a.png"})
    svc.update_status(done.job_id, "completed")
    svc.set_pipeline_result(done.job_id, {"optimizer": FakeOptimizer(), "args": args})

    # Improves for two passes, then flatlines -> should_repeat_prune stops
    # after the pass that first repeats the same loss.
    losses = iter([100.0, 90.0, 90.0, 90.0, 90.0])
    monkeypatch.setattr(
        pruning_helper, "_compute_loss_for_heightmap",
        lambda *_a, **_k: next(losses),
    )

    calls: list[bool] = []

    def fake_export_results(result, cancel_event=None, pause_event=None, apply_spike_removal=True):
        calls.append(apply_spike_removal)
        return {"pruning_completed": True}

    monkeypatch.setattr(pipeline_runner, "export_results", fake_export_results)

    r = client.post("/api/pruning/start", json={
        "job_id": done.job_id, "auto_repeat": True, "max_passes": 25,
    })
    assert r.status_code == 200, r.text
    prune_job_id = r.json()["job_id"]
    _wait_for_status(client, prune_job_id, "completed", timeout=10.0)

    assert len(calls) >= 2, "expected at least one intermediate pass plus a final spike pass"
    assert calls[:-1] == [False] * (len(calls) - 1), (
        f"spike removal must be skipped on every pass but the last: {calls}"
    )
    assert calls[-1] is True


def test_single_pass_pruning_still_applies_spike_removal(client, monkeypatch):
    """A single pass (auto-repeat off) is always "the end" — it must not be
    skipped waiting for a repeat that will never come."""
    import argparse

    from autoforge.webui.services.optimization_service import get_optimization_service
    import autoforge.Helper.PruningHelper as pruning_helper
    import autoforge.webui.helpers.pipeline_runner as pipeline_runner

    _active_filament(client)
    svc = get_optimization_service()

    class FakeOptimizer:
        def get_discretized_solution(self, best=True):
            return (1, 1)

    args = argparse.Namespace(spike_removal=True)
    done = svc.create_job({"iterations": 10, "input_image": "a.png"})
    svc.update_status(done.job_id, "completed")
    svc.set_pipeline_result(done.job_id, {"optimizer": FakeOptimizer(), "args": args})

    monkeypatch.setattr(
        pruning_helper, "_compute_loss_for_heightmap", lambda *_a, **_k: 100.0,
    )

    calls: list[bool] = []

    def fake_export_results(result, cancel_event=None, pause_event=None, apply_spike_removal=True):
        calls.append(apply_spike_removal)
        return {"pruning_completed": True}

    monkeypatch.setattr(pipeline_runner, "export_results", fake_export_results)

    r = client.post("/api/pruning/start", json={"job_id": done.job_id, "auto_repeat": False})
    assert r.status_code == 200, r.text
    _wait_for_status(client, r.json()["job_id"], "completed", timeout=10.0)

    assert calls == [True]


# ---------------------------------------------------------------------------
# 17. A successful prune must clone into its own job/directory, not
# overwrite the job it started from — or history steps sharing that job id
# can never be told apart again.
# ---------------------------------------------------------------------------


def test_successful_prune_clones_into_its_own_job_and_directory(client, monkeypatch, tmp_path):
    """Regression ("pruning runs in place and destroys history"): pruning
    used to always write final_model.png/.stl/colored.ply etc. into
    checkpoints/<job_id> — the *same* directory the optimization job (and
    any earlier prune pass on it) had already written to. A second prune
    pass silently replaced the first pass's files in place, so a history
    step (or an F5 reload) that pointed at the *first* pass's job id showed
    whatever the *second* pass had most recently done to that shared
    directory instead. Confirmed with a real Playwright run: navigating to
    "prune 1a" in history after "prune 1b" had run later showed 1b's mesh
    under 1a's slider layout.

    Fixed by giving every successful prune its own job id and its own
    output directory (api/pruning.py), and having the frontend adopt that
    id as `currentJob` once it completes (appStore.startPruning) — so a
    later prune pass's own new directory is what changes, never an earlier
    one. This test covers the backend half: the new job's directory, its
    real JobStatus (restorable like any optimization job), and its aliased
    pipeline result (so a follow-up prune, or a slider-derivation lookup,
    still finds the live optimizer)."""
    import argparse

    from autoforge.webui.services.optimization_service import get_optimization_service
    from autoforge.webui.config import config
    import autoforge.webui.helpers.pipeline_runner as pipeline_runner

    _active_filament(client)
    svc = get_optimization_service()

    class FakeOptimizer:
        def get_discretized_solution(self, best=True):
            return (1, 1)

    args = argparse.Namespace(spike_removal=False)
    original_result = {"optimizer": FakeOptimizer(), "args": args}
    done = svc.create_job({"iterations": 10, "input_image": "history-clone.png"})
    svc.update_status(done.job_id, "completed")
    svc.set_pipeline_result(done.job_id, original_result)

    captured_output_dirs = []

    def fake_export_results(result, cancel_event=None, pause_event=None, apply_spike_removal=True):
        captured_output_dirs.append(result["args"].output_folder)
        return {"pruning_completed": True}

    monkeypatch.setattr(pipeline_runner, "export_results", fake_export_results)

    r = client.post("/api/pruning/start", json={"job_id": done.job_id, "auto_repeat": False})
    assert r.status_code == 200, r.text
    prune_job_id = r.json()["job_id"]
    _wait_for_status(client, prune_job_id, "completed", timeout=10.0)

    # A fresh, independent directory — not the original job's.
    assert len(captured_output_dirs) == 1
    assert os.path.basename(captured_output_dirs[0]) == prune_job_id
    assert os.path.basename(captured_output_dirs[0]) != done.job_id
    assert os.path.isdir(os.path.join(config.checkpoints_path, prune_job_id))

    # A real, independent job — restorable like any optimization job.
    prune_job = svc.get_job(prune_job_id)
    assert prune_job is not None
    assert prune_job.status == "completed"
    assert prune_job.input_image == "history-clone.png"
    assert svc.get_latest_job().job_id == prune_job_id

    # Its pipeline result is aliased, not moved: both ids resolve to a
    # result, and the original job's is untouched (a follow-up prune off
    # the *original* job, or an already-open tab still pointed at it,
    # must not find it silently emptied).
    assert svc.get_pipeline_result(prune_job_id) is not None
    assert svc.get_pipeline_result(prune_job_id)["optimizer"] is original_result["optimizer"]
    assert svc.get_pipeline_result(done.job_id) is original_result
    assert svc.get_pipeline_result(done.job_id)["optimizer"] is original_result["optimizer"]
