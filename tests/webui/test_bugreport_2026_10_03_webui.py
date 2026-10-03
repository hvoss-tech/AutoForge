"""Webui regression tests for the 2026-10-03 bug report (BUG_REPORT.md: H3,
M7 and the webui L-items)."""

import argparse
import asyncio
import os
import threading
import time

import numpy as np
import pytest
import torch
from fastapi.testclient import TestClient

from autoforge.webui.server import create_app


@pytest.fixture
def client():
    return TestClient(create_app())


# ---------------------------------------------------------------------------
# H3: DNS rebinding
# ---------------------------------------------------------------------------


def test_h3_unknown_host_names_are_refused(client):
    from starlette.websockets import WebSocketDisconnect

    from autoforge.webui.server import host_allowed

    for ok in ("localhost:8000", "127.0.0.1:8000", "[::1]:8000", "192.168.1.20", "app.localhost"):
        assert host_allowed(ok), ok
    assert not host_allowed("evil.example")
    assert not host_allowed("evil.example:8000")

    # A rebound attacker domain: Origin and Host agree, but neither is ours.
    rebound = {"Host": "evil.example:8000", "Origin": "http://evil.example:8000"}
    assert client.get("/api/system/health", headers=rebound).status_code == 403
    assert client.get("/api/system/health", headers={"Host": "evil.example"}).status_code == 403
    assert client.get("/api/system/health", headers={"Host": "127.0.0.1:8000"}).status_code == 200
    with pytest.raises(WebSocketDisconnect):
        with client.websocket_connect("/ws/preview", headers=rebound) as ws:
            ws.receive_text()


def test_h3_configured_hosts_are_allowed(monkeypatch):
    from autoforge.webui.config import config
    from autoforge.webui.server import host_allowed

    monkeypatch.setattr(config, "allowed_hosts", "autoforge.lan, other.example")
    assert host_allowed("autoforge.lan:8000")
    assert host_allowed("OTHER.example")
    assert not host_allowed("third.example")


# ---------------------------------------------------------------------------
# M7: atomic project-state / snapshot writes
# ---------------------------------------------------------------------------


def test_m7_failed_write_keeps_the_previous_file(tmp_path):
    from autoforge.webui.services.project_service import atomic_write_json

    path = tmp_path / "state.json"
    atomic_write_json(str(path), {"a": 1})
    with pytest.raises(TypeError):
        atomic_write_json(str(path), {"a": object()})
    assert path.read_text(encoding="utf-8").strip().startswith("{")
    assert '"a": 1' in path.read_text(encoding="utf-8")
    assert os.listdir(tmp_path) == ["state.json"]


# ---------------------------------------------------------------------------
# L16-L26
# ---------------------------------------------------------------------------


def test_l16_worker_is_registered_before_it_starts(monkeypatch):
    from autoforge.webui.api import workers

    seen = {}
    real_start = threading.Thread.start

    def start(self):
        with workers._lock:
            seen["registered"] = any(t is self for t in workers._threads.values())
        seen["running"] = workers.running_worker()
        return real_start(self)

    monkeypatch.setattr(threading.Thread, "start", start)
    gate = threading.Event()
    t = workers.start_worker("job-l16", gate.wait)
    assert seen == {"registered": True, "running": "job-l16"}
    gate.set()
    t.join(2)


def test_l17_slow_preview_client_gets_only_the_newest_update(monkeypatch):
    from autoforge.webui.api import ws as ws_api

    loop = asyncio.new_event_loop()
    runner = threading.Thread(target=loop.run_forever, daemon=True)
    runner.start()
    sent = []
    first_started = threading.Event()
    release = asyncio.Event()

    class SlowSocket:
        async def send_text(self, msg):
            sent.append(msg)
            if len(sent) == 1:
                first_started.set()
                await release.wait()

    sock = SlowSocket()
    monkeypatch.setattr(ws_api, "_main_loop", loop)
    monkeypatch.setattr(ws_api, "_preview_connections", {sock})
    try:
        ws_api._send_to_all("m1")
        assert first_started.wait(2)
        for msg in ("m2", "m3", "m4"):
            ws_api._send_to_all(msg)
        loop.call_soon_threadsafe(release.set)
        deadline = time.time() + 2
        while time.time() < deadline and sent[-1] != "m4":
            time.sleep(0.01)
        assert sent[0] == "m1" and sent[-1] == "m4"
        assert len(sent) <= 3  # m2/m3 superseded, not each queued
    finally:
        loop.call_soon_threadsafe(loop.stop)
        runner.join(2)


def test_l18_slider_composite_matches_the_discrete_composite():
    from autoforge.Helper.OptimizerHelper import composite_image_disc
    from autoforge.Helper.PruningHelper import disc_to_logits
    from autoforge.Helper.HeightAssign import heights_to_logits
    from autoforge.webui.helpers.slider_render import composite_from_slider_stack

    g = torch.Generator().manual_seed(4)
    L, M = 60, 3  # more layers than one chunk
    z = torch.randint(0, L + 1, (14, 11), generator=g)
    dg = torch.randint(0, M, (L,), generator=g)
    mc = torch.rand(M, 3, generator=g)
    td = torch.rand(M, generator=g) * 4 + 0.5
    bg = torch.rand(3, generator=g)
    ours = composite_from_slider_stack(z, dg.numpy(), mc, td, bg, 0.04, L)
    ref = composite_image_disc(heights_to_logits(z, L), disc_to_logits(dg, M), 0.01, 0.01, 0.04, L, mc, td, bg, rng_seed=0)
    assert torch.allclose(ours, ref, atol=0.05)


def test_l19_slider_derivation_runs_off_the_event_loop(client, monkeypatch):
    from autoforge.webui.api import sliders as sliders_api
    from autoforge.webui.services.optimization_service import get_optimization_service

    svc = get_optimization_service()
    job = svc.create_job({"iterations": 1})
    svc.update_status(job.job_id, "completed")
    svc.set_pipeline_result(job.job_id, {"optimizer": object(), "args": argparse.Namespace()})
    where = {}

    def derive(_result):
        try:
            asyncio.get_running_loop()
            where["loop"] = True
        except RuntimeError:
            where["loop"] = False
        return {"sliders": [], "min_layer": 0, "max_layer": 5}

    monkeypatch.setattr(sliders_api, "derive_sliders_from_result", derive)
    r = client.get(f"/api/sliders/from-optimizer?job_id={job.job_id}")
    assert r.status_code == 200 and r.json()["max_layer"] == 5
    assert where == {"loop": False}


def _edited_result(material_uuids, background="#000000", base_index=None):
    class Opt:
        max_layers = 4

        def get_discretized_solution(self, best=True):
            return torch.tensor([0, 1, 1, 0]), torch.full((3, 3), 4)

    args = argparse.Namespace(background_height=0.24, layer_height=0.04, max_layers=4,
                              background_color=background, stl_output_size=20,
                              background_material_index=base_index)
    return {"optimizer": Opt(), "args": args, "output_target": torch.zeros(3, 3, 3),
            "material_uuids": material_uuids}


def test_l20_edited_export_keeps_per_band_tds_and_the_edited_base(tmp_path):
    import json

    from autoforge.webui.helpers.edited_export import edited_stack, write_edited_outputs

    sliders = [
        {"layer": 2, "enabled": True, "filament_uuid": "a", "td": 2.0},
        {"layer": 4, "enabled": True, "filament_uuid": "a", "td": 6.0},
    ]
    dg, uuids, mats = edited_stack(sliders, 4)
    assert uuids == ["a", "a"] and [m["td"] for m in mats] == [2.0, 6.0]
    assert dg.tolist() == [0, 0, 1, 1]

    filaments = {"a": {"brand": "B", "name": "Red", "color": "#ff0000"},
                 "w": {"brand": "B", "name": "White", "color": "#ffffff"}}
    edited = [
        {"layer": 2, "enabled": True, "filament_uuid": "a", "td": 2.0},
        {"layer": 4, "enabled": True, "filament_uuid": "w", "td": 5.0},
    ]
    # Run base: filament "a" (index 0 of the run). The UI changed it to white.
    out = write_edited_outputs(_edited_result(["a"], "#FF0000", 0), edited, filaments, str(tmp_path),
                               background_color="#ffffff")
    hfp = json.loads(open(out["project_file.hfp"], encoding="utf-8").read())
    background = [f for f in hfp["filament_set"] if f["Color"].upper() == "#FFFFFF"]
    assert background, hfp["filament_set"]
    text = open(out["swap_instructions.txt"], encoding="utf-8").read()
    assert "B - White" in text.splitlines()[0]


def test_l21_csv_rows_without_td_are_skipped_and_reported(client):
    csv = "Brand,Name,Color,TD\nB,Good,#ff0000,1.5\nB,NoTd,#00ff00,\nB,Zero,#0000ff,0\n"
    r = client.post("/api/filaments/import-csv", json={"contents": csv})
    assert r.status_code == 200, r.text
    assert r.json()["count"] == 1 and r.json()["skipped"] == 2
    names = [f["name"] for f in client.get("/api/filaments").json()]
    assert "Good" in names and "NoTd" not in names and "Zero" not in names

    r = client.post("/api/filaments/import-csv", json={"contents": "Brand,Name,Color,TD\nB,X,#ff0000,\n", "mode": "replace"})
    assert r.status_code == 400
    assert "Good" in [f["name"] for f in client.get("/api/filaments").json()]  # nothing replaced


def test_l22_library_path_is_absolute(tmp_path, monkeypatch):
    from autoforge.webui.config import config
    from autoforge.webui.services import filament_service
    from autoforge.webui.services.catalog_service import get_catalog_service, reset_catalog_service

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(config, "library_dir", "relative_library")
    filament_service.reset_service()
    reset_catalog_service()
    try:
        expected = str(tmp_path / "relative_library")
        assert filament_service.get_filament_service()._library_path == expected
        assert get_catalog_service()._library_path == expected
    finally:
        filament_service.reset_service()
        reset_catalog_service()


def test_l23_telemetry_scrubs_the_home_folder(monkeypatch):
    from pathlib import Path

    from autoforge.webui.helpers import telemetry

    home = str(Path.home())
    assert telemetry.scrub_paths(f"File {home}/AutoForge2/x.py, line 3") == "File ~/AutoForge2/x.py, line 3"

    captured = {}

    class Client:
        def capture(self, **kw):
            captured.update(kw)

    monkeypatch.setattr(telemetry, "_get_client", lambda: Client())
    try:
        raise ValueError(f"cannot open {home}/secret/photo.png")
    except ValueError as e:
        telemetry.capture_exception(e, {"path": f"{home}/x"})
    props = captured["properties"]
    assert home not in props["$exception_message"]
    assert home not in props["$exception_stack_trace_raw"]
    assert props["path"] == "~/x"


def test_l24_alpha_mask_of_a_one_pixel_high_image():
    from autoforge.webui.helpers.colored_mesh import _valid_mask, top_vertex_pixel_indices

    alpha = np.full((1, 5, 1), 255, np.uint8)
    assert _valid_mask(alpha, 1, 5).shape == (1, 5)
    assert top_vertex_pixel_indices(1, 5, alpha).size == 0  # no 2x2 cell to mesh


def test_l25_finished_mesh_writes_are_forgotten():
    from autoforge.webui.helpers import mesh_persist

    done = threading.Event()
    mesh_persist.schedule_persist("l25-target", done.set)
    assert done.wait(3)
    mesh_persist.wait_for_pending_mesh("l25-target", timeout=3)
    deadline = time.time() + 2
    while time.time() < deadline and "l25-target" in mesh_persist._persist:
        time.sleep(0.02)
    assert "l25-target" not in mesh_persist._persist


def test_l26_identical_uploads_share_one_file(client):
    import cv2

    ok, png = cv2.imencode(".png", np.full((8, 8, 3), 77, np.uint8))
    names = [
        client.post("/api/images/upload", files={"file": (f"p{i}.png", png.tobytes(), "image/png")}).json()["filename"]
        for i in range(3)
    ]
    assert len(set(names)) == 1
    from autoforge.webui.config import config

    assert len([f for f in os.listdir(config.uploads_path) if not f.endswith(".tmp")]) == 1
