"""Regressions for the 2026-09-24 WebUI bug hunt.

Each test fails on the pre-fix code and passes after the fix.
"""

import json
import os
import time

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from autoforge.webui.server import create_app


@pytest.fixture
def client():
    return TestClient(create_app())


def _active_filament(client, uuid="bf-fil-1", color="#123456", td=5.0):
    body = {"brand": "BugFix", "name": f"bf-{uuid}", "color": color, "td": td,
            "filament_type": "PLA", "uuid": uuid}
    client.post("/api/filaments", json=body)
    client.post("/api/filaments/active", json=body)
    return uuid


def _upload_png(client, name, img):
    ok, buf = cv2.imencode(".png", img)
    assert ok
    r = client.post("/api/images/upload", files={"file": (name, buf.tobytes(), "image/png")})
    assert r.status_code == 200, r.text
    return r.json()["filename"]


def _wait_for_status(client, job_id, status, timeout=10.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        r = client.get(f"/api/optimize/status/{job_id}")
        if r.is_success and r.json().get("status") == status:
            return r.json()
        time.sleep(0.05)
    raise AssertionError(f"job {job_id} never reached {status!r}")


def _fake_init_result():
    return {
        "result": {"optimizer": object()},
        "preview_b64": "png",
        "range": {"min_layer": 0, "max_layer": 30},
        "base": {"color": "#111111", "height_mm": 0.24, "layers": 6, "filament_uuid": "", "auto": True},
    }


# ---------------------------------------------------------------------------
# 1. /api/init/run must only ever read files from uploads/
# ---------------------------------------------------------------------------


def test_init_run_reads_the_upload_not_a_same_named_file_in_the_cwd(client, monkeypatch, tmp_path):
    """The init endpoint tried the name as a path first, so a file with the
    upload's name in the server's working directory (the repo ships an
    input.png) was used for the preview instead of the uploaded image."""
    import autoforge.webui.api.init as init_api
    from autoforge.webui.config import config

    _active_filament(client)
    os.makedirs(config.uploads_path, exist_ok=True)
    with open(os.path.join(config.uploads_path, "input.png"), "wb") as f:
        f.write(b"uploaded")
    cwd = tmp_path / "cwd"
    cwd.mkdir()
    (cwd / "input.png").write_bytes(b"not the upload")
    monkeypatch.chdir(cwd)

    seen = {}

    def fake_run_init_sync(input_image_path, filament_dicts, settings_dict):
        seen["path"] = input_image_path
        return _fake_init_result()

    monkeypatch.setattr(init_api, "_run_init_sync", fake_run_init_sync)
    r = client.post("/api/init/run", json={"input_image": "input.png", "num_init_rounds": 1})
    assert r.status_code == 200, r.text
    assert os.path.realpath(seen["path"]) == os.path.realpath(os.path.join(config.uploads_path, "input.png"))


def test_init_run_rejects_paths_outside_uploads(client, monkeypatch, tmp_path):
    """/api/optimize/start already refused these; the preview accepted any
    existing server path."""
    import autoforge.webui.api.init as init_api

    _active_filament(client)
    outside = tmp_path / "secret.png"
    outside.write_bytes(b"x")
    called = []
    monkeypatch.setattr(init_api, "_run_init_sync", lambda *a, **k: called.append(a) or _fake_init_result())

    for name in (str(outside), "../" * 12 + str(outside).lstrip("/")):
        r = client.post("/api/init/run", json={"input_image": name, "num_init_rounds": 1})
        assert r.status_code == 404, (name, r.text)
    assert called == []
    assert client.get("/api/init/status").json()["status"] == "idle"


# ---------------------------------------------------------------------------
# 2. Library imports: HueForge JSON, Owned column, colors, active copies
# ---------------------------------------------------------------------------


def test_hueforge_personal_library_json_imports_real_filaments(client):
    """{"Filaments": [...]} with PascalCase keys (what the CLI's --json_file
    reads) was imported as ONE nameless white filament, reported as success."""
    library = {"Filaments": [
        {"Brand": "Bambu", "Name": "Jade White", "Color": "#F0F0E8", "Transmissivity": 5.2,
         "Type": "PLA", "Owned": True, "Uuid": "hf-1"},
        {"Brand": "Polymaker", "Name": "Blue", "Color": "#1e90ff", "Transmissivity": "3.1",
         "Type": "PETG", "Owned": False, "Uuid": "hf-2"},
    ]}
    r = client.post("/api/filaments/import-json?mode=replace", json=library)
    assert r.status_code == 200, r.text
    assert r.json()["count"] == 2
    by_uuid = {f["uuid"]: f for f in client.get("/api/filaments").json()}
    assert set(by_uuid) == {"hf-1", "hf-2"}
    assert by_uuid["hf-1"]["brand"] == "Bambu"
    assert by_uuid["hf-1"]["name"] == "Jade White"
    assert by_uuid["hf-1"]["td"] == pytest.approx(5.2)
    assert by_uuid["hf-1"]["owned"] is True
    assert by_uuid["hf-1"]["filament_type"] == "PLA"
    assert by_uuid["hf-2"]["td"] == pytest.approx(3.1)
    assert by_uuid["hf-2"]["owned"] is False


def test_json_import_rejects_non_filament_entries(client):
    client.post("/api/filaments", json={"uuid": "keep", "brand": "B", "name": "Keep", "td": 1.0})
    for body in ([{"foo": 1}], {"something": "else"}, {"Filaments": []}, []):
        r = client.post("/api/filaments/import-json?mode=replace", json=body)
        assert r.status_code == 400, (body, r.text)
    assert [f["uuid"] for f in client.get("/api/filaments").json()] == ["keep"]


def test_csv_import_keeps_the_owned_column(client):
    csv = "Brand,Type,Color,Name,TD,Owned,UUID\nB,PLA,#000000,Black,0.6,True,c-1\nB,PLA,#ffffff,White,5,false,c-2\n"
    r = client.post("/api/filaments/import-csv", json={"contents": csv, "mode": "replace"})
    assert r.status_code == 200, r.text
    owned = {f["uuid"]: f["owned"] for f in client.get("/api/filaments").json()}
    assert owned == {"c-1": True, "c-2": False}


def test_filament_colors_are_normalized_or_rejected(client):
    """hex_to_rgb slices fixed offsets: '#fff' or 'red' reached the pipeline
    and failed only once a run started; a color without '#' drew nothing."""
    r = client.post("/api/filaments", json={"uuid": "c1", "name": "A", "color": "00ae42", "td": 1})
    assert r.status_code == 200 and r.json()["color"] == "#00ae42"
    r = client.post("/api/filaments", json={"uuid": "c2", "name": "B", "color": "#abc", "td": 1})
    assert r.json()["color"] == "#aabbcc"
    r = client.post("/api/filaments", json={"uuid": "c3", "name": "C", "color": " #11223344 ", "td": 1})
    assert r.json()["color"] == "#112233"
    r = client.post("/api/filaments", json={"uuid": "c4", "name": "D", "color": "red", "td": 1})
    assert r.status_code == 422


def test_csv_with_an_invalid_color_is_rejected_without_changing_the_library(client):
    client.post("/api/filaments", json={"uuid": "keep", "brand": "B", "name": "Keep", "td": 1.0})
    csv = "Brand,Name,Color,TD\nB,Good,#ff0000,1\nB,Bad,not-a-color,1\n"
    r = client.post("/api/filaments/import-csv", json={"contents": csv, "mode": "replace"})
    assert r.status_code == 400, r.text
    assert "color" in r.json()["detail"].lower()
    assert [f["uuid"] for f in client.get("/api/filaments").json()] == ["keep"]


@pytest.mark.parametrize("mode", ["merge", "replace"])
def test_reimporting_the_library_updates_active_filaments(client, mode):
    """Runs read the active list. A re-import with corrected TDs/colors
    updated the library but left every active copy on the old values."""
    csv_v1 = "Brand,Name,Color,TD,UUID\nB,Red,#ff0000,1.0,r-1\n"
    csv_v2 = "Brand,Name,Color,TD,UUID\nB,Red,#ee1111,4.5,r-1\n"
    client.post("/api/filaments/import-csv", json={"contents": csv_v1})
    [red] = client.get("/api/filaments").json()
    client.post("/api/filaments/active", json=red)

    r = client.post("/api/filaments/import-csv", json={"contents": csv_v2, "mode": mode})
    assert r.status_code == 200, r.text
    [active] = client.get("/api/filaments/active").json()
    assert active["td"] == pytest.approx(4.5)
    assert active["color"] == "#ee1111"


def test_one_invalid_library_entry_does_not_take_the_library_down(client):
    """FilamentService.__init__ raised on the first entry pydantic rejected,
    so every filament route (and server startup seeding) failed."""
    from autoforge.webui.config import config
    from autoforge.webui.services import filament_service

    os.makedirs(config.library_dir, exist_ok=True)
    good = {"uuid": "ok-1", "brand": "B", "name": "Good", "color": "#123456", "td": 2.0}
    bad = {"uuid": "bad-1", "brand": "B", "name": "Bad", "color": "#123456", "td": "lots"}
    with open(os.path.join(config.library_dir, "library.json"), "w") as f:
        json.dump([good, bad], f)
    with open(os.path.join(config.library_dir, "active.json"), "w") as f:
        json.dump([bad, good], f)
    filament_service.reset_service()

    r = client.get("/api/filaments")
    assert r.status_code == 200
    assert [f["uuid"] for f in r.json()] == ["ok-1"]
    assert [f["uuid"] for f in client.get("/api/filaments/active").json()] == ["ok-1"]


# ---------------------------------------------------------------------------
# 3. Images the upload accepts must also be runnable
# ---------------------------------------------------------------------------


def test_grayscale_16bit_and_gray_alpha_images_become_8bit_bgr():
    from autoforge.webui.helpers.pipeline_runner import to_bgr_or_bgra_uint8

    gray = np.full((4, 5), 200, np.uint8)
    out = to_bgr_or_bgra_uint8(gray)
    assert out.shape == (4, 5, 3) and out.dtype == np.uint8 and (out == 200).all()

    deep = np.full((4, 5, 3), 65535, np.uint16)
    deep[0, 0] = (0, 257 * 128, 65535)
    out = to_bgr_or_bgra_uint8(deep)
    assert out.dtype == np.uint8
    assert tuple(out[0, 0]) == (0, 128, 255)
    assert (out[1:] == 255).all()

    gray_alpha = np.zeros((4, 5, 2), np.uint8)
    gray_alpha[..., 0] = 90
    gray_alpha[..., 1] = 255
    gray_alpha[0, 0, 1] = 0
    out = to_bgr_or_bgra_uint8(gray_alpha)
    assert out.shape == (4, 5, 4)
    assert (out[..., :3] == 90).all()
    assert out[0, 0, 3] == 0 and out[1, 1, 3] == 255

    color = np.random.randint(0, 255, (3, 3, 3), np.uint8)
    assert to_bgr_or_bgra_uint8(color) is color

    with pytest.raises(ValueError):
        to_bgr_or_bgra_uint8(None)


def test_pipeline_state_builds_from_a_grayscale_upload(monkeypatch, tmp_path):
    """build_pipeline_state indexed img.shape[2]: a grayscale PNG (which
    /api/images/upload accepts) failed with 'tuple index out of range'."""
    import autoforge.webui.helpers.pipeline_runner as pr

    path = tmp_path / "gray.png"
    cv2.imwrite(str(path), np.tile(np.linspace(0, 255, 32, dtype=np.uint8), (24, 1)))
    assert cv2.imread(str(path), cv2.IMREAD_UNCHANGED).ndim == 2

    seen = {}

    def stop_after_background(args, img_rgb, alpha, *rest):
        seen["shape"] = img_rgb.shape
        seen["alpha"] = alpha
        raise RuntimeError("stop here")

    monkeypatch.setattr(pr, "_auto_select_background_color", stop_after_background)
    filaments = [{"color": "#ffffff", "td": 5.0, "name": "W", "uuid": "w"}]
    with pytest.raises(RuntimeError, match="stop here"):
        pr.build_pipeline_state(str(path), filaments, str(tmp_path / "out"), {"device": "cpu", "random_seed": 1})
    assert seen["shape"] == (24, 32, 3)
    assert seen["alpha"] is None


def test_upload_never_keeps_a_non_image_extension(client):
    ok, buf = cv2.imencode(".png", np.zeros((4, 4, 3), np.uint8))
    for name in ("page.html", "script.js", "noext", "x.svg"):
        r = client.post("/api/images/upload", files={"file": (name, buf.tobytes(), "image/png")})
        assert r.status_code == 200, r.text
        assert r.json()["filename"].endswith(".png"), name
    r = client.post("/api/images/upload", files={"file": ("photo.JPG", buf.tobytes(), "image/jpeg")})
    assert r.json()["filename"].endswith(".jpg")


# ---------------------------------------------------------------------------
# 4. Base height must be a whole number of layers (the CLI's rule)
# ---------------------------------------------------------------------------


def test_run_rejects_a_base_height_that_is_not_a_multiple_of_the_layer_height(client):
    _active_filament(client)
    image = _upload_png(client, "base.png", np.full((8, 8, 3), 120, np.uint8))
    r = client.post("/api/optimize/start", json={
        "input_image": image, "iterations": 1, "layer_height": 0.08, "background_height": 0.2,
    })
    assert r.status_code == 400, r.text
    assert "multiple of the layer height" in r.json()["detail"]
    assert client.get("/api/optimize/history").json() == []


def test_base_height_check_tolerates_floating_point_ratios():
    from autoforge.webui.models import OptimizationSettings

    # 0.28 / 0.04 == 7.000000000000001 — the CLI's is_integer() check would
    # reject this; it is a perfectly valid 7-layer base.
    for bh, lh in ((0.28, 0.04), (0.24, 0.04), (0.6, 0.12), (0.0, 0.04), (0.36, 0.12)):
        assert OptimizationSettings(background_height=bh, layer_height=lh).base_height_error() is None, (bh, lh)
    assert OptimizationSettings(background_height=0.3, layer_height=0.08).base_height_error()


# ---------------------------------------------------------------------------
# 5. Pruning claims the shared optimizer: the job it started from must stop
#    resolving to it
# ---------------------------------------------------------------------------


class _FakeOptimizer:
    def get_discretized_solution(self, best=True):
        return (1, 1)


def _completed_job_with_result(svc, input_image="a.png"):
    import argparse

    job = svc.create_job({"iterations": 10, "input_image": input_image})
    svc.update_status(job.job_id, "completed")
    result = {"optimizer": _FakeOptimizer(), "args": argparse.Namespace(spike_removal=False)}
    svc.set_pipeline_result(job.job_id, result)
    return job, result


def test_a_pruned_result_no_longer_serves_the_job_it_started_from(client, monkeypatch):
    """Pruning mutates the optimizer in place and aliases it to the prune
    job — but the original job id kept resolving to that same, now pruned,
    optimizer. Undoing to the unpruned result and editing a color rendered
    the pruned heights into the unpruned job's folder."""
    import autoforge.Helper.PruningHelper as pruning_helper
    import autoforge.webui.helpers.pipeline_runner as pipeline_runner
    import autoforge.webui.api.preview as preview_api
    from autoforge.webui.services.optimization_service import get_optimization_service

    _active_filament(client)
    svc = get_optimization_service()
    job, result = _completed_job_with_result(svc)
    monkeypatch.setattr(pruning_helper, "_compute_loss_for_heightmap", lambda *_a, **_k: 1.0)
    monkeypatch.setattr(pipeline_runner, "export_results", lambda *a, **k: {"pruning_completed": True})
    renders = []
    monkeypatch.setattr(preview_api, "render_with_sliders", lambda pr, *a, **k: renders.append(pr) or None)

    r = client.post("/api/pruning/start", json={"job_id": job.job_id})
    assert r.status_code == 200, r.text
    prune_id = r.json()["job_id"]
    _wait_for_status(client, prune_id, "completed")

    assert svc.get_pipeline_result(prune_id) is result
    assert svc.get_pipeline_result(job.job_id) is None

    stale = client.post("/api/preview/render-with-sliders", json={"job_id": job.job_id, "sliders": []})
    assert stale.status_code == 409
    assert "pruned" in stale.json()["detail"]
    assert renders == []
    ok = client.post("/api/preview/render-with-sliders", json={"job_id": prune_id, "sliders": []})
    assert ok.status_code == 200, ok.text
    assert renders == [result]

    # Pruning the old id again would have re-pruned the pruned optimizer
    # while claiming to start from the unpruned result.
    again = client.post("/api/pruning/start", json={"job_id": job.job_id})
    assert again.status_code == 200
    failed = _wait_for_status(client, again.json()["job_id"], "failed")
    assert "pruned" in failed["error"]


def test_pruning_keeps_the_original_jobs_slider_edits(client, monkeypatch):
    """Pruning writes into its own folder now, yet it still deleted the
    original job's slider-edited mesh — an undo back to that result then
    showed the optimizer's colors next to the user's restored layers."""
    import autoforge.Helper.PruningHelper as pruning_helper
    import autoforge.webui.helpers.pipeline_runner as pipeline_runner
    from autoforge.webui.api.outputs import EDITED_PLY
    from autoforge.webui.config import config
    from autoforge.webui.services.optimization_service import get_optimization_service

    _active_filament(client)
    svc = get_optimization_service()
    job, _result = _completed_job_with_result(svc)
    job_dir = os.path.join(config.checkpoints_path, job.job_id)
    os.makedirs(job_dir, exist_ok=True)
    with open(os.path.join(job_dir, "final_model_colored.ply"), "w") as f:
        f.write("optimizer colors")
    with open(os.path.join(job_dir, EDITED_PLY), "w") as f:
        f.write("user colors")
    monkeypatch.setattr(pruning_helper, "_compute_loss_for_heightmap", lambda *_a, **_k: 1.0)
    monkeypatch.setattr(pipeline_runner, "export_results", lambda *a, **k: {"pruning_completed": True})

    r = client.post("/api/pruning/start", json={"job_id": job.job_id})
    _wait_for_status(client, r.json()["job_id"], "completed")
    assert client.get(f"/api/outputs/colored-ply/{job.job_id}").text == "user colors"


def test_slider_lookups_for_a_pruned_job_fall_through_to_the_newest_result(client):
    from autoforge.webui.services.optimization_service import get_optimization_service

    svc = get_optimization_service()
    job, result = _completed_job_with_result(svc)
    claimed = svc.claim_pipeline_result("prune-abc", job.job_id)
    assert claimed is result
    assert svc.get_pipeline_result(job.job_id) is None
    assert svc.claim_pipeline_result("prune-def", job.job_id) is None
    r = client.get(f"/api/sliders/from-optimizer?job_id={job.job_id}")
    assert r.json() == {"sliders": [], "min_layer": 0, "max_layer": 75}
    assert client.get(f"/api/sliders/base?job_id={job.job_id}").json() == {"base": None, "source": None}

    # A fresh run owns its own result again.
    svc.set_pipeline_result(job.job_id, result)
    assert svc.get_pipeline_result(job.job_id) is result


# ---------------------------------------------------------------------------
# 6. HueForge personal library: found where HueForge keeps it, offered once
# ---------------------------------------------------------------------------

HUEFORGE_LIBRARY = {"Filaments": [
    # Shape of a real HueForge file (lowercase uuid in braces, Tags list).
    {"Brand": "MY", "Color": "#595959", "Name": "HPLA GREY", "Owned": True, "Tags": [],
     "Transmissivity": 1, "Type": "PLA", "uuid": "{0268695a-962a-40f8-b9a5-4ff887755846}"},
    {"Brand": "MY", "Color": "#F2F2F2", "Name": "HPLA WHITE", "Owned": False, "Tags": [],
     "Transmissivity": 4.5, "Type": "PLA", "uuid": "{11111111-962a-40f8-b9a5-4ff887755846}"},
]}


def _write_hueforge_library(content=HUEFORGE_LIBRARY):
    from autoforge.webui.config import config

    path = config.hueforge_library_path
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(content if isinstance(content, str) else json.dumps(content))
    return path


def test_hueforge_library_locations_per_platform(monkeypatch, tmp_path):
    from autoforge.webui.config import WebUIConfig

    tail = os.path.join("HueForge", "Filaments", "personal_library.json")
    monkeypatch.setenv("HOME", str(tmp_path))
    for var in ("APPDATA", "XDG_DATA_HOME", "XDG_CONFIG_HOME", "AUTOFORGE_WEBUI_HUEFORGE_LIBRARY"):
        monkeypatch.delenv(var, raising=False)
    cfg = WebUIConfig()

    monkeypatch.setattr("sys.platform", "linux")
    assert cfg.hueforge_library_candidates() == [
        str(tmp_path / ".local" / "share" / tail), str(tmp_path / ".config" / tail),
    ]
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "xdg"))
    assert cfg.hueforge_library_candidates()[0] == str(tmp_path / "xdg" / tail)

    monkeypatch.setattr("sys.platform", "darwin")
    assert cfg.hueforge_library_candidates() == [str(tmp_path / "Library" / "Application Support" / tail)]

    monkeypatch.setattr("sys.platform", "win32")
    assert cfg.hueforge_library_candidates() == []
    monkeypatch.setenv("APPDATA", str(tmp_path / "Roaming"))
    assert cfg.hueforge_library_candidates() == [os.path.join(str(tmp_path / "Roaming"), tail)]

    # The first candidate that exists wins, else the most likely one.
    monkeypatch.setattr("sys.platform", "linux")
    monkeypatch.delenv("APPDATA")
    monkeypatch.delenv("XDG_DATA_HOME")
    assert cfg.hueforge_library_path == str(tmp_path / ".local" / "share" / tail)
    fallback = tmp_path / ".config" / tail
    fallback.parent.mkdir(parents=True)
    fallback.write_text("{}")
    assert cfg.hueforge_library_path == str(fallback)

    monkeypatch.setenv("AUTOFORGE_WEBUI_HUEFORGE_LIBRARY", str(tmp_path / "custom.json"))
    assert WebUIConfig().hueforge_library_candidates() == [str(tmp_path / "custom.json")]


def test_hueforge_library_is_reported_and_imported(client):
    info = client.get("/api/filaments/hueforge-library").json()
    assert info["found"] is False and info["count"] == 0
    assert client.post("/api/filaments/import-hueforge").status_code == 404

    path = _write_hueforge_library()
    info = client.get("/api/filaments/hueforge-library").json()
    assert info == {"found": True, "path": path, "count": 2, "error": None, "offered": False}

    r = client.post("/api/filaments/import-hueforge?mode=merge")
    assert r.status_code == 200, r.text
    assert r.json()["count"] == 2
    by_name = {f["name"]: f for f in client.get("/api/filaments").json()}
    assert by_name["HPLA GREY"]["uuid"] == "{0268695a-962a-40f8-b9a5-4ff887755846}"
    assert by_name["HPLA GREY"]["owned"] is True and by_name["HPLA GREY"]["td"] == 1
    assert by_name["HPLA WHITE"]["color"] == "#F2F2F2" and by_name["HPLA WHITE"]["td"] == 4.5
    # Importing it is as good as having been offered it.
    assert client.get("/api/filaments/hueforge-library").json()["offered"] is True


def test_the_first_start_offer_is_made_once_and_survives_a_restart(client):
    from autoforge.webui.services import filament_service

    _write_hueforge_library()
    assert client.get("/api/filaments/hueforge-library").json()["offered"] is False
    assert client.post("/api/filaments/hueforge-library/offered").status_code == 200
    filament_service.reset_service()  # a server restart
    assert client.get("/api/filaments/hueforge-library").json()["offered"] is True


def test_an_unreadable_hueforge_library_is_reported_not_imported(client):
    _write_hueforge_library("{not json")
    info = client.get("/api/filaments/hueforge-library").json()
    assert info["found"] is False and "could not be read" in info["error"]
    r = client.post("/api/filaments/import-hueforge")
    assert r.status_code == 400
    assert client.get("/api/filaments").json() == []
