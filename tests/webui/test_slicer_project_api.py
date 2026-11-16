"""File › Slicer project (.3mf) in the webui: printer list, export endpoint
and the export bundle."""

import io
import json
import os
import sys
import tempfile
import urllib.parse
import xml.etree.ElementTree as ET
import zipfile

import numpy as np
import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from orca_fixtures import make_profiles  # noqa: E402

from autoforge.Helper import OrcaProfiles as OP  # noqa: E402
from autoforge.Helper.OutputHelper import generate_stl  # noqa: E402
from autoforge.webui.server import create_app  # noqa: E402

LH, BG = 0.08, 0.24
MATERIALS = [
    {"uuid": "u-black", "color": "#000000", "td": 1.0, "name": "Black", "brand": "Acme", "filament_type": "PLA"},
    {"uuid": "u-red", "color": "#FF0000", "td": 2.0, "name": "Red", "brand": "Acme", "filament_type": "PETG"},
    {"uuid": "u-white", "color": "#FFFFFF", "td": 3.0, "name": "White", "brand": "Acme", "filament_type": "PLA"},
]


@pytest.fixture
def acme(tmp_path, monkeypatch):
    root = make_profiles(str(tmp_path / "profiles"))
    monkeypatch.setattr(OP, "_index_cache", OP.build_printer_index(root))
    monkeypatch.setenv("AUTOFORGE_ORCA_PROFILES", root)
    return root


@pytest.fixture
def client():
    return TestClient(create_app())


def _job(stack=(0, 0, 1, 1, 2, 2), base_index=0, size_mm=30, nozzle=None):
    """A completed job with an STL and the saved solution it came from."""
    from autoforge.webui.config import config
    from autoforge.webui.helpers.result_store import SOLUTION_FILE
    from autoforge.webui.services.optimization_service import get_optimization_service

    svc = get_optimization_service()
    job = svc.create_job({"iterations": 10, "input_image": "a.png"})
    svc.update_status(job.job_id, "completed")
    job_dir = os.path.join(config.checkpoints_dir, job.job_id)
    os.makedirs(job_dir, exist_ok=True)
    heights = np.full((20, 20), len(stack), dtype=np.int64)
    heights[:5, :5] = 2
    generate_stl(heights.astype(np.float32) * LH, os.path.join(job_dir, "final_model.stl"), BG, maximum_x_y_size=size_mm)
    meta = {
        "version": 1, "materials": MATERIALS,
        "args": {"layer_height": LH, "background_height": BG, **({"nozzle_diameter": nozzle} if nozzle else {})},
        "background_color": "#000000", "background_material_index": base_index,
    }
    np.savez_compressed(
        os.path.join(job_dir, SOLUTION_FILE),
        heights=heights, stack=np.asarray(stack, dtype=np.int64),
        meta=np.frombuffer(json.dumps(meta).encode(), dtype=np.uint8),
    )
    return job.job_id, job_dir


def _swaps(data):
    with zipfile.ZipFile(io.BytesIO(data)) as zf:
        layers = ET.fromstring(zf.read("Metadata/custom_gcode_per_layer.xml")).findall(".//layer[@type='2']")
        cfg = json.loads(zf.read("Metadata/project_settings.config"))
    return [(l.attrib["top_z"], l.attrib["extruder"]) for l in layers], cfg


def _project(sliders, **settings):
    return {
        "settings": {"layer_height": LH, "background_height": BG, "background_color": "#000000", **settings},
        "colorSliders": sliders,
        "activeFilaments": MATERIALS,
    }


def test_printer_list(client, acme):
    data = client.get("/api/outputs/3mf/printers").json()
    printers = {p["name"]: p for p in data["printers"]}
    assert data["orca_version"] == OP.ORCA_TAG
    assert printers["Acme One 0.4 nozzle"]["auto_swaps"] is False
    assert printers["Acme One 0.4 nozzle"]["bed"] == [0, 0, 200, 200]


def test_export_needs_a_printer(client, acme):
    job_id, _ = _job()
    assert client.post(f"/api/outputs/3mf/{job_id}", json={}).status_code == 422


def test_export_from_the_saved_solution(client, acme):
    job_id, _ = _job(stack=(0, 0, 1, 1, 2, 2), base_index=0)
    r = client.post(f"/api/outputs/3mf/{job_id}", json={"printer": "Acme One 0.4 nozzle", "name": "My Cat"})
    assert r.status_code == 200, r.text
    assert 'filename="My-Cat_model.3mf"' in r.headers["content-disposition"]
    swaps, cfg = _swaps(r.content)
    # stack[2] is layer 4, top at 0.24 + 3 * 0.08
    assert swaps == [("0.4800", "2"), ("0.6400", "3")]
    assert cfg["filament_colour"] == ["#000000", "#FF0000", "#FFFFFF"]
    assert cfg["filament_type"] == ["PLA", "PETG", "PLA"]
    assert cfg["manual_filament_change"] == "1"


def test_export_follows_the_edited_colour_layers_and_base(client, acme):
    job_id, _ = _job(stack=(0, 0, 1, 1, 2, 2))
    sliders = [
        {"layer": 3, "filament_uuid": "u-white", "td": 3.0, "enabled": True},
        {"layer": 6, "filament_uuid": "u-red", "td": 2.0, "enabled": True},
    ]
    body = {
        "printer": "Acme One 0.4 nozzle", "manual_swaps": False,
        "project": _project(sliders), "base": {"filament_uuid": "u-black", "color": "#000000"},
    }
    r = client.post(f"/api/outputs/3mf/{job_id}", json=body)
    assert r.status_code == 200, r.text
    swaps, cfg = _swaps(r.content)
    # black base, white from layer #2, red from layer #5 (stack index 3)
    assert cfg["filament_colour"] == ["#000000", "#FFFFFF", "#FF0000"]
    assert swaps == [("0.3200", "2"), ("0.5600", "3")]
    assert "manual_filament_change" not in cfg  # the user's choice beats the printer default


def test_one_filament_with_two_tds_is_no_swap(client, acme):
    job_id, _ = _job(stack=(0, 0, 1, 1, 2, 2))
    sliders = [
        {"layer": 2, "filament_uuid": "u-white", "td": 3.0, "enabled": True},
        {"layer": 4, "filament_uuid": "u-white", "td": 1.5, "enabled": True},
        {"layer": 6, "filament_uuid": "u-red", "td": 2.0, "enabled": True},
    ]
    body = {"printer": "Acme One 0.4 nozzle", "project": _project(sliders), "base": {"filament_uuid": "u-black"}}
    swaps, cfg = _swaps(client.post(f"/api/outputs/3mf/{job_id}", json=body).content)
    assert cfg["filament_colour"] == ["#000000", "#FFFFFF", "#FF0000"]
    assert swaps == [("0.3200", "2"), ("0.6400", "3")]


def test_model_too_big_for_the_printer(client, acme):
    job_id, _ = _job(size_mm=260)
    r = client.post(f"/api/outputs/3mf/{job_id}", json={"printer": "Acme One 0.4 nozzle"})
    assert r.status_code == 422
    assert "bed" in r.json()["detail"]


def test_unknown_printer_and_job(client, acme):
    job_id, _ = _job()
    assert client.post(f"/api/outputs/3mf/{job_id}", json={"printer": "Nope"}).status_code == 422
    assert client.post("/api/outputs/3mf/no-such-job", json={"printer": "Acme One 0.4 nozzle"}).status_code == 404


def test_profile_download_failure_is_reported(client, acme, monkeypatch):
    job_id, _ = _job()

    def offline(*a, **k):
        raise OP.ProfileDownloadError("Couldn't download the OrcaSlicer profile; needs internet")

    monkeypatch.setattr(OP.OrcaProfiles, "resolve", offline)
    r = client.post(f"/api/outputs/3mf/{job_id}", json={"printer": "Acme One 0.4 nozzle"})
    assert r.status_code == 503
    assert "internet" in r.json()["detail"]


def test_export_leaves_no_temp_files(client, acme, tmp_path, monkeypatch):
    job_id, _ = _job()
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    monkeypatch.setattr(tempfile, "tempdir", str(scratch))
    assert client.post(f"/api/outputs/3mf/{job_id}", json={"printer": "Acme One 0.4 nozzle"}).status_code == 200
    assert client.post(f"/api/outputs/3mf/{job_id}", json={"printer": "Nope"}).status_code == 422
    assert os.listdir(scratch) == []


def test_zip_has_a_3mf_only_once_a_printer_was_picked(client, acme):
    job_id, _ = _job()
    sliders = [{"layer": 6, "filament_uuid": "u-red", "td": 2.0, "enabled": True}]
    r = client.post(f"/api/outputs/export/{job_id}", json={"name": "Cat", "project": _project(sliders)})
    assert r.status_code == 200
    assert not any(n.endswith(".3mf") for n in zipfile.ZipFile(io.BytesIO(r.content)).namelist())

    project = _project(sliders, orca_printer="Acme One 0.4 nozzle", orca_manual_swaps=True)
    r = client.post(f"/api/outputs/export/{job_id}", json={"name": "Cat", "project": project, "base": {"filament_uuid": "u-black"}})
    with zipfile.ZipFile(io.BytesIO(r.content)) as zf:
        assert "Cat_model.3mf" in zf.namelist()
        assert not any("warning" in n for n in zf.namelist())  # nozzle fits: nothing to warn about
        swaps, cfg = _swaps(zf.read("Cat_model.3mf"))
    assert cfg["filament_colour"] == ["#000000", "#FF0000"]
    assert swaps == [("0.3200", "2")]


def test_zip_still_exports_when_the_3mf_fails(client, acme, monkeypatch):
    job_id, _ = _job()

    def offline(*a, **k):
        raise OP.ProfileDownloadError("offline")

    monkeypatch.setattr(OP.OrcaProfiles, "resolve", offline)
    project = _project([{"layer": 6, "filament_uuid": "u-red", "enabled": True}], orca_printer="Acme One 0.4 nozzle")
    r = client.post(f"/api/outputs/export/{job_id}", json={"name": "Cat", "project": project})
    assert r.status_code == 200
    names = zipfile.ZipFile(io.BytesIO(r.content)).namelist()
    assert "Cat_model.stl" in names and not any(n.endswith(".3mf") for n in names)


def test_settings_accept_the_printer_choice():
    from autoforge.webui.models import OptimizationSettings

    s = OptimizationSettings.model_validate({"orca_printer": "Acme One 0.4 nozzle", "orca_manual_swaps": True})
    assert s.orca_printer == "Acme One 0.4 nozzle" and s.orca_manual_swaps is True
    assert OptimizationSettings().orca_manual_swaps is None


def test_solution_keeps_filament_types_for_the_project(tmp_path):
    """A rebuilt or pruned result must still know which filament is PETG."""
    from autoforge.webui.helpers.result_store import solution_from_sliders

    sol = solution_from_sliders(
        np.array([[1, 2], [2, 3]]),
        [{"layer": 1, "filament_uuid": "u-black", "enabled": True}, {"layer": 3, "filament_uuid": "u-red", "enabled": True}],
        MATERIALS,
        {"layer_height": LH, "background_height": BG, "max_layers": 3},
    )
    types = {m["uuid"]: m["filament_type"] for m in sol["meta"]["materials"]}
    assert types["u-red"] == "PETG"


def test_swaps_use_the_runs_layer_height_not_edited_settings(client, acme):
    """The STL was built with the run's layers; a layer height changed in
    the settings afterwards (no new run yet) must not move the swaps."""
    job_id, _ = _job(stack=(0, 0, 1, 1, 2, 2))
    sliders = [
        {"layer": 2, "filament_uuid": "u-black", "enabled": True},
        {"layer": 6, "filament_uuid": "u-red", "enabled": True},
    ]
    project = _project(sliders)
    project["settings"].update(layer_height=0.04, background_height=0.4)
    body = {"printer": "Acme One 0.4 nozzle", "project": project, "base": {"filament_uuid": "u-black"}}
    swaps, cfg = _swaps(client.post(f"/api/outputs/3mf/{job_id}", json=body).content)
    assert swaps == [("0.4800", "2")]
    assert cfg["layer_height"] == "0.08" and cfg["initial_layer_print_height"] == "0.24"


# ------------------------------------------------- background export + progress

def _wait(client, task_id, timeout=30):
    import time

    seen = []
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        status = client.get(f"/api/outputs/tasks/{task_id}").json()
        seen.append(status["progress"])
        if status["done"]:
            return status, seen
        time.sleep(0.02)
    raise AssertionError("export did not finish")


def test_background_export_reports_progress_and_hands_the_file_out_once(client, acme):
    job_id, _ = _job()
    r = client.post(f"/api/outputs/3mf/{job_id}/start", json={"printer": "Acme One 0.4 nozzle", "name": "Cat"})
    assert r.status_code == 200
    task_id = r.json()["task_id"]
    status, seen = _wait(client, task_id)
    assert status["error"] is None and status["progress"] == 1.0
    assert seen == sorted(seen)  # never goes backwards
    f = client.get(f"/api/outputs/tasks/{task_id}/file")
    assert f.status_code == 200
    assert 'filename="Cat_model.3mf"' in f.headers["content-disposition"]
    swaps, _ = _swaps(f.content)
    assert swaps == [("0.4800", "2"), ("0.6400", "3")]
    assert client.get(f"/api/outputs/tasks/{task_id}/file").status_code == 404


def test_background_export_failure_carries_its_status(client, acme):
    job_id, _ = _job(size_mm=260)
    task_id = client.post(f"/api/outputs/3mf/{job_id}/start", json={"printer": "Acme One 0.4 nozzle"}).json()["task_id"]
    status, _ = _wait(client, task_id)
    assert status["error_status"] == 422 and "bed" in status["error"]
    assert client.get(f"/api/outputs/tasks/{task_id}/file").status_code == 404
    assert client.post(f"/api/outputs/3mf/{job_id}/start", json={}).status_code == 422
    assert client.get("/api/outputs/tasks/nope").status_code == 404


def test_cancelled_export_leaves_no_file(client, acme, tmp_path, monkeypatch):
    import threading

    from autoforge.Helper import Export3MF as E

    job_id, _ = _job()
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    monkeypatch.setattr(tempfile, "tempdir", str(scratch))
    gate = threading.Event()
    original = E.read_stl

    def slow_read(src):
        gate.wait(10)
        return original(src)

    monkeypatch.setattr(E, "read_stl", slow_read)
    task_id = client.post(f"/api/outputs/3mf/{job_id}/start", json={"printer": "Acme One 0.4 nozzle"}).json()["task_id"]
    assert client.delete(f"/api/outputs/tasks/{task_id}").status_code == 200
    gate.set()
    status, _ = _wait(client, task_id)
    assert status["cancelled"]
    assert client.get(f"/api/outputs/tasks/{task_id}/file").status_code == 404
    assert os.listdir(scratch) == []


def test_prepare_loads_the_printer_profiles_ahead(client, acme):
    import time

    assert client.post("/api/outputs/3mf/printers/prepare", json={"printer": "Nope"}).status_code == 404
    assert client.post("/api/outputs/3mf/printers/prepare", json={"printer": "Acme One 0.4 nozzle"}).status_code == 200
    profiles = OP.shared_profiles()
    end = time.monotonic() + 10
    wanted = [("Acme", "machine", "Acme One 0.4 nozzle"), ("Acme", "filament", "Acme PETG @Acme One")]
    while not all(k in profiles._resolved for k in wanted) and time.monotonic() < end:
        time.sleep(0.02)
    assert all(k in profiles._resolved for k in wanted)


# --------------------------------------------------------------------- nozzle

def test_export_switches_to_the_runs_nozzle_preset(client, acme):
    job_id, _ = _job(nozzle=0.6)
    task_id = client.post(f"/api/outputs/3mf/{job_id}/start", json={"printer": "Acme One 0.4 nozzle"}).json()["task_id"]
    status, _ = _wait(client, task_id)
    assert status["printer"] == "Acme One 0.6 nozzle"
    assert status["warnings"] == [] and "0.6 mm" in status["notes"][0]
    _, cfg = _swaps(client.get(f"/api/outputs/tasks/{task_id}/file").content)
    assert cfg["printer_settings_id"] == "Acme One 0.6 nozzle"


def test_the_runs_nozzle_beats_the_current_setting(client, acme):
    """The heightmap was made for the run's nozzle; a nozzle changed in the
    settings afterwards (no new run yet) doesn't count."""
    job_id, _ = _job(nozzle=0.6)
    body = {"printer": "Acme One 0.4 nozzle", "project": _project([], nozzle_diameter=0.4)}
    r = client.post(f"/api/outputs/3mf/{job_id}", json=body)
    assert json.loads(urllib.parse.unquote(r.headers["x-autoforge-export-info"]))["printer"] == "Acme One 0.6 nozzle"


def test_settings_nozzle_counts_when_the_run_has_none(client, acme):
    job_id, _ = _job()
    body = {"printer": "Acme One 0.4 nozzle", "project": _project([], nozzle_diameter=0.6)}
    r = client.post(f"/api/outputs/3mf/{job_id}", json=body)
    assert json.loads(urllib.parse.unquote(r.headers["x-autoforge-export-info"]))["printer"] == "Acme One 0.6 nozzle"


def test_missing_nozzle_preset_warns_but_exports(client, acme):
    job_id, _ = _job(nozzle=0.8)
    task_id = client.post(f"/api/outputs/3mf/{job_id}/start", json={"printer": "Acme One 0.4 nozzle"}).json()["task_id"]
    status, _ = _wait(client, task_id)
    assert status["error"] is None and status["printer"] == "Acme One 0.4 nozzle"
    assert "no 0.8 mm nozzle preset" in status["warnings"][0]
    assert client.get(f"/api/outputs/tasks/{task_id}/file").status_code == 200

    # the export bundle carries the warning next to the project
    project = _project([{"layer": 6, "filament_uuid": "u-red", "enabled": True}], orca_printer="Acme One 0.4 nozzle")
    names = zipfile.ZipFile(io.BytesIO(client.post(f"/api/outputs/export/{job_id}", json={"name": "Cat", "project": project}).content))
    assert "Cat_model.3mf" in names.namelist()
    assert "no 0.8 mm nozzle preset" in names.read("Cat_slicer_project_warning.txt").decode()


def test_prepare_loads_the_preset_for_the_nozzle(client, acme):
    import time

    r = client.post("/api/outputs/3mf/printers/prepare", json={"printer": "Acme One 0.4 nozzle", "nozzle": 0.6})
    assert r.json()["printer"] == "Acme One 0.6 nozzle"
    profiles = OP.shared_profiles()
    end = time.monotonic() + 10
    while ("Acme", "machine", "Acme One 0.6 nozzle") not in profiles._resolved and time.monotonic() < end:
        time.sleep(0.02)
    assert ("Acme", "machine", "Acme One 0.6 nozzle") in profiles._resolved


# ---------------------------------------------------------- AutoForge print settings

def test_print_settings_on_by_default_and_can_be_turned_off(client, acme):
    job_id, _ = _job()
    _, on = _swaps(client.post(f"/api/outputs/3mf/{job_id}", json={"printer": "Acme One 0.4 nozzle"}).content)
    _, off = _swaps(client.post(f"/api/outputs/3mf/{job_id}", json={"printer": "Acme One 0.4 nozzle", "autoforge_settings": False}).content)
    assert on["wall_loops"] == "1" and on["top_surface_pattern"] == "monotonic"
    assert "wall_loops" not in off and off["outer_wall_speed"] == ["200"]
    data = client.get("/api/outputs/3mf/printers").json()
    assert any("wall" in line for line in data["autoforge_settings"])


def test_zip_follows_the_remembered_print_settings_choice(client, acme):
    job_id, _ = _job()
    sliders = [{"layer": 6, "filament_uuid": "u-red", "enabled": True}]
    project = _project(sliders, orca_printer="Acme One 0.4 nozzle", orca_autoforge_settings=False)
    zf = zipfile.ZipFile(io.BytesIO(client.post(f"/api/outputs/export/{job_id}", json={"name": "Cat", "project": project}).content))
    _, cfg = _swaps(zf.read("Cat_model.3mf"))
    assert "wall_loops" not in cfg
