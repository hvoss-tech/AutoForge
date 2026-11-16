"""File › Slicer project (.3mf) in the webui: printer list, export endpoint
and the export bundle."""

import io
import json
import os
import sys
import tempfile
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


def _job(stack=(0, 0, 1, 1, 2, 2), base_index=0, size_mm=30):
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
        "version": 1, "materials": MATERIALS, "args": {"layer_height": LH, "background_height": BG},
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
        layers = ET.fromstring(zf.read("Metadata/custom_gcode_per_layer.xml")).findall(".//layer")
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
