"""Export zip: project-named files, bundled project file, off the event loop.

Regression (2026-09-29): "Everything (.zip)" appeared to do nothing — the
bundle of a real run (~270 MB STL + ~110 MB PLY) was deflated at the default
level on the event loop and then streamed from a BytesIO line by line, which
took ~37 s before the browser's download could start. And every download
had a fixed name regardless of the project name.
"""
import io
import json
import os
import zipfile

import pytest
from fastapi.testclient import TestClient

from autoforge.webui.server import create_app


@pytest.fixture
def client():
    return TestClient(create_app())


def _completed_job(files):
    from autoforge.webui.config import config
    from autoforge.webui.services.optimization_service import get_optimization_service

    svc = get_optimization_service()
    job = svc.create_job({"iterations": 10, "input_image": "a.png"})
    svc.update_status(job.job_id, "completed")
    job_dir = os.path.join(config.checkpoints_dir, job.job_id)
    os.makedirs(job_dir, exist_ok=True)
    for name in files:
        with open(os.path.join(job_dir, name), "wb") as f:
            f.write(b"x" * 1000)
    return job


RESULT_FILES = ("final_model.stl", "final_model_colored.ply", "final_model.png",
                "swap_instructions.txt", "final_loss.txt", "project_file.hfp")


def test_named_export_names_the_zip_and_its_files(client):
    job = _completed_job(RESULT_FILES)
    project = {"version": 1, "name": "My Car", "colorSliders": []}
    r = client.post(f"/api/outputs/export/{job.job_id}", json={"name": "My Car", "project": project})
    assert r.status_code == 200, r.text
    assert 'filename="My-Car_project.zip"' in r.headers["content-disposition"]
    assert int(r.headers["content-length"]) == len(r.content)
    zf = zipfile.ZipFile(io.BytesIO(r.content))
    assert set(zf.namelist()) == {
        "My-Car_model.stl", "My-Car_model_colored.ply", "My-Car_result.png",
        "My-Car_swap_instructions.txt", "My-Car_loss.txt", "My-Car_hueforge.hfp",
        "My-Car_project.json",
    }
    assert json.loads(zf.read("My-Car_project.json")) == project


def test_unnamed_export_keeps_the_default_names(client):
    job = _completed_job(RESULT_FILES)
    r = client.get(f"/api/outputs/export/{job.job_id}")
    assert r.status_code == 200, r.text
    assert f'filename="{job.job_id}_export.zip"' in r.headers["content-disposition"]
    assert set(zipfile.ZipFile(io.BytesIO(r.content)).namelist()) == set(RESULT_FILES)


def test_flatforge_stls_keep_their_material_after_the_prefix(client):
    job = _completed_job(("Red_#ff0000.stl", "final_model.png"))
    r = client.post(f"/api/outputs/export/{job.job_id}", json={"name": "car"})
    assert r.status_code == 200, r.text
    assert set(zipfile.ZipFile(io.BytesIO(r.content)).namelist()) == {"car_Red_#ff0000.stl", "car_result.png"}


def test_export_leaves_no_temp_zip_behind(client, tmp_path, monkeypatch):
    import tempfile

    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))
    job = _completed_job(RESULT_FILES)
    assert client.post(f"/api/outputs/export/{job.job_id}", json={"name": "car"}).status_code == 200
    assert client.get("/api/outputs/export/not-a-job").status_code == 400
    assert os.listdir(tmp_path) == []


def test_name_cannot_escape_into_a_path():
    from autoforge.webui.api.outputs import _export_prefix

    assert _export_prefix("../../etc/x") == "etc-x"
    assert _export_prefix("  ") == ""
    assert _export_prefix("Katze ü 2") == "Katze-ü-2"
