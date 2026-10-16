"""Borders in the webui (helpers/border, api/outputs): a preview mesh for the
3D view and bordered STLs on export, from a finished result without running
the optimizer again."""

import base64
import io
import json
import os
import zipfile

import numpy as np
import pytest
import trimesh
from fastapi.testclient import TestClient

from autoforge.Helper.BorderMesh import load_mesh
from autoforge.Helper.OutputHelper import generate_stl
from autoforge.webui.server import create_app

BH = 0.32
SIZE = 40.0
RECT = {"shape": "rectangle", "width": 5, "corner_radius": 0, "hanger": "none", "hanger_diameter": 4}


@pytest.fixture
def client():
    return TestClient(create_app())


def _heights():
    return (np.random.default_rng(3).integers(0, 5, size=(30, 20)) * 0.08).astype(np.float32)


def _job(background_height=BH, files=("final_model.stl",)):
    from autoforge.webui.config import config
    from autoforge.webui.services.optimization_service import get_optimization_service

    svc = get_optimization_service()
    job = svc.create_job({"iterations": 10, "input_image": "a.png", "background_height": background_height, "layer_height": 0.08})
    svc.update_status(job.job_id, "completed")
    job_dir = os.path.join(config.checkpoints_dir, job.job_id)
    os.makedirs(job_dir, exist_ok=True)
    if "final_model.stl" in files:
        generate_stl(_heights(), os.path.join(job_dir, "final_model.stl"), background_height, SIZE)
    for name in files:
        if name != "final_model.stl":
            with open(os.path.join(job_dir, name), "wb") as f:
                f.write(b"x" * 100)
    return job, job_dir


def _stl(content: bytes) -> trimesh.Trimesh:
    path = os.path.join(os.path.dirname(__file__), "_tmp_border.stl")
    try:
        with open(path, "wb") as f:
            f.write(content)
        return trimesh.Trimesh(*load_mesh(path), process=False)
    finally:
        os.remove(path)


def test_preview_returns_the_ring_and_the_sizes(client):
    job, _ = _job()
    r = client.post(f"/api/outputs/border-preview/{job.job_id}", json={"border": RECT})
    assert r.status_code == 200, r.text
    data = r.json()
    positions = np.frombuffer(base64.b64decode(data["positions"]), np.float32).reshape(-1, 3)
    indices = np.frombuffer(base64.b64decode(data["indices"]), np.uint32).reshape(-1, 3)
    ring = trimesh.Trimesh(positions, indices, process=False)
    assert ring.is_watertight
    # The job's own base height, not the default.
    assert data["background_height"] == pytest.approx(BH)
    assert positions[:, 2].max() == pytest.approx(BH)
    w, h = data["model_size_mm"]
    assert data["size_mm"] == pytest.approx([w + 10, h + 10])
    # (sizes are rounded to 0.01 mm)
    assert ring.volume == pytest.approx(((w + 10) * (h + 10) - w * h) * BH, rel=1e-3)


def test_preview_without_border_is_empty(client):
    job, _ = _job()
    data = client.post(f"/api/outputs/border-preview/{job.job_id}", json={"border": {"shape": "none"}}).json()
    assert data["positions"] == "" and data["size_mm"] == data["model_size_mm"]


def test_preview_of_the_auto_preview_uses_its_mesh(client, tmp_path):
    from autoforge.webui.api.init import init_mesh_path
    from autoforge.webui.helpers.colored_mesh import generate_colored_preview_mesh

    heights = _heights()
    os.makedirs(os.path.dirname(init_mesh_path()), exist_ok=True)
    generate_colored_preview_mesh(heights, np.zeros(heights.shape + (3,), np.uint8), 0.24, SIZE).export(
        init_mesh_path(), encoding="binary"
    )
    # No init result held any more: the client's base height is used.
    r = client.post("/api/outputs/border-preview/__init__", json={"border": RECT, "background_height": 0.24})
    assert r.status_code == 200, r.text
    assert r.json()["background_height"] == pytest.approx(0.24)


def test_preview_errors(client):
    assert client.post("/api/outputs/border-preview/nope", json={"border": RECT}).status_code == 404
    assert client.post("/api/outputs/border-preview/..%2F..", json={"border": RECT}).status_code in (404, 405)
    job, _ = _job(background_height=0.0)
    r = client.post(f"/api/outputs/border-preview/{job.job_id}", json={"border": RECT})
    assert r.status_code == 422 and "base height" in r.json()["detail"]


def test_stl_download_carries_the_border(client):
    job, job_dir = _job()
    plain = client.post(f"/api/outputs/stl/{job.job_id}", json={"border": {"shape": "none"}})
    with open(os.path.join(job_dir, "final_model.stl"), "rb") as f:
        assert plain.content == f.read()
    bordered = client.post(f"/api/outputs/stl/{job.job_id}", json={"border": RECT})
    assert bordered.status_code == 200
    mesh = _stl(bordered.content)
    original = _stl(plain.content)
    assert mesh.is_watertight
    assert (mesh.bounds[1] - mesh.bounds[0])[:2] == pytest.approx((original.bounds[1] - original.bounds[0])[:2] + 10)
    # The download doesn't leave the job's own STL changed.
    with open(os.path.join(job_dir, "final_model.stl"), "rb") as f:
        assert f.read() == plain.content


def test_export_zip_takes_the_border_from_the_project_settings(client):
    job, _ = _job(files=("final_model.stl", "final_model.png"))
    project = {"settings": {"border": RECT}}
    r = client.post(f"/api/outputs/export/{job.job_id}", json={"name": "cat", "project": project})
    assert r.status_code == 200, r.text
    with zipfile.ZipFile(io.BytesIO(r.content)) as zf:
        mesh = _stl(zf.read("cat_model.stl"))
        assert json.loads(zf.read("cat_project.json"))["settings"]["border"]["shape"] == "rectangle"
    assert mesh.is_watertight
    assert mesh.bounds[0, 0] == pytest.approx(-5)


def test_export_zip_without_border_keeps_the_stl(client):
    job, job_dir = _job()
    r = client.post(f"/api/outputs/export/{job.job_id}", json={"project": {"settings": {}}})
    with zipfile.ZipFile(io.BytesIO(r.content)) as zf, open(os.path.join(job_dir, "final_model.stl"), "rb") as f:
        assert zf.read("final_model.stl") == f.read()


def test_flatforge_base_layer_download_carries_the_border(client):
    from autoforge.webui.helpers.colored_mesh import generate_colored_preview_mesh

    job, job_dir = _job(files=("Red_ff0000.stl",))
    heights = _heights()
    generate_colored_preview_mesh(heights, np.zeros(heights.shape + (3,), np.uint8), BH, SIZE).export(
        os.path.join(job_dir, "final_model_colored.ply"), encoding="binary"
    )
    base = trimesh.creation.box(bounds=[[0, 0, 0], [SIZE * 19 / 29, SIZE, BH]])
    base.export(os.path.join(job_dir, "Background_000000.stl"))
    r = client.post(f"/api/outputs/file/{job.job_id}/Background_000000.stl", json={"border": RECT})
    assert r.status_code == 200, r.text
    mesh = _stl(r.content)
    assert mesh.is_watertight and mesh.bounds[0, 0] == pytest.approx(-5) and mesh.bounds[1, 2] == pytest.approx(BH)
    # Other materials' STLs are served as they are.
    other = client.post(f"/api/outputs/file/{job.job_id}/Red_ff0000.stl", json={"border": RECT})
    assert other.content == b"x" * 100


def test_border_setting_survives_the_project_state_round_trip(client):
    state = client.get("/api/project/state").json()
    state["settings"]["border"] = {**RECT, "shape": "oval"}
    assert client.post("/api/project/state", json=state).status_code == 200
    assert client.get("/api/project/state").json()["settings"]["border"]["shape"] == "oval"


def test_an_odd_border_value_never_blocks_the_settings(client):
    from autoforge.webui.models import OptimizationSettings

    border = OptimizationSettings(border={"shape": "triangle", "width": 0, "hangerDiameter": "big"}).border
    assert (border.shape, border.width, border.hanger_diameter) == ("none", 1.0, 4.0)


def test_without_manifold3d_only_the_border_fails(client, monkeypatch):
    """An install that predates the dependency: the app (and its settings)
    must keep working; the border answers with what to install."""
    from autoforge.Helper import BorderMesh

    monkeypatch.setattr(BorderMesh, "mf", None)
    job, _ = _job()
    r = client.post(f"/api/outputs/border-preview/{job.job_id}", json={"border": RECT})
    assert r.status_code == 422 and "manifold3d" in r.json()["detail"]
    r = client.post(f"/api/outputs/stl/{job.job_id}", json={"border": RECT})
    assert r.status_code == 422 and "manifold3d" in r.json()["detail"]
    assert client.get("/api/project/state").status_code == 200
