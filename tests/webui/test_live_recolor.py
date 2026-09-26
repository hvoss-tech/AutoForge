"""Live 3D recoloring on slider edits.

A slider edit changes colors, never heights, so the render endpoint can hand
the client the new vertex colors of the mesh it already shows instead of a
rebuilt ~25MB PLY. These tests pin the contract that makes that safe: the
colors are in exactly the mesh's vertex order, and the PLY on disk still
catches up (in the background) for reloads.
"""

import argparse
import base64
import os
import time

import numpy as np
import pytest
import torch
from fastapi.testclient import TestClient

from autoforge.webui.helpers.colored_mesh import generate_colored_preview_mesh, top_vertex_pixel_indices
from autoforge.webui.server import create_app


@pytest.fixture
def client():
    return TestClient(create_app())


def _alpha(h, w, seed):
    rng = np.random.default_rng(seed)
    alpha = np.full((h, w, 1), 255, np.uint8)
    alpha[rng.random((h, w)) < 0.15] = 0  # scattered holes
    alpha[:, :3] = 0  # a fully transparent border
    alpha[h // 2:, w // 2:] = 0  # and a big cut-out
    return alpha


@pytest.mark.parametrize("alpha_kind", ["none", "holes"])
@pytest.mark.parametrize("max_dim", [0, 13])
def test_top_vertex_pixels_are_the_meshs_top_vertices_in_order(alpha_kind, max_dim):
    h, w = 31, 47
    rng = np.random.default_rng(3)
    heights = rng.uniform(0, 2, (h, w)).astype(np.float32)
    colors = rng.integers(0, 256, (h, w, 3), dtype=np.uint8)
    alpha = None if alpha_kind == "none" else _alpha(h, w, 5)

    mesh = generate_colored_preview_mesh(heights, colors, 0.24, 100.0, alpha_mask=alpha, max_grid_dim=max_dim)
    idx = top_vertex_pixel_indices(h, w, alpha, max_grid_dim=max_dim)

    n = len(mesh.vertices)
    assert n == 2 * len(idx), "top and bottom halves must be the same size"
    vertex_colors = np.asarray(mesh.visual.vertex_colors)[:, :3]
    np.testing.assert_array_equal(vertex_colors[: n // 2], colors.reshape(-1, 3)[idx])
    # Positions agree too: each top vertex sits over its pixel.
    rows, cols = np.divmod(idx, w)
    scale = 100.0 / max(w - 1, h - 1)
    np.testing.assert_allclose(mesh.vertices[: n // 2, 0], cols * scale, rtol=0, atol=1e-4)
    np.testing.assert_allclose(mesh.vertices[: n // 2, 1], (h - 1 - rows) * scale, rtol=0, atol=1e-4)
    assert (mesh.vertices[n // 2:, 2] == 0).all()


class _FakeOptimizer:
    max_layers = 10

    def __init__(self, heights):
        self._h = torch.tensor(heights)

    def get_discretized_solution(self, best=True):
        return torch.zeros(10, dtype=torch.long), self._h


def _result(h=24, w=40, alpha=None):
    heights = np.tile(np.arange(w) % 11, (h, 1)).astype(np.int64)
    return {
        "optimizer": _FakeOptimizer(heights),
        "args": argparse.Namespace(layer_height=0.04, background_height=0.24, stl_output_size=50),
        "background": torch.zeros(3),
        "alpha": alpha,
    }


def _completed_job_with(result, job_id="recolor-job"):
    from autoforge.webui.services.optimization_service import get_optimization_service

    svc = get_optimization_service()
    svc.create_job({"iterations": 1}, job_id=job_id)
    svc.update_status(job_id, "completed")
    svc.set_pipeline_result(job_id, result)
    return job_id


SLIDERS = [
    {"layer": 4, "enabled": True, "filament_uuid": "k", "td": 0.6},
    {"layer": 10, "enabled": True, "filament_uuid": "w", "td": 5.0},
]
FILAMENTS = [
    {"uuid": "k", "color": "#101010", "td": 0.6, "name": "K"},
    {"uuid": "w", "color": "#f0e0d0", "td": 5.0, "name": "W"},
]


def test_render_returns_the_meshs_new_vertex_colors_and_the_ply_catches_up(client):
    from autoforge.webui.api.outputs import EDITED_PLY
    from autoforge.webui.config import config
    from autoforge.webui.helpers.mesh_persist import wait_for_pending_mesh
    from autoforge.webui.helpers.slider_render import render_with_sliders

    alpha = _alpha(24, 40, 9)
    result = _result(alpha=alpha)
    job = _completed_job_with(result)
    lookup = {f["uuid"]: f for f in FILAMENTS}

    t0 = time.perf_counter()
    r = client.post("/api/preview/render-with-sliders", json={
        "job_id": job, "sliders": SLIDERS, "active_filaments": FILAMENTS, "vertex_colors": True, "render_id": "r1",
    })
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["status"] == "ok"
    rgb = np.frombuffer(base64.b64decode(body["vertex_colors"]), np.uint8).reshape(-1, 3)
    assert body["top_vertex_count"] == len(rgb)

    # The same colors a full render bakes into the mesh's top vertices.
    reference = render_with_sliders(result, SLIDERS, lookup, os.path.join(config.checkpoints_path, "ref"))
    import trimesh

    ref_mesh = trimesh.load(reference["colored_ply"], process=False)
    ref_colors = np.asarray(ref_mesh.visual.vertex_colors)[:, :3]
    assert len(ref_colors) == 2 * len(rgb)
    np.testing.assert_array_equal(rgb, ref_colors[: len(rgb)])

    # The edited PNG is written before the response; the PLY in the
    # background — and the mesh endpoint waits for it.
    job_dir = os.path.join(config.checkpoints_path, job)
    assert os.path.exists(os.path.join(job_dir, "edited_model.png"))
    ply = client.get(f"/api/outputs/colored-ply/{job}")
    assert ply.status_code == 200
    served = trimesh.load(trimesh.util.wrap_as_stream(ply.content), file_type="ply", process=False)
    np.testing.assert_array_equal(np.asarray(served.visual.vertex_colors)[:, :3], ref_colors)
    wait_for_pending_mesh(job)
    assert os.path.exists(os.path.join(job_dir, EDITED_PLY))
    assert time.perf_counter() - t0 < 10


def test_without_the_flag_the_render_is_unchanged(client):
    from autoforge.webui.api.outputs import EDITED_PLY
    from autoforge.webui.config import config

    job = _completed_job_with(_result(), job_id="legacy-job")
    r = client.post("/api/preview/render-with-sliders", json={
        "job_id": job, "sliders": SLIDERS, "active_filaments": FILAMENTS,
    })
    assert r.json() == {"status": "ok", "job_id": job, "slider_count": 2}
    # Written synchronously, as before.
    assert os.path.exists(os.path.join(config.checkpoints_path, job, EDITED_PLY))


def test_dragging_writes_only_the_last_edits_mesh(client, monkeypatch):
    """A drag sends an edit per position; the background write must not
    build a mesh for every one of them — just the one it settles on."""
    import autoforge.webui.api.preview as preview_api

    written = []
    real = preview_api.write_slider_ply

    def counting(render, output_dir, ply_name):
        written.append(render["comp_np"].copy())
        return real(render, output_dir, ply_name)

    monkeypatch.setattr(preview_api, "write_slider_ply", counting)
    job = _completed_job_with(_result(), job_id="drag-job")
    for layer in (2, 3, 4, 5, 6, 7):
        sliders = [dict(SLIDERS[0], layer=layer), SLIDERS[1]]
        r = client.post("/api/preview/render-with-sliders", json={
            "job_id": job, "sliders": sliders, "active_filaments": FILAMENTS, "vertex_colors": True,
        })
        assert r.json()["status"] == "ok"
    from autoforge.webui.helpers.mesh_persist import wait_for_pending_mesh

    wait_for_pending_mesh(job)
    assert 1 <= len(written) <= 2, len(written)


def test_render_id_rides_on_the_broadcast(client, monkeypatch):
    import autoforge.webui.api.preview as preview_api

    sent = []
    monkeypatch.setattr(preview_api, "broadcast_preview", lambda *a, **k: sent.append(k))
    job = _completed_job_with(_result(), job_id="rid-job")
    client.post("/api/preview/render-with-sliders", json={
        "job_id": job, "sliders": SLIDERS, "active_filaments": FILAMENTS, "vertex_colors": True, "render_id": "abc",
    })
    assert sent and sent[-1]["render_id"] == "abc"


def test_resetting_the_preview_drops_a_pending_mesh_write(monkeypatch):
    from autoforge.webui.helpers import mesh_persist

    ran = []
    monkeypatch.setattr(mesh_persist, "_PERSIST_QUIET_S", 0.3)
    mesh_persist.schedule_persist("__test_reset__", lambda: ran.append(1))
    mesh_persist.cancel_pending_mesh("__test_reset__")
    mesh_persist.wait_for_pending_mesh("__test_reset__", timeout=5)
    assert ran == []
