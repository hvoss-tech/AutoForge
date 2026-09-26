"""Regression: changing the base/background filament must change the
rendered composite image, the same way editing any other color slider does.

Previously `compute_slider_render`/`render_with_sliders` always used
`pipeline_result["background"]`, frozen at whatever the pipeline resolved at
run time. The frontend's base-color picker (FileMenu/SettingsModal) only
ever updated `resolvedBase`/`settings.background_color` client-side, so the
composited preview kept showing the old background forever.
"""

import argparse

import numpy as np
import torch
from fastapi.testclient import TestClient

from autoforge.webui.server import create_app


class _FakeOptimizer:
    max_layers = 10

    def __init__(self, heights):
        self._h = torch.tensor(heights)

    def get_discretized_solution(self, best=True):
        return torch.zeros(10, dtype=torch.long), self._h


def _result(h=12, w=16):
    # No printed layers at all -> every pixel is pure background, so the
    # composite is maximally sensitive to the background color.
    heights = np.zeros((h, w), dtype=np.int64)
    return {
        "optimizer": _FakeOptimizer(heights),
        "args": argparse.Namespace(layer_height=0.04, background_height=0.24, stl_output_size=50),
        "background": torch.zeros(3),  # pipeline resolved black
        "alpha": None,
    }


def _completed_job_with(result, job_id):
    from autoforge.webui.services.optimization_service import get_optimization_service

    svc = get_optimization_service()
    svc.create_job({"iterations": 1}, job_id=job_id)
    svc.update_status(job_id, "completed")
    svc.set_pipeline_result(job_id, result)
    return job_id


SLIDERS = [{"layer": 1, "enabled": True, "filament_uuid": "w", "td": 5.0}]
FILAMENTS = [{"uuid": "w", "color": "#f0e0d0", "td": 5.0, "name": "W"}]


def test_render_with_sliders_uses_pipeline_background_by_default():
    from autoforge.webui.helpers.slider_render import compute_slider_render

    result = _result()
    lookup = {f["uuid"]: f for f in FILAMENTS}
    render = compute_slider_render(result, SLIDERS, lookup)
    assert render is not None
    # No override given -> falls back to the pipeline's own (black) background.
    assert render["comp_np"].max() < 5


def test_render_with_sliders_honors_background_rgb_override():
    from autoforge.webui.helpers.slider_render import compute_slider_render

    result = _result()
    lookup = {f["uuid"]: f for f in FILAMENTS}
    render = compute_slider_render(result, SLIDERS, lookup, background_rgb=(1.0, 0.0, 0.0))
    assert render is not None
    # Overridden to red -> the composite should be red-dominated, not black.
    mean_rgb = render["comp_np"].reshape(-1, 3).mean(axis=0)
    assert mean_rgb[0] > mean_rgb[1] + 50
    assert mean_rgb[0] > mean_rgb[2] + 50


def test_render_endpoint_changes_image_when_background_color_changes():
    client = TestClient(create_app())
    job = _completed_job_with(_result(), job_id="bg-color-job")

    r1 = client.post("/api/preview/render-with-sliders", json={
        "job_id": job, "sliders": SLIDERS, "active_filaments": FILAMENTS,
        "background_color": "#000000", "vertex_colors": True, "render_id": "r1",
    })
    assert r1.status_code == 200, r1.text
    body1 = r1.json()
    assert body1["status"] == "ok"

    r2 = client.post("/api/preview/render-with-sliders", json={
        "job_id": job, "sliders": SLIDERS, "active_filaments": FILAMENTS,
        "background_color": "#ff0000", "vertex_colors": True, "render_id": "r2",
    })
    assert r2.status_code == 200, r2.text
    body2 = r2.json()
    assert body2["status"] == "ok"

    # Same slider stack, only the base color changed -> the two renders'
    # vertex colors must differ.
    assert body1["vertex_colors"] != body2["vertex_colors"]
