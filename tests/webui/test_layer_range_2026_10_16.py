"""GET /api/sliders/range: a result's own layer range for the color core.

Bug (2026-10-16): after a page reload the color core kept its placeholder
range (the max_layers default, 75) instead of the result's top layer, so it
ran far past the layers the model actually has.
"""

import json
import os

import numpy as np
import pytest
from fastapi.testclient import TestClient

from autoforge.webui.server import create_app


@pytest.fixture
def client():
    return TestClient(create_app())


def _job_with_solution(heights):
    from autoforge.webui.config import config
    from autoforge.webui.services.optimization_service import get_optimization_service

    svc = get_optimization_service()
    # max_layers far above the result's own layers, like the default 75.
    job = svc.create_job({"iterations": 10, "input_image": "a.png", "max_layers": 75})
    svc.update_status(job.job_id, "completed")
    job_dir = os.path.join(config.checkpoints_dir, job.job_id)
    os.makedirs(job_dir, exist_ok=True)
    np.savez_compressed(
        os.path.join(job_dir, "solution.npz"),
        heights=np.asarray(heights, dtype=np.uint16),
        stack=np.zeros(30, dtype=np.int32),
        meta=np.frombuffer(json.dumps({"args": {"max_layers": 75}}).encode(), dtype=np.uint8),
    )
    return job


def test_range_comes_from_the_saved_result_not_the_setting(client):
    job = _job_with_solution([[2, 30], [7, 12]])
    r = client.get(f"/api/sliders/range?job_id={job.job_id}")
    assert r.status_code == 200
    assert r.json() == {"min_layer": 2, "max_layer": 30}


def test_range_without_a_result_is_404(client):
    assert client.get("/api/sliders/range?job_id=nope").status_code == 404
    assert client.get("/api/sliders/range?job_id=__init__").status_code == 404
    assert client.get("/api/sliders/range?job_id=../..").status_code == 404
