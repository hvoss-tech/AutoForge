import pytest
from fastapi.testclient import TestClient
from autoforge.webui import create_app


@pytest.fixture
def client():
    app = create_app()
    return TestClient(app)


def test_get_settings(client):
    resp = client.get("/api/settings")
    assert resp.status_code == 200


def test_update_settings(client):
    resp = client.put("/api/settings", json={"iterations": 1000})
    assert resp.status_code == 200


def test_get_settings_schema(client):
    resp = client.get("/api/settings/schema")
    assert resp.status_code == 200


def test_project_state(client):
    resp = client.get("/api/project/state")
    assert resp.status_code == 200


def test_save_project_state(client):
    resp = client.post("/api/project/state", json={
        "colorSliders": [],
        "globalParams": {},
        "settings": {},
        "activeFilaments": [],
    })
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


def test_outputs_not_found(client):
    resp = client.get("/api/outputs/stl/nonexistent")
    assert resp.status_code == 404


def test_pruning_start_requires_a_job(client):
    resp = client.post("/api/pruning/start", json={
        "pruningMaxColors": 50,
        "pruningMaxSwaps": 50,
        "pruningMaxLayer": 50,
    })
    assert resp.status_code == 400
    assert "Run an optimization first" in resp.json()["detail"]


def test_pruning_start_rejects_unfinished_job(client):
    from autoforge.webui.services.optimization_service import get_optimization_service

    svc = get_optimization_service()
    svc.create_job({"iterations": 1}, job_id="still-running")
    svc.update_status("still-running", "running")
    resp = client.post("/api/pruning/start", json={"job_id": "still-running"})
    # The GPU-mutex busy check (any non-terminal job, including this one)
    # is now evaluated before the "target must be completed" check, so a
    # still-running target is rejected as "busy" (409) rather than
    # "not completed" (400).
    assert resp.status_code == 409


def test_pruning_of_result_without_pipeline_fails_cleanly(client):
    """A completed job from before a server restart has no in-memory result
    to prune; the prune job must fail with an explanation, not crash."""
    import time
    from autoforge.webui.services.optimization_service import get_optimization_service

    svc = get_optimization_service()
    svc.create_job({"iterations": 1}, job_id="old-result")
    svc.update_status("old-result", "completed")
    resp = client.post("/api/pruning/start", json={"job_id": "old-result"})
    assert resp.status_code == 200
    prune_id = resp.json()["job_id"]

    deadline = time.time() + 10
    while time.time() < deadline:
        status = client.get(f"/api/optimize/status/{prune_id}").json()
        if status["status"] == "failed":
            break
        time.sleep(0.05)
    assert status["status"] == "failed"
    assert "run optimization first" in status["error"].lower()
    # Pruning jobs are bookkeeping only, never "the current job".
    assert client.get("/api/optimize/latest").json()["job_id"] == "old-result"
