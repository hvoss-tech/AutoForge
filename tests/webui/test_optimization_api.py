import time
import pytest
from fastapi.testclient import TestClient
from autoforge.webui import create_app


@pytest.fixture
def client():
    app = create_app()
    c = TestClient(app)
    # active.json persists in the session-wide temp library between tests.
    c.put("/api/filaments/active", json=[])
    return c


def test_start_rejected_without_active_filaments(client):
    resp = client.post("/api/optimize/start", json={"iterations": 100, "input_image": "x.png"})
    assert resp.status_code == 400
    assert "filament" in resp.json()["detail"].lower()


def test_start_rejected_without_input_image(client):
    client.post("/api/filaments/active", json={"uuid": "f1", "name": "Red", "color": "#ff0000", "td": 1.0})
    resp = client.post("/api/optimize/start", json={"iterations": 100})
    assert resp.status_code == 400
    assert "image" in resp.json()["detail"].lower()


def test_start_with_missing_image_file_fails_the_job_and_records_it(client):
    client.post("/api/filaments/active", json={"uuid": "f1", "name": "Red", "color": "#ff0000", "td": 1.0})
    resp = client.post("/api/optimize/start", json={"iterations": 100, "input_image": "missing.png"})
    assert resp.status_code == 200
    job_id = resp.json()["job_id"]

    deadline = time.time() + 10
    status = None
    while time.time() < deadline:
        status = client.get(f"/api/optimize/status/{job_id}").json()
        if status["status"] == "failed":
            break
        time.sleep(0.05)
    assert status["status"] == "failed"
    assert "Input image not found" in status["error"]
    assert job_id in [j["job_id"] for j in client.get("/api/optimize/history").json()]
    assert client.get(f"/api/optimize/result/{job_id}").json()["status"] == "failed"
    # Control on a finished job leaves it alone.
    assert client.post(f"/api/optimize/cancel/{job_id}").json() == {"status": "failed"}


def test_status_of_unknown_job_404(client):
    assert client.get("/api/optimize/status/nope").status_code == 404
    assert client.post("/api/optimize/pause/nope").status_code == 404


def test_sliders_endpoint(client):
    resp = client.get("/api/sliders/from-optimizer")
    assert resp.status_code == 200
    body = resp.json()
    assert set(body.keys()) == {"sliders", "min_layer", "max_layer"}
    assert isinstance(body["sliders"], list)
    assert isinstance(body["min_layer"], int)
    assert isinstance(body["max_layer"], int)


def test_preview_render_requires_a_renderable_target(client):
    from autoforge.webui.api.init import reset_init_state

    reset_init_state()
    assert client.post("/api/preview/render-with-sliders", json={"sliders": []}).status_code == 400
    resp = client.post("/api/preview/render-with-sliders", json={"sliders": [], "job_id": "__init__"})
    assert resp.status_code == 400
    assert "auto-preview" in resp.json()["detail"]
