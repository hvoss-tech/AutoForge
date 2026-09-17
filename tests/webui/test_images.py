import io
import pytest
from fastapi.testclient import TestClient
from autoforge.webui import create_app


@pytest.fixture
def client():
    app = create_app()
    return TestClient(app)


def test_upload_image(client):
    import cv2
    import numpy as np

    _ok, buf = cv2.imencode(".png", np.zeros((8, 8, 3), dtype=np.uint8))
    resp = client.post("/api/images/upload", files={"file": ("test.png", io.BytesIO(buf.tobytes()), "image/png")})
    assert resp.status_code == 200
    result = resp.json()
    assert result["filename"].endswith(".png")
    assert result["url"] == f"/uploads/{result['filename']}"


def test_upload_rejects_undecodable_file(client):
    """A corrupt or non-image file used to be accepted and only fail later."""
    for data in (b"fake-image-data", b""):
        resp = client.post("/api/images/upload", files={"file": ("test.png", io.BytesIO(data), "image/png")})
        assert resp.status_code == 400
        assert "image" in resp.json()["detail"]


def test_get_image_not_found(client):
    resp = client.get("/api/images/nonexistent.png")
    assert resp.status_code == 404


def test_init_run_requires_settings_body(client):
    assert client.post("/api/init/run").status_code == 422


def test_init_run_rejects_missing_prerequisites_without_getting_stuck(client):
    """Each rejection must leave the preview idle, not stuck "initializing"
    (which would make every later attempt fail with "Already initializing")."""
    from autoforge.webui.api.init import reset_init_state

    reset_init_state()
    resp = client.post("/api/init/run", json={"input_image": "cat.png"})
    assert resp.status_code == 400
    assert "filament" in resp.json()["detail"].lower()
    assert client.get("/api/init/status").json()["status"] == "idle"

    client.post("/api/filaments/active", json={"uuid": "f1", "name": "Red", "color": "#ff0000", "td": 1.0})
    resp = client.post("/api/init/run", json={})
    assert resp.status_code == 400
    assert "image" in resp.json()["detail"].lower()

    resp = client.post("/api/init/run", json={"input_image": "does-not-exist.png"})
    assert resp.status_code == 404
    assert client.get("/api/init/status").json()["status"] == "idle"


def test_init_status(client):
    resp = client.get("/api/init/status")
    assert resp.status_code == 200
