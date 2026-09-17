import time
import pytest
from fastapi.testclient import TestClient
from autoforge.webui import create_app


@pytest.fixture
def client():
    app = create_app()
    return TestClient(app)


def test_save_and_list_snapshots(client):
    snap = {"timestamp": time.time(), "label": "test-snapshot"}
    resp = client.post("/api/state/snapshot", json=snap)
    assert resp.status_code == 200
    assert "snapshot_id" in resp.json()

    history = client.get("/api/state/history")
    assert history.status_code == 200
    assert len(history.json()) >= 1


def test_restore_snapshot(client):
    resp = client.get("/api/state/history")
    if resp.json():
        snap = resp.json()[0]
        restore = client.post("/api/state/restore", json={"timestamp": snap["timestamp"]})
        assert restore.status_code == 200
