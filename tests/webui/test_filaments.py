import pytest
from fastapi.testclient import TestClient
from autoforge.webui import create_app
from autoforge.webui.config import config
from autoforge.webui.services.filament_service import reset_service


@pytest.fixture
def client(tmp_path):
    reset_service()
    orig = config.library_dir
    config.library_dir = str(tmp_path)
    try:
        app = create_app()
        yield TestClient(app)
    finally:
        config.library_dir = orig
        reset_service()


def test_list_filaments_empty(client):
    resp = client.get("/api/filaments")
    assert resp.status_code == 200
    assert resp.json() == []


@pytest.mark.parametrize("type_key", ["filament_type", "filamentType"])
def test_create_and_list_filament(client, type_key):
    """Accepts either casing on input; always answers in snake_case, which is
    what the frontend's Filament type reads."""
    payload = {
        "brand": "BambuLab",
        "name": "Basic Green",
        "color": "#00ff00",
        "td": 2.0,
        "uuid": "test-001",
        type_key: "PLA",
        "source": "user",
    }
    resp = client.post("/api/filaments", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert data["name"] == "Basic Green"
    assert data["brand"] == "BambuLab"
    assert data["filament_type"] == "PLA"
    assert "filamentType" not in data

    resp2 = client.get("/api/filaments")
    assert [f["uuid"] for f in resp2.json()] == ["test-001"]


def test_list_types(client):
    resp = client.get("/api/filaments/types")
    assert resp.status_code == 200


def test_list_brands(client):
    resp = client.get("/api/filaments/brands")
    assert resp.status_code == 200


def test_active_filaments(client):
    payload = {
        "brand": "Test",
        "name": "Active Test",
        "color": "#fff",
        "td": 1.0,
        "uuid": "active-001",
        "filamentType": "PLA",
    }
    resp = client.post("/api/filaments/active", json=payload)
    assert resp.status_code == 200

    resp2 = client.get("/api/filaments/active")
    items = resp2.json()
    assert len(items) == 1
    assert items[0]["name"] == "Active Test"


def test_json_import_rejects_invalid_entry_without_changing_library(client):
    client.post("/api/filaments", json={"uuid": "keep-me", "brand": "B", "name": "Keep", "td": 1.0})
    resp = client.post(
        "/api/filaments/import-json?mode=replace",
        json=[{"brand": "B", "name": "Good", "td": 1.0}, {"brand": "B", "name": "Bad", "td": "not-a-number"}],
    )
    assert resp.status_code == 400
    assert "Entry 2" in resp.json()["detail"]
    assert [f["uuid"] for f in client.get("/api/filaments").json()] == ["keep-me"]
