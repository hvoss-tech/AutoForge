import pytest
from fastapi.testclient import TestClient
from autoforge.webui import create_app
from autoforge.webui.config import config


@pytest.fixture
def client():
    app = create_app()
    return TestClient(app)


def test_telemetry_disabled_without_key(client, monkeypatch):
    monkeypatch.setattr(config, "telemetry_enabled", True)
    monkeypatch.setattr(config, "posthog_key", "")
    resp = client.get("/api/system/telemetry")
    assert resp.status_code == 200
    body = resp.json()
    assert body["enabled"] is False
    assert body["apiKey"] is None
    assert body["host"] is None
    assert body["distinctId"] is None


def test_telemetry_disabled_via_no_telemetry_flag(client, monkeypatch):
    monkeypatch.setattr(config, "telemetry_enabled", False)
    monkeypatch.setattr(config, "posthog_key", "phc_test_key")
    resp = client.get("/api/system/telemetry")
    assert resp.status_code == 200
    body = resp.json()
    assert body["enabled"] is False
    assert body["apiKey"] is None


def test_telemetry_enabled_with_key(client, monkeypatch):
    monkeypatch.setattr(config, "telemetry_enabled", True)
    monkeypatch.setattr(config, "posthog_key", "phc_test_key")
    monkeypatch.setattr(config, "posthog_host", "https://us.i.posthog.com")
    resp = client.get("/api/system/telemetry")
    assert resp.status_code == 200
    body = resp.json()
    assert body["enabled"] is True
    assert body["apiKey"] == "phc_test_key"
    assert body["host"] == "https://us.i.posthog.com"
    assert body["distinctId"]
