from autoforge.webui.helpers import telemetry
from autoforge.webui.config import config


def _reset_telemetry_module_state(monkeypatch):
    monkeypatch.setattr(telemetry, "_client", None)
    monkeypatch.setattr(telemetry, "_client_built", False)
    monkeypatch.setattr(telemetry, "_distinct_id", None)


def test_capture_exception_noop_when_disabled(monkeypatch):
    _reset_telemetry_module_state(monkeypatch)
    monkeypatch.setattr(config, "telemetry_enabled", False)
    monkeypatch.setattr(config, "posthog_key", "phc_test_key")

    # Must not raise, must not build a client, with telemetry off.
    telemetry.capture_exception(ValueError("boom"), {"phase": "test"})
    assert telemetry._get_client() is None


def test_capture_exception_noop_without_key(monkeypatch):
    _reset_telemetry_module_state(monkeypatch)
    monkeypatch.setattr(config, "telemetry_enabled", True)
    monkeypatch.setattr(config, "posthog_key", "")

    telemetry.capture_exception(ValueError("boom"), {"phase": "test"})
    assert telemetry._get_client() is None


def test_capture_exception_sends_event_when_enabled(monkeypatch):
    _reset_telemetry_module_state(monkeypatch)
    monkeypatch.setattr(config, "telemetry_enabled", True)
    monkeypatch.setattr(config, "posthog_key", "phc_test_key")

    captured = {}

    class FakeClient:
        def capture(self, distinct_id, event, properties):
            captured["distinct_id"] = distinct_id
            captured["event"] = event
            captured["properties"] = properties

    monkeypatch.setattr(telemetry, "_get_client", lambda: FakeClient())

    try:
        raise ValueError("boom")
    except ValueError as e:
        telemetry.capture_exception(e, {"phase": "optimization", "job_id": "abc"})

    assert captured["event"] == "$exception"
    assert captured["properties"]["$exception_type"] == "ValueError"
    assert captured["properties"]["$exception_message"] == "boom"
    assert captured["properties"]["phase"] == "optimization"
    assert captured["properties"]["job_id"] == "abc"
    assert captured["distinct_id"]


def test_capture_exception_never_raises_on_client_failure(monkeypatch):
    _reset_telemetry_module_state(monkeypatch)

    def _broken_client():
        raise RuntimeError("client build failed")

    monkeypatch.setattr(telemetry, "_get_client", _broken_client)

    # Must swallow the failure rather than propagate it into the caller's
    # own exception-handling path.
    telemetry.capture_exception(ValueError("boom"))


def test_anon_distinct_id_persists_across_calls(tmp_path, monkeypatch):
    _reset_telemetry_module_state(monkeypatch)
    monkeypatch.setattr(telemetry.Path, "home", classmethod(lambda cls: tmp_path))

    first = telemetry._anon_distinct_id()
    monkeypatch.setattr(telemetry, "_distinct_id", None)
    second = telemetry._anon_distinct_id()

    assert first == second
    assert (tmp_path / ".autoforge" / "telemetry_id").read_text().strip() == first
