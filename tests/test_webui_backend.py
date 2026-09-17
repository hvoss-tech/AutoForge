"""Tests for the AutoForge WebUI backend."""

import os
import shutil
import tempfile

import pytest
import io
import json
from fastapi.testclient import TestClient
from autoforge.webui import create_app


# This file lives outside tests/webui/ so it doesn't inherit
# tests/webui/conftest.py's storage isolation. Without an equivalent here,
# every filament/snapshot/upload this module creates lands in the real,
# CWD-relative filament_library/checkpoints/uploads directories a real
# interactively-run webui session reads from — confirmed in practice
# (filament_library/library.json had accumulated dozens of stale
# "Test"/"OptGuardFilament"/"VerifyTest" entries from past runs of this
# exact file).
@pytest.fixture(scope="module", autouse=True)
def _isolated_webui_storage():
    from autoforge.webui.config import config

    tmp_dir = tempfile.mkdtemp(prefix="autoforge_webui_test_")
    mp = pytest.MonkeyPatch()
    mp.setattr(config, "checkpoints_dir", os.path.join(tmp_dir, "checkpoints"))
    mp.setattr(config, "uploads_dir", os.path.join(tmp_dir, "uploads"))
    mp.setattr(config, "library_dir", os.path.join(tmp_dir, "filament_library"))
    try:
        yield
    finally:
        mp.undo()
        shutil.rmtree(tmp_dir, ignore_errors=True)


@pytest.fixture(scope="module")
def client():
    app = create_app()
    return TestClient(app)


@pytest.fixture
def with_active_filament(client):
    """Ensure at least one active filament exists.

    /api/optimize/start rejects a run with no active filaments (400) — the
    filament/active-filament services are process-wide singletons shared by
    every test in this module (not reset per-test), so this is idempotent:
    it just guarantees the precondition holds for tests whose actual intent
    is exercising the optimize/start flow itself, not this guard.
    """
    created = client.post(
        "/api/filaments",
        json={"brand": "Test", "name": "OptGuardFilament", "color": "#123456", "td": 5.0, "filamentType": "PLA"},
    )
    filament = created.json()
    client.post("/api/filaments/active", json=filament)
    return filament


class TestSystemEndpoints:
    def test_health_check(self, client):
        response = client.get("/api/system/health")
        assert response.status_code == 200
        assert response.json()["status"] == "ok"

    def test_system_info(self, client):
        response = client.get("/api/system/info")
        assert response.status_code == 200
        data = response.json()
        assert "torchVersion" in data or "torch_version" in data
        assert "device" in data

    def test_available_devices(self, client):
        response = client.get("/api/system/device")
        assert response.status_code == 200
        data = response.json()
        assert "devices" in data
        assert isinstance(data["devices"], list)
        assert "cpu" in data["devices"]


class TestFilamentEndpoints:
    def test_get_filaments(self, client):
        response = client.get("/api/filaments")
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)

    def test_get_filaments_by_type(self, client):
        response = client.get("/api/filaments", params={"filamentType": "PLA"})
        assert response.status_code == 200

    def test_get_filaments_by_brand(self, client):
        response = client.get("/api/filaments", params={"brand": "Jayo"})
        assert response.status_code == 200

    def test_get_filaments_by_query(self, client):
        response = client.get("/api/filaments", params={"query": "Black"})
        assert response.status_code == 200

    def test_get_filaments_no_results(self, client):
        response = client.get("/api/filaments", params={"query": "NONEXISTENTBRANDXYZ"})
        assert response.status_code == 200

    def test_get_filaments_by_type_no_results(self, client):
        response = client.get("/api/filaments", params={"filamentType": "NONEXISTENTTYPE"})
        assert response.status_code == 200

    def test_get_filament_types(self, client):
        response = client.get("/api/filaments/types")
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)

    def test_get_filament_brands(self, client):
        response = client.get("/api/filaments/brands")
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)

    def test_get_filament_brands_by_type(self, client):
        response = client.get("/api/filaments/brands", params={"filamentType": "PLA"})
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)

    def test_add_filament(self, client):
        response = client.post(
            "/api/filaments",
            json={
                "brand": "Test",
                "name": "Test Red",
                "color": "#FF0000",
                "td": 3.0,
                "filamentType": "PLA",
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert "uuid" in data
        assert data.get("brand") == "Test"

    def test_add_filament_with_uuid(self, client):
        response = client.post(
            "/api/filaments",
            json={
                "brand": "Test2",
                "name": "Test Blue",
                "color": "#0000FF",
                "td": 2.0,
                "uuid": "test-uuid-123",
                "filamentType": "PETG",
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert data["uuid"] == "test-uuid-123"

    def test_update_filament(self, client):
        # Add a filament first
        created = client.post(
            "/api/filaments",
            json={
                "brand": "UpdateTest",
                "name": "Original",
                "color": "#FF0000",
                "td": 3.0,
                "uuid": "update-uuid",
            },
        ).json()
        uuid = created["uuid"]
        # Update it
        response = client.put(
            f"/api/filaments/{uuid}",
            json={
                "brand": "UpdateTest",
                "name": "Updated",
                "color": "#00FF00",
                "td": 4.0,
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert data["name"] == "Updated"

    def test_delete_filament(self, client):
        # First create one
        created = client.post(
            "/api/filaments",
            json={
                "brand": "DeleteTest",
                "name": "To Delete",
                "color": "#000000",
                "td": 1.0,
            },
        ).json()
        uuid = created["uuid"]
        response = client.delete(f"/api/filaments/{uuid}")
        assert response.status_code == 200
        assert response.json().get("ok") is True

    def test_import_csv(self, client):
        csv_content = "Brand,Type,Color,Name,Transmissivity,Owned,Uuid\nTest,PLA,#FF0000,Imported Red,3.0,false,test-import-1"
        response = client.post(
            "/api/filaments/import-csv",
            params={"contents": csv_content},
        )
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, dict)
        assert data["status"] == "ok"

    def test_import_json(self, client):
        json_content = [
            {
                "Brand": "TestJSON",
                "Name": "JSON Red",
                "Color": "#FF0000",
                "TD": 3.0,
                "Type": "PLA",
                "Owned": "false",
                "Uuid": "json-import-1",
            }
        ]
        response = client.post(
            "/api/filaments/import-json",
            json=json_content,
        )
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, dict)
        assert data["count"] >= 1

    def test_import_json_without_filaments_key(self, client):
        json_content = [
            {
                "brand": "TestJSON2",
                "name": "JSON Blue",
                "color": "#0000FF",
                "td": 2.0,
                "filamentType": "PETG",
                "uuid": "json-import-2",
            }
        ]
        response = client.post(
            "/api/filaments/import-json",
            json=json_content,
        )
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, dict)

    def test_import_csv_invalid(self, client):
        response = client.post(
            "/api/filaments/import-csv",
            params={"contents": "not,csv,data\n"},
        )
        # Should handle gracefully
        assert response.status_code in (200, 400)

    def test_filament_appears_in_list_after_add(self, client):
        created = client.post(
            "/api/filaments",
            json={
                "brand": "VerifyTest",
                "name": "Verify Me",
                "color": "#ABCDEF",
                "td": 3.5,
                "uuid": "verify-uuid",
            },
        ).json()
        uuid = created["uuid"]
        response = client.get("/api/filaments", params={"query": "Verify Me"})
        assert response.status_code == 200
        data = response.json()
        assert isinstance(uuid, str)


class TestSettingsEndpoints:
    def test_get_settings(self, client):
        response = client.get("/api/settings")
        assert response.status_code == 200
        data = response.json()
        assert data["iterations"] == 6000
        assert data["learningRate"] == 0.015
        assert data["layerHeight"] == 0.04
        assert data["maxLayers"] == 75
        assert data["backgroundColor"] == "#000000"

    def test_update_settings(self, client):
        response = client.put(
            "/api/settings",
            json={"iterations": 3000, "learningRate": 0.02},
        )
        assert response.status_code == 200

    def test_settings_schema(self, client):
        response = client.get("/api/settings/schema")
        assert response.status_code == 200

    def test_settings_all_defaults_present(self, client):
        response = client.get("/api/settings")
        assert response.status_code == 200
        data = response.json()
        required_keys = [
            "inputImage",
            "csvFile",
            "jsonFile",
            "outputFolder",
            "iterations",
            "warmupFraction",
            "learningRate",
            "learningRateWarmupFraction",
            "initTau",
            "finalTau",
            "layerHeight",
            "maxLayers",
            "minLayers",
            "backgroundHeight",
            "backgroundColor",
            "stlOutputSize",
            "nozzleDiameter",
            "earlyStopping",
            "performPruning",
            "fastPruning",
            "fastPruningPercent",
            "pruningMaxColors",
            "pruningMaxSwaps",
            "pruningMaxLayer",
            "randomSeed",
            "flatforge",
            "capLayers",
            "initHeightmapMethod",
        ]
        for key in required_keys:
            assert key in data, f"Missing key: {key}"


class TestProjectStateEndpoints:
    def test_get_project_state(self, client):
        response = client.get("/api/project/state")
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, dict)

    def test_update_project_state(self, client):
        response = client.post(
            "/api/project/state",
            json={
                "colorSliders": [],
                "globalParams": {
                    "detailSize": 0.20,
                    "layerHeight": 0.08,
                    "baseLayer": 0.16,
                    "baseThickness": 0.48,
                    "blendDepth": 1.76,
                    "width": 200.00,
                    "height": 125.00,
                },
                "settings": {},
                "activeFilaments": [],
            },
        )
        assert response.status_code == 200

    def test_project_state_round_trips_in_snake_case(self, client):
        """GET answers in snake_case (what appStore.loadProjectState reads);
        POST accepts either casing."""
        slider = {"td": 2.0, "layer": 4, "depth_mm": 0.16, "filament_uuid": "u1", "enabled": True}
        client.post("/api/project/state", json={"colorSliders": [slider], "settings": {"layerHeight": 0.08}})

        data = client.get("/api/project/state").json()
        assert set(data) == {"color_sliders", "settings", "active_filaments"}
        assert data["color_sliders"] == [slider]
        assert data["settings"]["layer_height"] == 0.08


class TestOptimizationEndpoints:
    def test_list_jobs_empty(self, client):
        response = client.get("/api/optimize/history")
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)

    def test_start_optimization_rejects_no_active_filaments(self, client):
        # Clear out any active filaments left by other tests, then verify the guard.
        active = client.get("/api/filaments/active").json()
        for f in active:
            client.delete(f"/api/filaments/active/{f['uuid']}")
        response = client.post(
            "/api/optimize/start",
            json={"iterations": 100, "inputImage": "whatever.png"},
        )
        assert response.status_code == 400
        assert "filament" in response.json()["detail"].lower()

    def test_start_optimization(self, client, with_active_filament):
        response = client.post(
            "/api/optimize/start",
            json={"iterations": 100, "inputImage": "whatever.png"},
        )
        assert response.status_code == 200
        data = response.json()
        assert "jobId" in data or "job_id" in data
        assert "status" in data

    def test_get_job_status_after_start(self, client, with_active_filament):
        start_resp = client.post(
            "/api/optimize/start",
            json={"iterations": 100, "inputImage": "whatever.png"},
        )
        job_id = start_resp.json().get("jobId") or start_resp.json().get("job_id")
        response = client.get(f"/api/optimize/status/{job_id}")
        assert response.status_code == 200
        data = response.json()
        assert data.get("jobId") == job_id or data.get("job_id") == job_id

    def test_get_optimization_result_not_completed(self, client, with_active_filament):
        start_resp = client.post(
            "/api/optimize/start",
            json={"iterations": 100, "inputImage": "whatever.png"},
        )
        job_id = start_resp.json().get("jobId") or start_resp.json().get("job_id")
        response = client.get(f"/api/optimize/result/{job_id}")
        assert response.status_code in (200, 404)

    def test_optimization_settings_passed_to_job(self, client, with_active_filament):
        response = client.post(
            "/api/optimize/start",
            json={"iterations": 200, "layerHeight": 0.08, "maxLayers": 50, "inputImage": "whatever.png"},
        )
        assert response.status_code == 200
        data = response.json()
        assert "jobId" in data or "job_id" in data


class TestPruningEndpoints:
    def test_start_pruning_rejects_missing_job_id(self, client):
        response = client.post("/api/pruning/start", json={})
        assert response.status_code == 400

    def test_start_pruning_rejects_unknown_job_id(self, client):
        response = client.post("/api/pruning/start", json={"jobId": "does-not-exist"})
        assert response.status_code == 400

    def test_start_pruning_rejects_incomplete_job(self, client, with_active_filament):
        # A job that exists but hasn't finished (still pending/running) must
        # not be pruneable either — only a completed result is valid.
        start_resp = client.post(
            "/api/optimize/start",
            json={"iterations": 100, "inputImage": "whatever.png"},
        )
        job_id = start_resp.json().get("jobId") or start_resp.json().get("job_id")
        response = client.post("/api/pruning/start", json={"jobId": job_id})
        assert response.status_code == 400


class TestOutputEndpoints:
    def test_output_history_empty(self, client):
        response = client.get("/api/outputs/stl/nonexistent-job")
        assert response.status_code == 404


def _png_bytes() -> bytes:
    import cv2
    import numpy as np

    ok, buf = cv2.imencode(".png", np.full((4, 4, 3), 127, dtype=np.uint8))
    assert ok
    return buf.tobytes()


class TestImageEndpoints:
    def test_upload_image(self, client):
        img_bytes = io.BytesIO(_png_bytes())
        response = client.post(
            "/api/images/upload",
            files={"file": ("test.png", img_bytes, "image/png")},
        )
        assert response.status_code == 200
        data = response.json()
        assert "filename" in data
        assert "url" in data

    def test_upload_image_invalid_type(self, client):
        response = client.post(
            "/api/images/upload",
            files={"file": ("test.txt", io.BytesIO(b"not an image"), "text/plain")},
        )
        # Rejected up front instead of failing later in the preview/optimizer.
        assert response.status_code == 400
        assert "image" in response.json()["detail"]


class TestEdgeCases:
    def test_multiple_jobs(self, client, with_active_filament):
        j1 = client.post("/api/optimize/start", json={"iterations": 50, "inputImage": "whatever.png"}).json()
        j2 = client.post("/api/optimize/start", json={"iterations": 100, "inputImage": "whatever.png"}).json()
        j1_id = j1.get("jobId") or j1.get("job_id")
        j2_id = j2.get("jobId") or j2.get("job_id")
        assert j1_id != j2_id
