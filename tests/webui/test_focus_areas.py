"""Focus areas (priority mask) painted in the WebUI.

The image panel uploads the painted mask like any image and stores its
uploads/ file name in ``settings.priority_mask``. The optimization job has to
turn that name into a path the pipeline can open — only under uploads/,
like the input image — and the auto-preview (which doesn't use the mask)
must not try to open the bare name at all.
"""

import os
import time

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from autoforge.webui.server import create_app


@pytest.fixture
def client():
    return TestClient(create_app())


def _active_filament(client, uuid="focus-fil-1"):
    fil = {"brand": "FocusTest", "name": f"focus-{uuid}", "color": "#224466", "td": 3.0, "filament_type": "PLA", "uuid": uuid}
    client.post("/api/filaments", json=fil)
    client.post("/api/filaments/active", json=fil)


def _upload_png(client, name, image):
    ok, png = cv2.imencode(".png", image)
    assert ok
    r = client.post("/api/images/upload", files={"file": (name, png.tobytes(), "image/png")})
    assert r.status_code == 200, r.text
    return r.json()["filename"]


def _wait_for_status(client, job_id, statuses, timeout=10.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        r = client.get(f"/api/optimize/status/{job_id}")
        if r.is_success and r.json().get("status") in statuses:
            return r.json()
        time.sleep(0.05)
    raise AssertionError(f"job {job_id} never reached {statuses!r}")


def test_run_resolves_the_uploaded_mask_to_a_path_the_pipeline_can_read(client, monkeypatch):
    import autoforge.webui.api.optimization as opt_api
    from autoforge.auto_forge import _load_priority_mask

    _active_filament(client)
    image = _upload_png(client, "picture.png", np.full((20, 40, 3), 128, np.uint8))
    mask = np.zeros((20, 40), np.uint8)
    mask[5:15, 10:20] = 255
    mask_name = _upload_png(client, "focus-areas.png", mask)

    seen = {}

    def fake_run_pipeline(**kwargs):
        seen["settings"] = kwargs["settings"]
        return {"cancelled": True, "device": None, "optimizer": object()}

    monkeypatch.setattr(opt_api, "_run_pipeline", fake_run_pipeline)
    r = client.post("/api/optimize/start", json={"input_image": image, "priority_mask": mask_name, "iterations": 10})
    assert r.status_code == 200, r.text
    _wait_for_status(client, r.json()["job_id"], {"cancelled", "failed"})

    path = seen["settings"]["priority_mask"]
    assert os.path.isabs(path) and os.path.basename(path) == mask_name
    # And it loads as the loss weighting expects: 1 where painted, 0 elsewhere.
    class Args:
        priority_mask = path
        output_folder = os.path.dirname(path)

    focus = _load_priority_mask(Args, np.zeros((20, 40, 3), np.uint8), "cpu")
    assert focus is not None and tuple(focus.shape) == (20, 40)
    assert float(focus[10, 15]) == pytest.approx(1.0)
    assert float(focus[0, 0]) == pytest.approx(0.0)


def test_a_missing_mask_fails_the_run_with_a_clear_message(client, monkeypatch):
    import autoforge.webui.api.optimization as opt_api

    _active_filament(client)
    image = _upload_png(client, "picture.png", np.full((20, 40, 3), 128, np.uint8))
    monkeypatch.setattr(opt_api, "_run_pipeline", lambda **kw: pytest.fail("pipeline must not start"))

    r = client.post("/api/optimize/start", json={"input_image": image, "priority_mask": "gone.png", "iterations": 10})
    assert r.status_code == 200, r.text
    status = _wait_for_status(client, r.json()["job_id"], {"failed"})
    assert "Focus-area mask not found" in status["error"]


@pytest.mark.parametrize("name", ["../../etc/passwd", "/etc/passwd"])
def test_a_mask_outside_uploads_is_refused(client, monkeypatch, name):
    import autoforge.webui.api.optimization as opt_api

    _active_filament(client)
    image = _upload_png(client, "picture.png", np.full((20, 40, 3), 128, np.uint8))
    monkeypatch.setattr(opt_api, "_run_pipeline", lambda **kw: pytest.fail("pipeline must not start"))

    r = client.post("/api/optimize/start", json={"input_image": image, "priority_mask": name, "iterations": 10})
    status = _wait_for_status(client, r.json()["job_id"], {"failed"})
    assert "Focus-area mask not found" in status["error"]


def test_the_auto_preview_ignores_the_mask(client, monkeypatch):
    import autoforge.webui.api.init as init_api

    _active_filament(client)
    image = _upload_png(client, "picture.png", np.full((20, 40, 3), 128, np.uint8))
    seen = {}

    def fake_run_init_sync(input_image_path, filament_dicts, settings_dict):
        seen["mask"] = settings_dict["priority_mask"]
        return {
            "result": {"optimizer": object()},
            "preview_b64": "png",
            "range": {"min_layer": 0, "max_layer": 30},
            "base": {"color": "#111111", "height_mm": 0.24, "layers": 6, "filament_uuid": "", "auto": True},
        }

    monkeypatch.setattr(init_api, "_run_init_sync", fake_run_init_sync)
    r = client.post("/api/init/run", json={"input_image": image, "priority_mask": "focus.png", "num_init_rounds": 1})
    assert r.status_code == 200, r.text
    assert seen["mask"] == ""


# ---------------------------------------------------------------------------
# Strength: how many times more a painted pixel counts (--priority_mask_strength)
# ---------------------------------------------------------------------------


def _loss_with_error_at(pixel, focus):
    """Loss when only `pixel` (0 or 1 of a 1x2 image) is off, weighted by `focus`."""
    import torch

    from autoforge.Loss.LossFunctions import compute_loss

    target = torch.full((1, 2, 3), 128.0)
    comp = target.clone()
    comp[0, pixel] = torch.tensor([200.0, 60.0, 60.0])
    return float(compute_loss(comp=comp, target=target, focus_map=focus))


@pytest.mark.parametrize("strength", [2.0, 10.0, 37.0, 100.0])
def test_strength_is_the_ratio_between_painted_and_unpainted_weight(strength):
    import torch

    from autoforge.auto_forge import priority_mask_scale

    # Pixel 0 painted (mask 1), pixel 1 not (mask 0), as _load_priority_mask scales it.
    focus = torch.tensor([[1.0, 0.0]]) * priority_mask_scale(strength)
    ratio = _loss_with_error_at(0, focus) / _loss_with_error_at(1, focus)
    assert ratio == pytest.approx(strength, rel=1e-4)


def test_default_strength_keeps_the_cli_behaviour():
    """10x is what a plain 0..1 mask always meant (0.1 outside, 1.0 inside)."""
    from autoforge.auto_forge import DEFAULT_PRIORITY_MASK_STRENGTH, priority_mask_scale

    assert DEFAULT_PRIORITY_MASK_STRENGTH == 10.0
    assert priority_mask_scale(10.0) == 1.0
    assert priority_mask_scale(0.5) == 0.0  # never below "no preference"


def test_load_priority_mask_applies_the_strength(tmp_path):
    from autoforge.auto_forge import _load_priority_mask

    mask = np.zeros((10, 10), np.uint8)
    mask[:5] = 255
    path = tmp_path / "mask.png"
    cv2.imwrite(str(path), mask)

    class Args:
        priority_mask = str(path)
        priority_mask_strength = 55.0
        output_folder = str(tmp_path)

    focus = _load_priority_mask(Args, np.zeros((10, 10, 3), np.uint8), "cpu")
    assert float(focus.max()) == pytest.approx(6.0)  # (55 - 1) / 9
    assert float(focus.min()) == 0.0
    # The diagnostic copy stays a plain 0..255 mask.
    assert cv2.imread(str(tmp_path / "priority_mask_resized.png"), cv2.IMREAD_GRAYSCALE).max() == 255


def test_the_webui_passes_the_strength_to_the_pipeline(client, monkeypatch):
    import autoforge.webui.api.optimization as opt_api

    _active_filament(client)
    image = _upload_png(client, "picture.png", np.full((20, 40, 3), 128, np.uint8))
    mask_name = _upload_png(client, "focus-areas.png", np.full((20, 40), 255, np.uint8))
    seen = {}

    def fake_run_pipeline(**kwargs):
        seen["settings"] = kwargs["settings"]
        return {"cancelled": True, "device": None, "optimizer": object()}

    monkeypatch.setattr(opt_api, "_run_pipeline", fake_run_pipeline)
    r = client.post("/api/optimize/start", json={"input_image": image, "priority_mask": mask_name, "priority_mask_strength": 40, "iterations": 10})
    assert r.status_code == 200, r.text
    _wait_for_status(client, r.json()["job_id"], {"cancelled", "failed"})
    assert seen["settings"]["priority_mask_strength"] == 40

    # Out of range is refused up front.
    r = client.post("/api/optimize/start", json={"input_image": image, "priority_mask_strength": 0.5, "iterations": 10})
    assert r.status_code == 422
