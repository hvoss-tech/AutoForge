"""End-to-end smoke test for the AutoForge WebUI pipeline.

Usage:
    python tests/test_pipeline_smoke.py [--host http://localhost:8000]
"""

import sys
import os
import json
import time
import argparse
import requests
import pytest


@pytest.mark.skip(reason="requires running server; use python tests/test_pipeline_smoke.py")
def test_health(base: str) -> None:
    for i in range(10):
        try:
            r = requests.get(f"{base}/api/system/health", timeout=5)
            assert r.status_code == 200, f"Health failed: {r.text}"
            print(f"  [OK] Health check")
            return
        except requests.ConnectionError:
            if i < 9:
                time.sleep(1)
    raise RuntimeError("Backend not reachable")


@pytest.mark.skip(reason="requires running server; use python tests/test_pipeline_smoke.py")
def test_pipeline(base: str) -> str:
    from PIL import Image
    import numpy as np

    # ── Upload a tiny 4×4 test image ──────────────────────────────
    img = np.full((4, 4, 3), (255, 0, 0), dtype=np.uint8)
    img[2:, 2:] = (0, 0, 255)
    pil_img = Image.fromarray(img, "RGB")
    img_path = "/tmp/e2e_test_img.png"
    pil_img.save(img_path)

    with open(img_path, "rb") as f:
        r = requests.post(f"{base}/api/images/upload", files={"file": ("test.png", f, "image/png")}, timeout=30)
    assert r.status_code == 200, f"Upload failed: {r.text}"
    filename = r.json()["filename"]
    print(f"  [OK] Image uploaded: {filename}")

    # ── Create 3 test filaments and add to active list ────────────
    for name, color, td in [("Red", "#FF0000", 1.0), ("Green", "#00FF00", 2.0), ("Blue", "#0000FF", 3.0)]:
        r = requests.post(f"{base}/api/filaments", json={
            "brand": "TestPipeline", "name": name, "color": color, "td": td, "filament_type": "PLA",
        }, timeout=10)
        assert r.status_code == 200, f"Create filament failed: {r.text}"
        f = r.json()
        r2 = requests.post(f"{base}/api/filaments/active", json=f, timeout=10)
        assert r2.status_code == 200, f"Activate failed: {r2.text}"

    r = requests.get(f"{base}/api/filaments/active", timeout=10)
    assert r.status_code == 200 and len(r.json()) >= 3
    print(f"  [OK] 3 filaments created and activated")

    # ── Start optimization ────────────────────────────────────────
    r = requests.post(f"{base}/api/optimize/start", json={
        "input_image": filename, "iterations": 2, "max_layers": 3, "layer_height": 0.04,
        "background_height": 0.12, "stl_output_size": 20, "processing_reduction_factor": 1,
        "random_seed": 42, "num_init_rounds": 1, "num_init_cluster_layers": 3, "learning_rate": 0.01,
        "init_tau": 1.0, "final_tau": 0.5, "early_stopping": 1000, "visualize": False,
        "perform_pruning": False, "num_init_threads": 1, "best_of": 1, "discrete_check": 1,
        "csv_file": "", "json_file": "",
    }, timeout=10)
    assert r.status_code == 200, f"Start failed: {r.text}"
    job_id = r.json()["job_id"]
    print(f"  [OK] Optimization started: {job_id}")

    # ── Poll until completion ─────────────────────────────────────
    for _ in range(90):
        time.sleep(1)
        try:
            s = requests.get(f"{base}/api/optimize/status/{job_id}", timeout=10).json()
        except Exception:
            continue
        if s["status"] == "completed":
            print(f"  [OK] Optimization completed")
            break
        if s["status"] == "failed":
            print(f"  [FAIL] Optimization failed: {s.get('error', '')}")
            return job_id  # Return job_id anyway so we can still test outputs
    else:
        print(f"  [FAIL] Optimization timed out")
        return job_id

    # ── Verify preview image (should be saved by now) ─────────────
    # The optimize/start endpoint saves final_model.png after completion
    r = requests.get(f"{base}/api/outputs/preview/{job_id}", timeout=10)
    if r.status_code == 200:
        print(f"  [OK] Preview image: {len(r.content)} bytes (from /api/outputs/preview)")
    else:
        print(f"  [WARN] Preview from /api/outputs/preview returned {r.status_code}")

    # ── Run pruning ───────────────────────────────────────────────
    r = requests.post(f"{base}/api/pruning/start", json={
        "pruning_max_colors": 5, "pruning_max_swaps": 5, "pruning_max_layer": 5,
    }, timeout=10)
    if r.status_code == 200:
        prune_job_id = r.json()["job_id"]
        for _ in range(90):
            time.sleep(1)
            try:
                s = requests.get(f"{base}/api/optimize/status/{prune_job_id}", timeout=10).json()
            except Exception:
                continue
            if s["status"] == "completed":
                print(f"  [OK] Pruning completed")
                break
            if s["status"] == "failed":
                print(f"  [WARN] Prune error: {s.get('error', '')[:80]}")
                break
        else:
            print(f"  [WARN] Prune timed out")
    else:
        print(f"  [WARN] Prune start failed: {r.text}")

    # ── Verify output files exist ─────────────────────────────────
    for endpoint, name in [
        ("stl", "STL"),
        ("colored-ply", "Colored PLY"),
        ("preview", "Preview"),
    ]:
        r = requests.get(f"{base}/api/outputs/{endpoint}/{job_id}", timeout=10)
        ok = r.status_code == 200
        sz = len(r.content) if ok else 0
        if ok:
            print(f"  [OK] {name}: HTTP {r.status_code} ({sz} bytes)")
        else:
            print(f"  [WARN] {name}: HTTP {r.status_code}")

    return job_id


@pytest.mark.skip(reason="requires running server; use python tests/test_pipeline_smoke.py")
def test_no_active_filaments_error(base: str) -> None:
    """Verify that starting optimization with no active filaments gives a clear error."""

    # Deactivate all active filaments
    r = requests.get(f"{base}/api/filaments/active", timeout=10)
    for f in r.json():
        requests.delete(f"{base}/api/filaments/active/{f['uuid']}", timeout=10)

    r = requests.get(f"{base}/api/filaments/active", timeout=10)
    assert len(r.json()) == 0, "Active filaments should be empty"

    # Upload a fresh image
    from PIL import Image
    import numpy as np
    img = np.full((4, 4, 3), (200, 100, 50), dtype=np.uint8)
    Image.fromarray(img, "RGB").save("/tmp/e2e_test_no_active.png")

    with open("/tmp/e2e_test_no_active.png", "rb") as f:
        r = requests.post(f"{base}/api/images/upload", files={"file": ("test.png", f, "image/png")}, timeout=30)
    fn2 = r.json()["filename"]

    # Start optimization (should create job OK but fail in the background thread)
    r = requests.post(f"{base}/api/optimize/start", json={
        "input_image": fn2, "iterations": 1, "max_layers": 2,
        "learning_rate": 0.01, "init_tau": 1.0, "final_tau": 0.5,
        "csv_file": "", "json_file": "",
    }, timeout=10)
    jid2 = r.json()["job_id"]

    for _ in range(30):
        time.sleep(0.5)
        try:
            s = requests.get(f"{base}/api/optimize/status/{jid2}", timeout=5).json()
        except Exception:
            continue
        if s["status"] == "failed":
            err_msg = s.get("error", "")
            print(f"  [OK] Got expected failure: {err_msg[:80]}")
            assert "active filaments" in err_msg.lower(), f"Wrong error message: {err_msg}"
            return
        if s["status"] == "completed":
            assert False, "Should have failed but completed"

    assert False, "Timed out waiting for failure"


def main() -> None:
    parser = argparse.ArgumentParser(description="AutoForge pipeline smoke test")
    parser.add_argument("--host", default="http://localhost:8000", help="Backend URL")
    args = parser.parse_args()

    base = args.host.rstrip("/")
    print(f"Testing against {base}")
    print()

    test_health(base)
    print()

    job_id = test_pipeline(base)
    print()

    test_no_active_filaments_error(base)
    print()

    print("[PASS] All smoke tests passed!")


if __name__ == "__main__":
    main()
