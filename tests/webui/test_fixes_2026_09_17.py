"""Backend half of the 2026-09-17 WebUI fixes.

Covers: the optional preview-mesh decimation (item 3), the resolved
base/background color the color layers show (item 1), the live pruning counts
(item 9), the auto-preview reset that stops a new image being shown with the
previous one's heightmap (item 10), the pipeline-result eviction that stops
VRAM climbing run after run (item 11), and the follow-up round: spike removal
no longer degrading a repeat prune, and auto-repeat pruning.
"""

import os
import types

import numpy as np
import pytest
from fastapi.testclient import TestClient

from autoforge.webui.server import create_app


@pytest.fixture
def client():
    return TestClient(create_app())


# --------------------------------------------------------------------------
# Item 3: preview mesh size
# --------------------------------------------------------------------------


def _mesh(height, width, **kwargs):
    from autoforge.webui.helpers.colored_mesh import generate_colored_preview_mesh

    rng = np.random.default_rng(0)
    return generate_colored_preview_mesh(
        height_map=rng.random((height, width)).astype(np.float32),
        color_image=rng.integers(0, 256, (height, width, 3), dtype=np.uint8),
        background_height=0.24,
        maximum_x_y_size=150.0,
        **kwargs,
    )


def test_downsample_grid_step():
    from autoforge.webui.helpers.colored_mesh import downsample_grid_step

    assert downsample_grid_step(100, 100, 384) == 1
    assert downsample_grid_step(384, 200, 384) == 1
    assert downsample_grid_step(768, 1024, 384) == 3
    assert downsample_grid_step(4000, 100, 384) == 11
    # 0 (or less) disables decimation entirely.
    assert downsample_grid_step(4000, 4000, 0) == 1


def test_preview_mesh_keeps_full_detail_by_default():
    """Decimation is opt-in.

    Capping the preview grid was measured at ~70ms instead of ~325ms per
    slider re-render — but at the default settings the solved grid is 750px
    and a 384 cap lands almost exactly on the *processing* resolution, so the
    preview visibly lost half its relief detail in each axis. On the client
    the full mesh costs ~28ms to parse and ~50ms for normals, both off the
    main thread, so the detail was being traded for time nobody waits on.
    """
    from autoforge.webui.helpers.colored_mesh import preview_mesh_max_dim

    assert preview_mesh_max_dim() == 0
    full = _mesh(768, 1024)
    assert full.vertices.shape[0] == 2 * 768 * 1024


def test_preview_mesh_is_decimated_when_a_cap_is_set():
    # Still available for very large stl_output_size values, where the mesh
    # grows with the *area* of the height map.
    big = _mesh(768, 1024, max_grid_dim=384)
    full = _mesh(768, 1024, max_grid_dim=0)
    assert big.vertices.shape[0] < full.vertices.shape[0] / 5
    assert big.faces.shape[0] > 0


def test_decimation_keeps_the_footprint_and_height():
    """Striding must not shrink or skew the model.

    The sampled grid is renumbered 0..W-1, so positions have to come from the
    original pixel indices — otherwise the final (narrower) row and column
    get stretched to full width and the mesh no longer matches the image's
    aspect ratio.
    """
    big = _mesh(768, 1024, max_grid_dim=384)
    full = _mesh(768, 1024, max_grid_dim=0)
    assert np.allclose(big.bounds[:, 0], full.bounds[:, 0], atol=1e-3)  # X
    assert np.allclose(big.bounds[:, 1], full.bounds[:, 1], atol=1e-3)  # Y
    # X:Y ratio follows the image, not the strided sample count.
    width, depth = big.extents[0], big.extents[1]
    assert width / depth == pytest.approx((1024 - 1) / (768 - 1), rel=1e-3)


def test_small_meshes_are_untouched():
    small = _mesh(16, 16, max_grid_dim=384)
    assert small.vertices.shape[0] == 2 * 16 * 16


def test_preview_mesh_max_dim_is_configurable(monkeypatch):
    from autoforge.webui.helpers import colored_mesh

    monkeypatch.setenv("AUTOFORGE_PREVIEW_MESH_MAX_DIM", "64")
    assert colored_mesh.preview_mesh_max_dim() == 64
    monkeypatch.setenv("AUTOFORGE_PREVIEW_MESH_MAX_DIM", "not-a-number")
    assert colored_mesh.preview_mesh_max_dim() == 0
    monkeypatch.delenv("AUTOFORGE_PREVIEW_MESH_MAX_DIM")
    assert colored_mesh.preview_mesh_max_dim() == 0


def test_decimated_mesh_still_carries_vertex_colors():
    big = _mesh(600, 600, max_grid_dim=384)
    assert big.visual.vertex_colors is not None
    assert big.visual.vertex_colors.shape[0] == big.vertices.shape[0]


# --------------------------------------------------------------------------
# Item 1: the resolved base / background color
# --------------------------------------------------------------------------


def _result(**overrides):
    args = types.SimpleNamespace(
        background_color="#112233",
        background_height=0.24,
        layer_height=0.04,
        auto_background_color=True,
    )
    for key, value in overrides.pop("args", {}).items():
        setattr(args, key, value)
    result = {
        "args": args,
        "material_uuids": ["uuid-a", "uuid-b", "uuid-c"],
        "colors_list": ["#aabbcc", "#112233", "#ffffff"],
    }
    result.update(overrides)
    return result


def test_derive_base_uses_the_auto_selected_filament():
    from autoforge.webui.helpers.sliders import derive_base_from_result

    # This is the case the UI cannot work out for itself: auto-selection
    # replaced args.background_color with an active filament's color, and
    # only the pipeline knows which index that was.
    result = _result(args={"background_material_index": 2, "background_color": "#ffffff"})
    base = derive_base_from_result(result)
    assert base["color"] == "#ffffff"
    assert base["filament_uuid"] == "uuid-c"
    assert base["auto"] is True
    assert base["height_mm"] == 0.24
    assert base["layers"] == 6


def test_derive_base_matches_a_hand_picked_color_back_to_a_filament():
    from autoforge.webui.helpers.sliders import derive_base_from_result

    base = derive_base_from_result(_result(args={"auto_background_color": False}))
    assert base["color"] == "#112233"
    assert base["filament_uuid"] == "uuid-b"
    assert base["auto"] is False


def test_derive_base_tolerates_a_color_no_filament_has():
    from autoforge.webui.helpers.sliders import derive_base_from_result

    base = derive_base_from_result(_result(args={"background_color": "#010203"}))
    assert base["color"] == "#010203"
    assert base["filament_uuid"] == ""


def test_derive_base_handles_an_out_of_range_index():
    from autoforge.webui.helpers.sliders import derive_base_from_result

    base = derive_base_from_result(_result(args={"background_material_index": 99}))
    # Falls through to the hex match rather than raising.
    assert base["filament_uuid"] == "uuid-b"


def test_base_endpoint_reports_null_without_a_result(client):
    assert client.get("/api/sliders/base").json() == {"base": None, "source": None}


def test_base_endpoint_answers_from_the_auto_preview(client, monkeypatch):
    from autoforge.webui.api import init as init_api

    monkeypatch.setattr(init_api, "_pipeline_result", _result(args={"background_material_index": 0}))
    payload = client.get("/api/sliders/base").json()
    assert payload["source"] == "init"
    assert payload["base"]["filament_uuid"] == "uuid-a"


def test_base_endpoint_prefers_a_completed_job(client, monkeypatch):
    from autoforge.webui.api import init as init_api
    from autoforge.webui.services.optimization_service import get_optimization_service

    svc = get_optimization_service()
    job = svc.create_job({"iterations": 1}, job_id="job-base")
    svc.update_status(job.job_id, "completed")
    svc.set_pipeline_result(job.job_id, _result(args={"background_material_index": 2}))
    monkeypatch.setattr(init_api, "_pipeline_result", _result(args={"background_material_index": 0}))

    payload = client.get("/api/sliders/base").json()
    assert payload["source"] == "job-base"
    assert payload["base"]["filament_uuid"] == "uuid-c"


# --------------------------------------------------------------------------
# Item 9: live pruning counts
# --------------------------------------------------------------------------


class _FakeOptimizer:
    """Just enough of FilamentOptimizer for the counting helper."""

    def __init__(self, stack, max_height):
        import torch

        self._stack = torch.tensor(stack, dtype=torch.int64)
        self._heights = torch.full((4, 4), float(max_height))

    def get_discretized_solution(self, best=False):
        return self._stack, self._heights


def test_result_counts_match_how_the_dialog_counts():
    from autoforge.webui.helpers.sliders import result_counts_from_optimizer

    # Three materials in four runs over 8 layers: 3 distinct + base = 4
    # colors, 4 runs - 1 = 3 swaps, top layer 8.
    counts = result_counts_from_optimizer(_FakeOptimizer([0, 0, 1, 1, 2, 2, 0, 0], 8))
    assert counts == {"colors": 4, "swaps": 3, "layers": 8}


def test_result_counts_ignore_layers_above_the_height_map():
    from autoforge.webui.helpers.sliders import result_counts_from_optimizer

    # disc_global covers max_layers, but only the layers the heightmap
    # actually reaches are printed — counting the rest inflated both colors
    # and swaps versus what the dialog shows.
    counts = result_counts_from_optimizer(_FakeOptimizer([0, 0, 1, 1, 2, 3, 4, 5], 4))
    assert counts == {"colors": 3, "swaps": 1, "layers": 4}


def test_result_counts_for_a_single_color():
    from autoforge.webui.helpers.sliders import result_counts_from_optimizer

    assert result_counts_from_optimizer(_FakeOptimizer([1, 1, 1, 1], 4)) == {
        "colors": 2,
        "swaps": 0,
        "layers": 4,
    }


def test_result_counts_for_an_empty_stack():
    from autoforge.webui.helpers.sliders import result_counts_from_optimizer

    assert result_counts_from_optimizer(_FakeOptimizer([0, 0, 0], 0)) == {
        "colors": 1,
        "swaps": 0,
        "layers": 0,
    }


def test_result_counts_none_without_a_solution():
    from autoforge.webui.helpers.sliders import result_counts_from_optimizer

    class NoSolution:
        def get_discretized_solution(self, best=False):
            return None, None

    assert result_counts_from_optimizer(NoSolution()) is None


def test_job_status_carries_the_live_counts(client):
    from autoforge.webui.services.optimization_service import get_optimization_service

    svc = get_optimization_service()
    svc.create_job({"iterations": 1}, job_id="prune-live")
    svc.update_status("prune-live", "running", progress=20.0, phase="Reducing colors",
                      result_colors=9, result_swaps=14, result_layers=61)

    status = client.get("/api/optimize/status/prune-live").json()
    assert status["result_colors"] == 9
    assert status["result_swaps"] == 14
    assert status["result_layers"] == 61


def test_a_paused_job_stops_advancing(client):
    """A progress report that lands after the pause must not be applied.

    The report describes work that finished *before* the user clicked — a
    callback already in flight can only arrive afterwards. Writing its
    progress/phase anyway made a paused job look like it was still working:
    the pruning overlay's phase label moved on ("Pruning" -> "Reducing
    colors") a moment after Pause, with the bar frozen next to it.
    """
    from autoforge.webui.services.optimization_service import get_optimization_service

    svc = get_optimization_service()
    svc.create_job({"iterations": 1}, job_id="prune-pause")
    svc.update_status("prune-pause", "running", progress=30.0, phase="Reducing colors", result_colors=9)
    assert svc.pause("prune-pause")

    # In-flight callback arrives now.
    svc.update_status("prune-pause", "running", progress=55.0, phase="Reducing swaps", result_colors=7)

    job = svc.get_job("prune-pause")
    assert job.status == "paused"
    assert job.progress == 30.0
    assert job.phase == "Reducing colors"
    assert job.result_colors == 9

    # Resuming lets reports through again.
    assert svc.resume("prune-pause")
    svc.update_status("prune-pause", "running", progress=60.0, phase="Reducing swaps", result_colors=7)
    job = svc.get_job("prune-pause")
    assert (job.status, job.progress, job.phase, job.result_colors) == ("running", 60.0, "Reducing swaps", 7)


def test_should_repeat_prune():
    """When auto-repeat keeps going, and when it stops.

    The stop rule is what makes "keep pruning until it stops improving" a
    feature rather than an infinite loop: a pass has to beat the best loss so
    far by more than measurement noise, and there is a hard pass ceiling.
    """
    from autoforge.webui.helpers.sliders import should_repeat_prune

    # The first pass has nothing to compare against yet.
    assert should_repeat_prune(1, 25, 100.0, None)[0] is True
    # A real improvement keeps it going.
    assert should_repeat_prune(2, 25, 99.0, 100.0)[0] is True
    # A worse or unchanged pass stops it.
    repeat, reason = should_repeat_prune(3, 25, 100.0, 100.0)
    assert repeat is False and reason == "no further improvement"
    assert should_repeat_prune(3, 25, 101.0, 100.0)[0] is False
    # So does a difference too small to be anything but re-scoring noise.
    assert should_repeat_prune(3, 25, 100.0 - 1e-9, 100.0)[0] is False
    # The pass ceiling wins even mid-improvement.
    repeat, reason = should_repeat_prune(25, 25, 1.0, 100.0)
    assert repeat is False and "25-pass limit" in reason
    # A pass whose loss couldn't be measured must not end the run silently.
    assert should_repeat_prune(2, 25, None, 100.0)[0] is True


def test_pruning_accepts_the_auto_repeat_settings(client):
    from autoforge.webui.models import PruningSettings

    settings = PruningSettings(job_id="x", auto_repeat=True, max_passes=8)
    assert settings.auto_repeat is True
    assert settings.max_passes == 8
    # Off, with a sane ceiling, unless asked for.
    assert PruningSettings().auto_repeat is False
    assert PruningSettings().max_passes == 25
    # And the ceiling is range-checked rather than accepted blindly.
    with pytest.raises(ValueError):
        PruningSettings(max_passes=0)
    with pytest.raises(ValueError):
        PruningSettings(max_passes=10_000)


def test_the_full_resolution_height_restore_happens_only_once():
    """The single biggest cause of "pruning again makes it worse".

    Training runs at the *processing* resolution, so the first export swaps in
    the full-resolution initial height map before writing anything out. Doing
    that again on a later export threw away a height map that pruning had
    already pruned, fine-tuned and de-spiked, and handed the next pass the
    untrained k-means init instead — measured jumping the discrete loss from
    68.08 back to 99.96 before a single pruning phase had run, so every pass
    after the first started from a worse solution than it was meant to
    improve (68.08 -> 70.86 -> 77.78 over three passes; monotonically
    improving and converging at 64.66 once fixed).
    """
    import inspect

    from autoforge.webui.helpers import pipeline_runner

    source = inspect.getsource(pipeline_runner.export_results)
    assert 'if not getattr(optimizer, "_full_res_height_restored", False):' in source
    assert "optimizer._full_res_height_restored = True" in source

    # And the flag lives on the optimizer, so a *different* job's optimizer
    # still gets its own first-time restore.
    class FakeOptimizer:
        pass

    first, second = FakeOptimizer(), FakeOptimizer()
    assert getattr(first, "_full_res_height_restored", False) is False
    first._full_res_height_restored = True
    assert getattr(second, "_full_res_height_restored", False) is False


def test_pruning_settings_expose_the_polish_passes(client):
    """Both used to be hardcoded — 200 seeds inside prune(), 50 fine-tune
    steps in export_results — so on is the behaviour-preserving default."""
    from autoforge.webui.models import PruningSettings

    defaults = PruningSettings()
    assert defaults.seed_search is True
    assert defaults.seed_search_count == 200
    assert defaults.fine_tune_height is True
    assert defaults.fine_tune_steps == 50

    custom = PruningSettings(seed_search=False, seed_search_count=5000, fine_tune_steps=200)
    assert (custom.seed_search, custom.seed_search_count, custom.fine_tune_steps) == (False, 5000, 200)

    for bad in ({"seed_search_count": 0}, {"seed_search_count": 99_999}, {"fine_tune_steps": 0}):
        with pytest.raises(ValueError):
            PruningSettings(**bad)


def test_pruning_forwards_the_polish_settings_to_the_optimizer():
    """The settings have to reach prune() — they are what makes the toggles
    do anything."""
    import inspect

    from autoforge.webui.api import pruning as pruning_api
    from autoforge.webui.helpers import pipeline_runner

    api_source = inspect.getsource(pruning_api.start_pruning)
    for field in ("seed_search", "seed_search_count", "fine_tune_height", "fine_tune_steps"):
        assert f"settings.{field}" in api_source

    runner_source = inspect.getsource(pipeline_runner.export_results)
    assert "prune_seed_search_count" in runner_source
    assert "prune_fine_tune_steps" in runner_source
    # The old unconditional 50-step fine-tune before pruning is gone; it is a
    # prune() phase now, so it reports progress like every other one.
    assert "optimizer.fine_tune_height_offsets(num_steps=50)" not in runner_source


def test_job_status_reports_the_start_loss(client):
    from autoforge.webui.services.optimization_service import get_optimization_service

    svc = get_optimization_service()
    svc.create_job({"iterations": 1}, job_id="prune-loss")
    svc.update_status("prune-loss", "running", pruning_start_loss=1081.2, loss=765.9)
    status = client.get("/api/optimize/status/prune-loss").json()
    # The dialog shows both: the polish passes change no counts at all, so
    # the loss is the only place their improvement is visible.
    assert status["pruning_start_loss"] == pytest.approx(1081.2)
    assert status["loss"] == pytest.approx(765.9)


def test_job_status_reports_the_pruning_pass(client):
    from autoforge.webui.services.optimization_service import get_optimization_service

    svc = get_optimization_service()
    svc.create_job({"iterations": 1}, job_id="prune-passes")
    svc.update_status("prune-passes", "running", pruning_pass=3, pruning_max_passes=25)
    status = client.get("/api/optimize/status/prune-passes").json()
    assert status["pruning_pass"] == 3
    assert status["pruning_max_passes"] == 25


def test_job_status_counts_default_to_null(client):
    from autoforge.webui.services.optimization_service import get_optimization_service

    get_optimization_service().create_job({"iterations": 1}, job_id="plain")
    status = client.get("/api/optimize/status/plain").json()
    assert status["result_colors"] is None
    assert status["result_layers"] is None


# --------------------------------------------------------------------------
# Item 10: a new image must not be shown with the previous one's preview
# --------------------------------------------------------------------------


def test_init_reset_clears_state_mesh_and_result(client, monkeypatch, tmp_path):
    from autoforge.webui.api import init as init_api

    mesh = tmp_path / "final_model_colored.ply"
    mesh.write_bytes(b"ply")
    monkeypatch.setattr(init_api, "_mesh_path", lambda: str(mesh))
    monkeypatch.setattr(init_api, "_state", {"status": "ready", "preview_image": "x", "error": None})
    monkeypatch.setattr(init_api, "_pipeline_result", _result())

    assert client.post("/api/init/reset").json() == {"status": "idle"}
    assert init_api._state["status"] == "idle"
    assert init_api._pipeline_result is None
    # The file has to go too: /api/init/mesh serves whatever is on disk, so
    # leaving it there kept the old image visible as the new one's preview.
    assert not mesh.exists()


def test_init_mesh_404s_after_a_reset(client, monkeypatch, tmp_path):
    from autoforge.webui.api import init as init_api

    mesh = tmp_path / "final_model_colored.ply"
    mesh.write_bytes(b"ply")
    monkeypatch.setattr(init_api, "_mesh_path", lambda: str(mesh))
    monkeypatch.setattr(init_api, "_state", {"status": "ready", "preview_image": None, "error": None})
    monkeypatch.setattr(init_api, "_pipeline_result", _result())

    assert client.get("/api/init/mesh").status_code == 200
    client.post("/api/init/reset")
    assert client.get("/api/init/mesh").status_code == 404


def test_init_reset_is_safe_without_a_mesh(client, monkeypatch, tmp_path):
    from autoforge.webui.api import init as init_api

    monkeypatch.setattr(init_api, "_mesh_path", lambda: str(tmp_path / "missing.ply"))
    assert client.post("/api/init/reset").status_code == 200


def test_init_status_reports_the_resolved_base(client, monkeypatch):
    from autoforge.webui.api import init as init_api

    monkeypatch.setattr(
        init_api,
        "_state",
        {"status": "ready", "preview_image": None, "error": None,
         "range": {"min_layer": 0, "max_layer": 20},
         "base": {"color": "#ffffff", "filament_uuid": "uuid-c", "height_mm": 0.24, "layers": 6, "auto": True}},
    )
    payload = client.get("/api/init/status").json()
    assert payload["base"]["color"] == "#ffffff"
    assert payload["max_layer"] == 20


# --------------------------------------------------------------------------
# Item 11: the VRAM leak
# --------------------------------------------------------------------------


def test_only_the_newest_pipeline_result_is_kept():
    """Each pipeline result pins a whole optimizer on the GPU.

    Keeping one per job is what made VRAM climb when optimizing one image
    after another: nothing ever dropped the earlier runs, and only the
    newest one is reachable from the UI anyway.
    """
    from autoforge.webui.services.optimization_service import get_optimization_service

    svc = get_optimization_service()
    first = {"optimizer": object(), "device": None}
    second = {"optimizer": object(), "device": None}

    svc.set_pipeline_result("job-1", first)
    assert svc.pipeline_result_job_ids() == ["job-1"]

    svc.set_pipeline_result("job-2", second)
    assert svc.pipeline_result_job_ids() == ["job-2"]
    assert svc.get_pipeline_result("job-1") is None
    assert svc.get_pipeline_result("job-2") is second
    # The evicted result was emptied, not merely unreferenced — a traceback
    # frame or a job thread's local can easily outlive the dict.
    assert first == {}


def test_replacing_a_job_s_own_result_releases_the_old_one():
    from autoforge.webui.services.optimization_service import get_optimization_service

    svc = get_optimization_service()
    old = {"optimizer": object()}
    new = {"optimizer": object()}
    svc.set_pipeline_result("job-1", old)
    svc.set_pipeline_result("job-1", new)
    assert svc.get_pipeline_result("job-1") is new
    assert old == {}


def test_clear_all_pipeline_results_releases_everything():
    from autoforge.webui.services.optimization_service import get_optimization_service

    svc = get_optimization_service()
    held = {"optimizer": object()}
    svc.set_pipeline_result("job-1", held)
    svc.clear_all_pipeline_results()
    assert svc.pipeline_result_job_ids() == []
    assert held == {}


def test_clear_pipeline_result_releases_it():
    from autoforge.webui.services.optimization_service import get_optimization_service

    svc = get_optimization_service()
    held = {"optimizer": object()}
    svc.set_pipeline_result("job-1", held)
    svc.clear_pipeline_result("job-1")
    assert held == {}
    # Clearing something that isn't there is fine.
    svc.clear_pipeline_result("job-1")


def test_release_pipeline_result_drops_a_captured_graph():
    from autoforge.webui.helpers.gpu_memory import release_pipeline_result

    released = []

    class Optimizer:
        device = None

        def _release_graph(self):
            released.append(True)

    result = {"optimizer": Optimizer()}
    release_pipeline_result(result)
    assert released == [True]
    assert result == {}


def test_release_pipeline_result_survives_an_optimizer_that_raises():
    from autoforge.webui.helpers.gpu_memory import release_pipeline_result

    class Optimizer:
        device = None

        def _release_graph(self):
            raise RuntimeError("illegal memory access")

    result = {"optimizer": Optimizer()}
    release_pipeline_result(result)  # must not propagate
    assert result == {}


def test_release_pipeline_result_accepts_nothing():
    from autoforge.webui.helpers.gpu_memory import empty_device_cache, release_pipeline_result

    release_pipeline_result(None)
    release_pipeline_result({})
    empty_device_cache(None)
