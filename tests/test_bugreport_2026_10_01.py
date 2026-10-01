"""Regression tests for the fixes from the 2026-10-01 bug report
(IDs C-*, W-*, F-* refer to that report)."""

import glob
import os
import struct

import numpy as np

from autoforge.Helper.OutputHelper import generate_flatforge_stls, generate_stl


def _stl_volume(path: str) -> float:
    """Signed volume of a binary STL (divergence theorem; outward normals)."""
    data = open(path, "rb").read()
    n = struct.unpack("<I", data[80:84])[0]
    rec = np.dtype([("n", "<f4", 3), ("v", "<f4", (3, 3)), ("a", "<u2")])
    tri = np.frombuffer(data[84:], dtype=rec, count=n)["v"].astype(np.float64)
    return float(np.einsum("ij,ij->i", tri[:, 0], np.cross(tri[:, 1], tri[:, 2])).sum() / 6.0)


def test_c1_flatforge_pieces_tile_the_single_stl(tmp_path):
    """C-1: the per-colour STLs plus the base fill exactly the single STL's
    solid, and with the clear part the full box - no voids where a cell's
    corners straddle a band boundary."""
    rng = np.random.default_rng(0)
    H, W = 12, 15
    heights = rng.integers(0, 10, (H, W))
    # Material 0 recurs in two separate bands.
    dg = np.array([0, 0, 0, 1, 1, 2, 2, 2, 0, 0])
    colors = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=float)
    tds = np.array([1.0, 2.0, 8.0])
    lh, base, size = 0.1, 0.2, 10.0
    generate_flatforge_stls(dg, heights, colors, ["a", "b", "c"], tds, lh, base, "#000000", size, str(tmp_path))
    volumes = {os.path.basename(f): _stl_volume(f) for f in glob.glob(str(tmp_path / "*.stl"))}
    background = sum(v for k, v in volumes.items() if k.startswith("Background"))
    clear = sum(v for k, v in volumes.items() if k.startswith("Clear"))
    colored = sum(v for k, v in volumes.items() if not k.startswith(("Background", "Clear")))

    single_path = str(tmp_path / "single.stl.bin")
    generate_stl(heights * lh, single_path, base, size)
    single = _stl_volume(single_path)
    scale = size / (W - 1)
    box = (W - 1) * (H - 1) * scale * scale * (base + heights.max() * lh)

    assert abs((background + colored) - single) < 1e-6 * single
    assert abs((background + colored + clear) - box) < 1e-6 * box


def test_c2_get_device_makes_the_chosen_gpu_current(monkeypatch):
    """C-2: the resolved CUDA device becomes the thread's current device."""
    import torch

    from autoforge.Helper import DeviceUtils, OtherHelper

    calls = []
    monkeypatch.setattr(DeviceUtils, "cuda_is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "set_device", lambda d: calls.append(d))
    monkeypatch.setattr(OtherHelper, "resolve_device", lambda args=None: torch.device("cuda", 1))
    monkeypatch.setattr(OtherHelper, "describe_device", lambda d: str(d))
    assert OtherHelper.get_device(None) == torch.device("cuda", 1)
    assert calls == [torch.device("cuda", 1)]

    calls.clear()
    DeviceUtils.activate_device(torch.device("cpu"))
    DeviceUtils.activate_device(torch.device("cuda"))  # bare "cuda": already current
    assert calls == []


def test_c3_best_of_counts_seeds_up_and_cleans_temp(tmp_path, monkeypatch):
    """C-3: --best_of N with a fixed seed runs seeds s, s+1, ... and removes
    output/temp after moving the best run's files."""
    import sys

    from autoforge import auto_forge

    seeds = []

    def fake_start(run_args):
        seeds.append(run_args.random_seed)
        with open(os.path.join(run_args.output_folder, "final_loss.txt"), "w") as f:
            f.write(str(len(seeds)))
        return float(10 - len(seeds))  # the last run is best

    monkeypatch.setattr(auto_forge, "start", fake_start)
    monkeypatch.setattr(sys, "argv", [
        "autoforge", "--input_image", "x.png", "--csv_file", "m.csv",
        "--output_folder", str(tmp_path), "--best_of", "3", "--random_seed", "42",
    ])
    auto_forge.main()
    assert seeds == [42, 43, 44]
    assert not os.path.exists(tmp_path / "temp")
    assert open(tmp_path / "final_loss.txt").read() == "3"


def test_c4_project_file_and_swap_instructions_agree_on_a_base_first_layer():
    """C-4: when the first layer is printed in the base filament, the .hfp
    lists no extra entry for it - the same swaps as the instructions."""
    from autoforge.Helper.OutputHelper import extract_filament_swaps, generate_swap_instructions

    disc_global = np.array([2, 2, 0, 0, 1, 1])
    disc_height = np.full((3, 3), 6)
    names = ["red", "blue", "base"]
    indices, sliders = extract_filament_swaps(disc_global, disc_height, 6, base_material_index=2)
    instructions = generate_swap_instructions(disc_global, disc_height, 0.04, 6, 0.24, names, "base", 2)
    swaps = [line for line in instructions if " swap to " in line]
    # Without the trailing repeated entry, one hfp entry per real swap.
    assert indices[:-1] == [0, 1]
    assert len(indices) - 1 == len(swaps)
    # Each entry starts at the layer the instructions name (layer # = slider + 1).
    assert [f"At layer #{s + 1} " in line for s, line in zip(sliders, swaps)] == [True, True]

    # Unchanged when the first layer is not the base filament.
    indices, _ = extract_filament_swaps(disc_global, disc_height, 6, base_material_index=1)
    assert indices[0] == 2


def test_c5_json_td_column_blank_brand_and_short_hex(tmp_path):
    """C-5: a JSON library using "TD" loads, a blank brand doesn't become
    "nan - ...", and #RGB / hash-less colours are accepted."""
    import json
    from argparse import Namespace

    from autoforge.Helper.FilamentHelper import hex_to_rgb, load_materials

    json_path = tmp_path / "lib.json"
    json_path.write_text(json.dumps({"Filaments": [
        {"Brand": "Acme", "Name": "Red", "Color": "#f00", "TD": 2.5},
        {"Brand": "Acme", "Name": "Blue", "Color": "0000FF", "TD": 4},
    ]}))
    colors, tds, names, colors_list = load_materials(Namespace(csv_file="", json_file=str(json_path)))
    assert tds.tolist() == [2.5, 4.0]
    assert colors_list == ["#FF0000", "#0000FF"]
    assert colors[0].tolist() == [1.0, 0.0, 0.0]

    csv_path = tmp_path / "lib.csv"
    csv_path.write_text("Brand, Name, Color, TD\n,White,#FFFFFF,5\nAcme,Black,#000000,1\n")
    _, _, names, _ = load_materials(Namespace(csv_file=str(csv_path), json_file=""))
    assert names == ["White", "Acme - Black"]

    assert hex_to_rgb("#11223344") == hex_to_rgb("#112233")


def test_c6_preview_callback_called_once_and_errors_contained():
    """C-6: a callback that raises TypeError in its own body is called once
    (no retry), its error never escapes, and unsupported keywords are
    dropped for older callbacks."""
    from autoforge.Modules.Optimizer import FilamentOptimizer

    opt = FilamentOptimizer.__new__(FilamentOptimizer)
    opt._current_prune_phase = "Reducing colors"
    calls = []

    def broken(o, percent, phase=None, loss=None):
        calls.append((percent, phase, loss))
        raise TypeError("bug inside the callback")

    opt.preview_callback = broken
    opt._prune_phase_progress(50.0, loss=1.5)  # must not raise
    assert calls == [(50.0, "Reducing colors", 1.5)]

    old_style = []
    opt.preview_callback = lambda o, percent: old_style.append(percent)
    opt._prune_phase_progress(75.0, loss=2.0)
    assert old_style == [75.0]

    any_kwargs = []
    opt.preview_callback = lambda o, percent, **kw: any_kwargs.append(kw)
    opt._prune_phase_progress(10.0, loss=3.0)
    assert any_kwargs == [{"phase": "Reducing colors", "loss": 3.0}]


def test_c7_custom_height_logits_used_and_live_params_untouched():
    """C-7: custom logits are honoured for best=True, never written into
    self.params / best_params, and the offsets are applied exactly once."""
    import torch
    from test_optimizer_training import _make_optimizer

    opt = _make_optimizer(H=8, W=8, labels=np.ones((8, 8)))
    opt.best_params = opt.get_current_parameters()
    opt.best_params["height_offsets"] = torch.full_like(opt.best_params["height_offsets"], 1.0)
    opt.best_seed = 0
    live = opt.params["pixel_height_logits"]
    stored = opt.best_params["pixel_height_logits"]
    custom = torch.full((8, 8), 2.0)

    _, dh = opt.get_discretized_solution(best=True, custom_height_logits=custom)
    expected = torch.round(opt.max_layers * torch.sigmoid(custom + 1.0)).to(torch.int32)
    assert torch.equal(dh, expected)
    _, dh_raw = opt.get_discretized_solution(best=True, custom_height_logits=custom, apply_height_offset=False)
    assert torch.equal(dh_raw, torch.round(opt.max_layers * torch.sigmoid(custom)).to(torch.int32))

    opt.get_discretized_solution(best=False, custom_height_logits=custom)
    assert opt.params["pixel_height_logits"] is live
    assert opt.best_params["pixel_height_logits"] is stored


def test_c8_remove_height_spikes_leaves_its_input_alone():
    """C-8: the input tensor is not modified, even when contiguous."""
    import torch

    from autoforge.Helper.PruningHelper import remove_height_spikes

    z = torch.zeros(5, 5)
    z[2, 2] = 9.0
    before = z.clone()
    cleaned, n = remove_height_spikes(z, threshold_layers=1)
    assert n == 1 and cleaned[2, 2] == 0
    assert torch.equal(z, before)


def test_c9_fine_tune_scores_the_last_step_and_survives_errors(monkeypatch):
    """C-9: the offsets after the final step are scored (and kept when they
    win), and an exception mid-loop restores the original offsets."""
    import pytest
    import torch
    from test_optimizer_training import _make_optimizer

    from autoforge.Helper import PruningHelper
    from autoforge.Loss import LossFunctions

    opt = _make_optimizer(H=8, W=8, labels=np.ones((8, 8)))
    opt.best_params = opt.get_current_parameters()
    opt.best_seed = 0
    orig = opt.best_params["height_offsets"].clone()

    scored = []

    def fake_loss(o, dg, **kw):
        scored.append(o.best_params["height_offsets"].clone())
        # Only the offsets after the last step beat the start.
        return 1.0 if len(scored) < 3 else 0.5

    monkeypatch.setattr(PruningHelper, "_compute_loss_for_heightmap", fake_loss)
    assert opt.fine_tune_height_offsets(num_steps=1) is True
    # pre-loss, the step's starting offsets, then the offsets after the step
    assert len(scored) == 3
    assert torch.equal(opt.best_params["height_offsets"], scored[-1])

    opt.best_params["height_offsets"] = orig.clone()

    def boom(*a, **k):
        raise RuntimeError("CUDA out of memory")

    monkeypatch.setattr(LossFunctions, "loss_fn", boom)
    with pytest.raises(RuntimeError):
        opt.fine_tune_height_offsets(num_steps=3)
    assert torch.equal(opt.best_params["height_offsets"], orig)


def test_c10_graph_replay_check_covers_the_smooth_height_field():
    """C-10: the graph replay validation compares pixel_delta's gradient."""
    from test_optimizer_training import _make_optimizer

    opt = _make_optimizer(H=8, W=8, delta_grid=4)
    assert opt.pixel_delta is not None and opt.pixel_delta.requires_grad
    assert any(p is opt.pixel_delta for p in opt._validated_params())


def test_c11_pruning_limits_raise_clear_errors_not_asserts():
    """C-11: impossible limits raise ValueError with guidance, and the limit
    checks no longer rely on assert (stripped under python -O)."""
    import inspect

    import pytest

    from autoforge.Helper import PruningHelper

    with pytest.raises(ValueError, match="swap limit"):
        PruningHelper.prune_num_swaps(None, -1, 0.01, None)
    with pytest.raises(ValueError, match="color limit"):
        PruningHelper.prune_num_colors(None, -1, 0.01, None)
    for fn in (PruningHelper.prune_num_colors, PruningHelper.prune_num_swaps):
        assert "assert " not in inspect.getsource(fn)


def test_c12_flatforge_file_names_use_the_filament_hex(tmp_path):
    """C-12: a #808080 filament's STL is named ..._808080, not ..._7f7f7f."""
    from autoforge.Helper.FilamentHelper import hex_to_rgb

    grey = hex_to_rgb("#808080")
    heights = np.full((4, 4), 2)
    generate_flatforge_stls(
        np.array([0, 0]), heights, np.array([grey, [1.0, 1.0, 1.0]]), ["grey", "clear"],
        np.array([1.0, 9.0]), 0.1, 0.2, "#000000", 10.0, str(tmp_path),
    )
    names = os.listdir(tmp_path)
    assert "grey_808080.stl" in names


def test_c13_mask_editor_uses_unicode_safe_image_io(tmp_path, monkeypatch):
    """C-13: the mask editor reads/writes through ImageHelper (non-ASCII paths)."""
    import inspect
    import sys

    import cv2

    from autoforge import priority_mask_editor as pme
    from autoforge.Helper.ImageHelper import imread, imwrite

    src = inspect.getsource(pme)
    assert "cv2.imread(" not in src and "cv2.imwrite(" not in src
    # A non-ASCII path round-trips through the helpers the editor now uses.
    path = str(tmp_path / "maske_ä.png")
    imwrite(path, np.full((4, 4), 200, np.uint8))
    assert imread(path, cv2.IMREAD_GRAYSCALE)[0, 0] == 200


def test_c14_matmul_precision_set_without_a_string_version_check():
    """C-14: no string comparison of torch.__version__."""
    import inspect

    from autoforge import auto_forge

    assert 'torch.__version__ >=' not in inspect.getsource(auto_forge)


def test_w1_webui_run_settings_match_the_cli_defaults():
    """W-1: every option the pipeline reads through getattr has the CLI's
    default in a webui run (height_assign and delta_grid were missing and
    fell back to "off"). Only the deliberate webui overrides differ."""
    import pathlib
    import re

    from autoforge.auto_forge import cli_defaults
    from autoforge.webui.helpers.pipeline_runner import _WEBUI_OVERRIDES, _make_settings

    cli = cli_defaults()
    webui = vars(_make_settings({}))
    src = pathlib.Path(__file__).resolve().parents[1] / "src" / "autoforge"
    read = set()
    for path in src.rglob("*.py"):
        read |= set(re.findall(r'getattr\((?:self\.|optimizer\.)*args,\s*"(\w+)"', path.read_text()))
    checked = sorted(k for k in read if k in cli and k not in _WEBUI_OVERRIDES)
    assert "height_assign" in checked and "delta_grid" in checked
    mismatched = {k: (webui.get(k), cli[k]) for k in checked if webui.get(k) != cli[k]}
    assert mismatched == {}
    assert webui["height_assign"] is True and webui["delta_grid"] == 4


def _webui_result(tmp_path, H=8, W=8, scale=2, **settings):
    """A small trained webui pipeline result (CPU) whose output resolution
    is ``scale`` times the processing resolution."""
    import torch

    from autoforge.Modules.Optimizer import FilamentOptimizer
    from autoforge.webui.helpers.pipeline_runner import _make_settings

    torch.manual_seed(0)
    np.random.seed(0)
    args = _make_settings({
        "max_layers": 6, "iterations": 6, "discrete_check": 1, "output_folder": str(tmp_path),
        "stl_output_size": 10, "cuda_graph": False, "intermediate_search_interval": 0, **settings,
    })
    yy, xx = np.mgrid[0:H * scale, 0:W * scale].astype(np.float32)
    full = torch.tensor(np.stack([255 * xx / (W * scale), 255 * yy / (H * scale), 128 + 0 * xx], -1))
    proc = full[::scale, ::scale].contiguous()
    colors = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    tds = torch.tensor([1.0, 2.0, 4.0])
    init_full = np.zeros((H * scale, W * scale), np.float32)
    labels_full = np.ones((H * scale, W * scale), np.int32)
    opt = FilamentOptimizer(
        args, proc, init_full[::scale, ::scale].copy(), labels_full[::scale, ::scale].copy(),
        np.random.randn(6, 3).astype(np.float32), colors, tds, torch.zeros(3), torch.device("cpu"), None,
    )
    for _ in range(args.iterations):
        opt.step(record_best=True)
    filaments = [
        {"color": c, "td": t, "name": f"A - {n}", "brand": "A", "short_name": n, "uuid": n}
        for c, t, n in (("#FF0000", 1.0, "r"), ("#00FF00", 2.0, "g"), ("#0000FF", 4.0, "b"))
    ]
    return {
        "optimizer": opt, "args": args, "device": torch.device("cpu"),
        "material_colors_np": colors.numpy(), "material_TDs_np": tds.numpy(),
        "material_names": [f["name"] for f in filaments], "material_uuids": [f["uuid"] for f in filaments],
        "active_filaments": filaments, "colors_list": [f["color"] for f in filaments],
        "background": torch.zeros(3), "alpha": None, "output_target": full,
        "focus_map_full": None, "focus_map_proc": None,
        "pixel_height_logits_init": init_full, "pixel_height_labels": labels_full,
        "processing_pixel_height_logits_init": init_full[::scale, ::scale].copy(),
    }


def test_w1_webui_export_assigns_heights_at_output_resolution(tmp_path):
    """W-1: with height assignment on (now the webui default), the export
    assigns every pixel's height at the output resolution for the best
    stack, like the CLI, and drops the processing-resolution height field."""
    import torch

    from autoforge.Helper.HeightAssign import heights_to_logits
    from autoforge.webui.helpers.pipeline_runner import export_results

    result = _webui_result(tmp_path)
    opt = result["optimizer"]
    assert opt.height_assign and opt.pixel_delta is not None
    export_results(result)
    assert opt.pixel_delta is None
    assert tuple(opt.best_params["pixel_height_logits"].shape) == (16, 16)
    assert float(opt.best_params["height_offsets"].abs().sum()) == 0.0
    dg, dh = opt.get_discretized_solution(best=True)
    # The export's later (autocast) loss calls cache a reduced-precision Lab
    # copy of the target; the assignment itself ran before that.
    if hasattr(opt.target, "_af_lab_cache"):
        del opt.target._af_lab_cache
    expected = opt.assigned_heights(dg, heights_to_logits(dh.float(), opt.max_layers))
    assert torch.equal(dh.long(), expected.long())
    assert os.path.exists(tmp_path / "final_model.stl")


def test_w2_init_rounds_default_to_one_and_extra_rounds_are_skipped(monkeypatch):
    """W-2: the webui no longer asks for 16/32 identical init rounds, and an
    old persisted setting > 1 doesn't spawn worker processes for them."""
    from autoforge.Helper.Heightmaps import FastTSPHeightMap as ft
    from autoforge.webui.helpers.pipeline_runner import _DEFAULTS
    from autoforge.webui.models import OptimizationSettings

    assert OptimizationSettings().num_init_rounds == 1
    assert _DEFAULTS["num_init_rounds"] == 1

    calls = []
    real = ft.init_height_map

    def counting(*a, **k):
        calls.append(k.get("rank_quality"))
        return real(*a, **k)

    monkeypatch.setattr(ft, "init_height_map", counting)
    yy, xx = np.mgrid[0:24, 0:24]
    img = np.stack([xx * 10, yy * 10, (xx + yy) * 5], -1).astype(np.uint8)
    ft.run_init_threads(img, 6, 0.2, (0.1, 0.1, 0.1), random_seed=0, num_threads=4, num_runs=8, cluster_layers=4)
    assert calls == [False]


def test_w3_webui_training_releases_the_cuda_graph(tmp_path, monkeypatch):
    """W-3: the webui training loop frees the captured graphs afterwards."""
    from autoforge.webui.helpers import pipeline_runner

    result = _webui_result(tmp_path)
    opt = result["optimizer"]
    released = []
    monkeypatch.setattr(opt, "release_cuda_graph", lambda: released.append(True))
    pipeline_runner._run_optimization_loop(opt, result["args"], result["device"])
    assert released == [True]


def test_w4_webui_runs_the_cli_end_of_training_pass(tmp_path, monkeypatch):
    """W-4: run_pipeline finishes training like the CLI: a height
    re-assignment and 60 more discrete checks of the final state."""
    import torch

    from autoforge.Modules.Optimizer import FilamentOptimizer
    from autoforge.webui.helpers import pipeline_runner

    result = _webui_result(tmp_path)
    opt = result["optimizer"]
    counts = {"assign": 0, "check": 0}
    real_check = FilamentOptimizer._maybe_update_best_discrete
    monkeypatch.setattr(opt, "assign_heights_step", lambda: counts.__setitem__("assign", counts["assign"] + 1))
    monkeypatch.setattr(opt, "_maybe_update_best_discrete",
                        lambda *a: (counts.__setitem__("check", counts["check"] + 1), real_check(opt, *a)))
    monkeypatch.setattr(pipeline_runner, "build_pipeline_state", lambda *a, **k: dict(result))
    monkeypatch.setattr(pipeline_runner, "_run_optimization_loop", lambda *a, **k: None)
    state = pipeline_runner.run_pipeline("x.png", result["active_filaments"], str(tmp_path), {})
    assert counts == {"assign": 1, "check": 60}
    assert state["cancelled"] is False

    # A cancelled run skips it.
    import threading

    ev = threading.Event()
    ev.set()
    counts.update(assign=0, check=0)
    pipeline_runner.run_pipeline("x.png", result["active_filaments"], str(tmp_path), {}, cancel_event=ev)
    assert counts == {"assign": 0, "check": 0}


def test_w8_export_zip_follows_slider_edits(tmp_path, monkeypatch):
    """W-8: with slider edits, the bundle carries the edited image and mesh
    and swap instructions / .hfp for the edited stack; without a live result
    the stale ones are left out."""
    import json
    import zipfile

    from autoforge.webui.api import outputs

    result = _webui_result(tmp_path / "run")
    job_dir = tmp_path / "job"
    job_dir.mkdir()
    for name in ("final_model.stl", "final_model.png", "final_model_colored.ply",
                 "swap_instructions.txt", "project_file.hfp", "final_loss.txt"):
        (job_dir / name).write_text("optimizer " + name)
    (job_dir / outputs.EDITED_PNG).write_text("edited png")
    (job_dir / outputs.EDITED_PLY).write_text("edited ply")

    class _Svc:
        def __init__(self, res):
            self.res = res

        def get_pipeline_result(self, job_id):
            return self.res

    class _Filaments:
        def list(self):
            return []

    import autoforge.webui.services.filament_service as fs

    monkeypatch.setattr(fs, "get_filament_service", lambda: _Filaments())
    # The user moved everything to the blue filament.
    project = {
        "colorSliders": [{"layer": 6, "td": 4.0, "filament_uuid": "b", "enabled": True}],
        # Frontend Filament records (name without the brand).
        "activeFilaments": [{"uuid": f["uuid"], "brand": "A", "name": f["short_name"], "color": f["color"], "td": f["td"]}
                            for f in result["active_filaments"]],
    }

    def bundle(svc_result):
        monkeypatch.setattr(outputs, "get_optimization_service", lambda: _Svc(svc_result))
        edit_dir = tmp_path / ("edit" if svc_result else "edit2")
        replace = outputs._edited_replacements("job", str(job_dir), project, str(edit_dir))
        zpath = tmp_path / "out.zip"
        outputs._write_export_zip(str(zpath), str(job_dir), "", project, replace)
        with zipfile.ZipFile(zpath) as zf:
            return {n: zf.read(n) for n in zf.namelist()}

    files = bundle(result)
    assert files["final_model.png"] == b"edited png"
    assert files["final_model_colored.ply"] == b"edited ply"
    instructions = files["swap_instructions.txt"].decode()
    assert "A - b" in instructions and "A - r" not in instructions and "A - g" not in instructions
    hfp = json.loads(files["project_file.hfp"])
    assert {f["Name"] for f in hfp["filament_set"]} <= {"b", "Background"}

    files = bundle(None)
    assert "swap_instructions.txt" not in files and "project_file.hfp" not in files
    assert files["final_model.png"] == b"edited png"
    assert files["final_model.stl"] == b"optimizer final_model.stl"


def test_w9_settings_ranges_are_validated():
    """W-9: invalid pruning limits and inconsistent settings are rejected
    with a message instead of failing (or silently misbehaving) in a job."""
    import pytest
    from pydantic import ValidationError

    from autoforge.webui.models import OptimizationSettings, PruningSettings

    for bad in ({"pruning_max_swaps": -1}, {"pruning_max_colors": 0}, {"pruning_max_layer": 0}):
        with pytest.raises(ValidationError):
            PruningSettings(**bad)
    PruningSettings(pruning_max_swaps=0, pruning_max_colors=1, pruning_max_layer=1)

    for bad in ({"final_tau": 2.0, "init_tau": 1.0}, {"min_layers": 80, "pruning_max_layer": 75},
                {"spike_threshold_layers": 0}, {"cap_layers": -1}):
        with pytest.raises(ValidationError):
            OptimizationSettings(**bad)
    OptimizationSettings()


def test_w14_duplicate_import_rows_merge_and_uuidless_filaments_get_one(tmp_path):
    """W-14: repeated brand+name rows in one import become one entry; active
    filaments without a uuid don't overwrite each other."""
    from autoforge.webui.models import Filament
    from autoforge.webui.services.filament_service import FilamentService

    svc = FilamentService(library_path=str(tmp_path))
    imported = svc.import_csv("Brand,Name,Color,TD\nAcme,Red,#ff0000,1\nAcme,Red,#ee0000,2\nAcme,Blue,#0000ff,3\n")
    assert len(svc.list()) == 2
    red = [f for f in svc.list() if f.name == "Red"][0]
    assert red.color == "#ee0000" and red.td == 2.0  # the later row wins
    assert len({f.uuid for f in imported}) == 2

    a = svc.set_active(Filament(name="A", color="#111111"))
    b = svc.set_active(Filament(name="B", color="#222222"))
    assert a.uuid and b.uuid and a.uuid != b.uuid
    assert {f.name for f in svc.get_active()} == {"A", "B"}
    replaced = svc.replace_active([Filament(name="C"), Filament(name="D")])
    assert len({f.uuid for f in replaced}) == 2 and all(f.uuid for f in replaced)


def test_w16_slider_td_null_and_vertex_cache_identity():
    """W-16: an unknown filament with td null renders with the default TD;
    the top-vertex cache is invalidated by a new alpha array even if it
    reuses the old one's id()."""
    from autoforge.webui.helpers import slider_render

    colors, tds = slider_render._resolve_slider_materials([{"filament_uuid": "gone", "td": None}], {})
    assert tds.tolist() == [5.0]

    calls = []
    real = slider_render.top_vertex_pixel_indices
    slider_render.top_vertex_pixel_indices = lambda h, w, a: (calls.append(1), real(h, w, a))[1]
    try:
        alpha_a = np.full((4, 4), 255, np.uint8)
        result = {"alpha": alpha_a, "alpha_proc": None}
        slider_render._cached_top_vertex_pixels(result, (4, 4), alpha_a)
        slider_render._cached_top_vertex_pixels(result, (4, 4), alpha_a)
        assert len(calls) == 1
        alpha_b = np.zeros((4, 4), np.uint8)
        result["alpha"] = alpha_b
        idx = slider_render._cached_top_vertex_pixels(result, (4, 4), alpha_b)
        assert len(calls) == 2 and len(idx) == 0
    finally:
        slider_render.top_vertex_pixel_indices = real


def test_w17_edge_bleed_is_per_thread():
    """W-17: a render/export on another thread can't change the bleed a
    running job composites with; threads that never set one use the last
    value set anywhere."""
    import threading

    from autoforge.Helper import OptimizerHelper as oh

    oh.set_edge_bleed(0.25)  # this thread: "the training job"
    seen = {}

    def other_job():
        oh.set_edge_bleed(0.6)
        seen["other"] = oh.get_edge_bleed()

    t = threading.Thread(target=other_job)
    t.start()
    t.join()
    assert seen["other"] == 0.6
    assert oh.get_edge_bleed() == 0.25

    def helper():
        seen["helper"] = oh.get_edge_bleed()

    t = threading.Thread(target=helper)
    t.start()
    t.join()
    assert seen["helper"] == 0.6
    oh.set_edge_bleed(oh.EDGE_BLEED)
