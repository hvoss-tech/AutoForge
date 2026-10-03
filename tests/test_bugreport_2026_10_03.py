"""Regression tests for the core-library fixes of the 2026-10-03 bug report
(BUG_REPORT.md: H1, H2, M1-M4, L1-L15)."""

import argparse
import ast
import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import numpy as np
import pytest
import torch

REPO = Path(__file__).resolve().parents[1]
SRC = REPO / "src" / "autoforge"


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _args(**over):
    from autoforge.auto_forge import cli_defaults

    ns = argparse.Namespace(**cli_defaults())
    ns.input_image = "x.png"
    ns.visualize = False
    ns.output_folder = "."
    ns.max_layers = 30
    ns.layer_height = 0.04
    ns.iterations = 20
    ns.cuda_graph = False
    for k, v in over.items():
        setattr(ns, k, v)
    return ns


def _optimizer(device="cpu", H=12, W=10, layers=30, n_mat=4, **over):
    from autoforge.Modules.Optimizer import FilamentOptimizer

    g = torch.Generator().manual_seed(0)
    target = (torch.rand(H, W, 3, generator=g) * 255).to(device)
    logits = np.random.default_rng(0).normal(0.0, 1.5, (H, W)).astype(np.float32)
    labels = np.random.default_rng(1).integers(0, 4, (H, W)).astype(np.int32)
    global_logits = np.random.default_rng(2).normal(0.0, 1.0, (layers, n_mat)).astype(np.float32)
    colors = torch.rand(n_mat, 3, generator=g).to(device)
    tds = (torch.rand(n_mat, generator=g) * 4 + 0.5).to(device)
    args = _args(max_layers=layers, **over)
    opt = FilamentOptimizer(
        args, target, logits, labels, global_logits, colors, tds, torch.zeros(3, device=device),
        torch.device(device), None,
    )
    opt.best_params = opt.get_current_parameters()
    opt.best_seed = 0
    return opt


# ---------------------------------------------------------------------------
# H1: the webui never runs with the matplotlib visualization
# ---------------------------------------------------------------------------


def test_h1_webui_settings_force_visualize_off():
    from autoforge.webui.helpers.pipeline_runner import _make_settings

    ns = _make_settings({"visualize": True, "max_layers": 20})
    assert ns.visualize is False
    assert ns.max_layers == 20


# ---------------------------------------------------------------------------
# H2: text files are written as UTF-8, whatever the platform encoding
# ---------------------------------------------------------------------------


def _text_mode_opens_without_encoding(path: Path):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    bad = []
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "open"):
            continue
        mode = None
        if len(node.args) > 1 and isinstance(node.args[1], ast.Constant):
            mode = node.args[1].value
        for kw in node.keywords:
            if kw.arg == "mode" and isinstance(kw.value, ast.Constant):
                mode = kw.value.value
        if mode is not None and "b" in mode:
            continue
        if not any(kw.arg == "encoding" for kw in node.keywords):
            bad.append(node.lineno)
    return bad


def test_h2_every_text_open_names_its_encoding():
    offenders = {}
    for path in SRC.rglob("*.py"):
        bad = _text_mode_opens_without_encoding(path)
        if bad:
            offenders[str(path.relative_to(REPO))] = bad
    assert offenders == {}


def test_h2_edited_export_with_non_ascii_names_under_an_ascii_locale(tmp_path):
    """The webui's export writes materials.csv and swap_instructions.txt and
    reads the CSV back with pandas (UTF-8). Run it in a process whose locale
    encoding is ASCII - like cp1252 on Windows, it cannot hold every
    character a filament name may have."""
    script = textwrap.dedent(
        """
        import argparse, sys
        import numpy as np, torch
        from autoforge.webui.helpers.edited_export import write_edited_outputs

        class Opt:
            max_layers = 4
            def get_discretized_solution(self, best=True):
                return torch.tensor([0, 1, 1, 0]), torch.full((3, 3), 4)

        result = {
            "optimizer": Opt(),
            "args": argparse.Namespace(background_height=0.24, layer_height=0.04, max_layers=4,
                                       background_color="#000000", stl_output_size=20),
            "output_target": torch.zeros(3, 3, 3),
            "material_uuids": ["a", "b"],
        }
        sliders = [
            {"layer": 2, "enabled": True, "filament_uuid": "a", "td": 2.0},
            {"layer": 4, "enabled": True, "filament_uuid": "b", "td": 3.0},
        ]
        filaments = {
            "a": {"brand": "\\u00dcnbrand", "name": "Caf\\u00e9 Cr\\u00e8me", "color": "#112233"},
            "b": {"brand": "", "name": "Gr\\u00fcn \\u2192 Blau", "color": "#445566"},
        }
        out = write_edited_outputs(result, sliders, filaments, sys.argv[1])
        print(open(out["swap_instructions.txt"], encoding="utf-8").read().encode("unicode_escape").decode())
        """
    )
    env = dict(os.environ, LC_ALL="C", LANG="C", PYTHONUTF8="0", PYTHONCOERCECLOCALE="0",
               PYTHONPATH=str(REPO / "src"))
    script_path = tmp_path / "run.py"
    script_path.write_text(script, encoding="ascii")
    proc = subprocess.run([sys.executable, str(script_path), str(tmp_path)], env=env, capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr
    assert "Gr\\xfcn" in proc.stdout


# ---------------------------------------------------------------------------
# M1: label 0 is the background cluster
# ---------------------------------------------------------------------------


def test_m1_background_first_labels_swaps_ids():
    from autoforge.Helper.Heightmaps.FastTSPHeightMap import background_first_labels

    labels = np.array([[0, 1, 2], [2, 1, 0]])
    out = background_first_labels(labels, 2)
    assert out.tolist() == [[2, 1, 0], [0, 1, 2]]
    assert background_first_labels(labels, 0) is labels


def test_m1_kmeans_init_gives_the_background_label_zero():
    from autoforge.Helper.Heightmaps.FastTSPHeightMap import run_init_threads

    img = np.zeros((24, 24, 3), dtype=np.uint8)
    img[:, :8] = (250, 250, 250)  # white background (the base colour below)
    img[:, 8:16] = (200, 30, 30)
    img[:, 16:] = (30, 30, 200)
    _logits, _gl, labels = run_init_threads(
        img, 6, 0.04, (1.0, 1.0, 1.0), random_seed=3, num_threads=1, cluster_layers=4,
    )
    assert labels[:, :8].max() == 0 and labels[:, :8].min() == 0
    assert labels[:, 8:].min() > 0


# ---------------------------------------------------------------------------
# M2: the CLI carries trained per-pixel heights to the output resolution
# ---------------------------------------------------------------------------


def test_m2_best_heights_keeps_the_per_pixel_changes():
    from autoforge.auto_forge import best_heights_at_output_res
    from autoforge.webui.helpers import pipeline_runner

    assert pipeline_runner._best_heights_at_output_res is best_heights_at_output_res
    proc_init = np.zeros((2, 2), dtype=np.float32)
    best = torch.tensor([[1.0, 0.0], [0.0, -2.0]])
    full = torch.zeros(4, 4)
    out = best_heights_at_output_res(best, proc_init, full, None)
    assert out[0, 0] == 1.0 and out[3, 3] == -2.0 and out[0, 3] == 0.0


def test_m2_cli_export_no_longer_discards_them():
    src = (SRC / "auto_forge.py").read_text(encoding="utf-8")
    assert "_with_delta" not in src
    assert "processing_pixel_height_logits_init=processing_pixel_height_logits_init" in src


# ---------------------------------------------------------------------------
# M3: one scorer for baseline and candidates
# ---------------------------------------------------------------------------


def test_m3_swap_positions_baseline_uses_the_candidate_scorer(monkeypatch):
    from autoforge.Helper import PruningHelper

    opt = _optimizer()
    calls = []
    real = PruningHelper._eval_candidates_batch

    def recording(o, cands, *a, **k):
        calls.append(len(cands))
        return real(o, cands, *a, **k)

    def forbidden(*a, **k):
        raise AssertionError("baseline scored with a different composite")

    monkeypatch.setattr(PruningHelper, "_eval_candidates_batch", recording)
    monkeypatch.setattr(PruningHelper, "composite_image_disc", forbidden)
    PruningHelper.optimise_swap_positions(opt, max_passes=1)
    assert calls and calls[0] == 1  # the baseline: the current stack alone


def test_m3_layer_pruning_baseline_uses_the_candidate_scorer(monkeypatch):
    from autoforge.Helper import PruningHelper

    opt = _optimizer(layers=8)

    def forbidden(*a, **k):
        raise AssertionError("baseline from get_initial_loss (bf16) while candidates are fp32")

    monkeypatch.setattr(PruningHelper, "get_initial_loss", forbidden)
    _params, loss, n = PruningHelper.prune_redundant_layers(opt, None, 0, 6, fast=True)
    assert n <= 6 and np.isfinite(loss)


# ---------------------------------------------------------------------------
# M4: the default materials ship with the package
# ---------------------------------------------------------------------------


def test_m4_default_materials_are_package_data():
    import tomllib

    data = tomllib.loads((REPO / "pyproject.toml").read_text(encoding="utf-8"))
    assert "default_materials.csv" in data["tool"]["setuptools"]["package-data"]["autoforge"]
    req = (REPO / "requirements.txt").read_text(encoding="utf-8").split()
    assert sorted(req[req.index("pyproject.toml.") + 1:]) == sorted(data["project"]["dependencies"])


# ---------------------------------------------------------------------------
# L1-L15
# ---------------------------------------------------------------------------


def test_l1_refine_stack_palette_runs():
    from autoforge.Helper.PixelHeightRefine import refine_stack_palette

    opt = _optimizer(layers=6)
    for window in (0, 2):
        dg = refine_stack_palette(opt, sweeps=1, window=window)
        assert dg.shape == (6,)


def test_l2_prune_fireflies_never_accepts_a_worse_map(monkeypatch):
    from autoforge.Helper import PruningHelper

    class Fake:
        target = torch.full((4, 4, 3), 100.0)
        focus_map = None
        alpha = None

        def __init__(self):
            self.best_params = {"pixel_height_logits": torch.zeros(4, 4)}

        def get_best_discretized_image(self, custom_height_logits=None):
            # Every cleaned map is ~2% worse than the original.
            off = 0.0 if torch.equal(custom_height_logits, torch.zeros(4, 4)) else 1.0
            return torch.full((4, 4, 3), 150.0 + off)

    monkeypatch.setattr(PruningHelper, "remove_outlier_pixels", lambda h, threshold: h + 1.0)
    fake = Fake()
    out = PruningHelper.prune_fireflies(fake, start_threshold=1.0)
    assert torch.equal(out, torch.zeros(4, 4))
    assert torch.equal(fake.best_params["pixel_height_logits"], torch.zeros(4, 4))


def test_l3_live_discretization_uses_the_global_tau(monkeypatch):
    opt = _optimizer(layers=6)
    seen = {}
    real = opt.discretize_solution

    def spy(params, tau_global, *a, **k):
        seen["tau"] = tau_global
        return real(params, tau_global, *a, **k)

    monkeypatch.setattr(opt, "discretize_solution", spy)
    monkeypatch.setattr(opt, "_get_tau", lambda: (0.7, 0.3))
    opt.get_discretized_solution(best=False)
    assert seen["tau"] == 0.3


def test_l4_negative_seed_reads_as_zero():
    opt = _optimizer(layers=6)
    a, _ = opt.discretize_solution(opt.best_params, 0.01, opt.h, opt.max_layers, rng_seed=-1)
    b, _ = opt.discretize_solution(opt.best_params, 0.01, opt.h, opt.max_layers, rng_seed=0)
    assert torch.equal(a, b)


def test_l5_loss_weights_follow_an_in_place_mask_edit():
    from autoforge.Helper.FusedComposite import loss_weights

    target = torch.rand(5, 5, 3) * 255
    mask = torch.zeros(5, 5)
    _, w0, _ = loss_weights(target, mask, None)
    mask[0, 0] = 1.0
    _, w1, _ = loss_weights(target, mask, None)
    assert float(w1[0, 0]) > float(w0[0, 0])
    other = torch.ones(5, 5)
    _, w2, _ = loss_weights(target, other, None)
    assert torch.allclose(w2, torch.ones(5, 5))


def test_l6_matmul_precision_restored_after_nested_kmeans():
    from autoforge.Helper.Heightmaps import _cluster

    before = torch.get_float32_matmul_precision()
    with _cluster._highest_matmul_precision():
        with _cluster._highest_matmul_precision():
            assert torch.get_float32_matmul_precision() == "highest"
        assert torch.get_float32_matmul_precision() == "highest"
    assert torch.get_float32_matmul_precision() == before
    _cluster.kmeans(np.random.default_rng(0).random((64, 3)), 4, device=torch.device("cpu"))
    assert torch.get_float32_matmul_precision() == before


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA graphs")
def test_l8_update_graph_is_captured_with_frozen_heights():
    opt = _optimizer(device="cuda", H=40, W=40, layers=20, cuda_graph=True, height_assign=True)
    opt.best_params = None
    assert not opt.height_offsets.requires_grad  # --height_assign freezes them
    for i in range(12):
        opt.step(record_best=(i % 5 == 0))
    assert opt._graph is not None
    assert opt._upd_graph is not None
    opt.release_cuda_graph()


def test_l10_spike_removal_prints_nothing(capsys):
    from autoforge.Helper.PruningHelper import remove_height_spikes

    z = torch.zeros(7, 7)
    z[3, 3] = 9
    _out, n = remove_height_spikes(z, threshold_layers=1)
    assert n == 1
    assert capsys.readouterr().out == ""


def test_l11_flatforge_one_pixel_and_duplicate_names(tmp_path):
    from autoforge.Helper.OutputHelper import _create_flatforge_box_mesh, generate_flatforge_stls

    one = np.ones((1, 1))
    assert _create_flatforge_box_mesh(one, one * 0, 0.0, 10.0, one > 0, one > 0) is None

    heights = np.full((6, 6), 4)
    files = generate_flatforge_stls(
        np.array([0, 0, 1, 1]), heights,
        np.array([[1.0, 0, 0], [1.0, 0, 0], [0, 0, 1.0]]), ["Same", "Same", "Clear"],
        np.array([1.0, 2.0, 50.0]), 0.04, 0.24, "#000000", 20, str(tmp_path),
    )
    color_files = [f for f in files if os.path.basename(f).startswith("Same")]
    assert len(color_files) == 2 and len(set(color_files)) == 2
    assert all(os.path.exists(f) for f in color_files)


def test_l12_invalid_material_files_exit_with_a_message(tmp_path, capsys):
    from autoforge.Helper.FilamentHelper import load_materials

    bad = tmp_path / "m.csv"
    bad.write_text("Brand,Name,Color,TD\nAcme,Red,#ff0000,\nAcme,Blue,nope,2\n", encoding="utf-8")
    with pytest.raises(SystemExit):
        load_materials(argparse.Namespace(csv_file=str(bad), json_file=""))
    err = capsys.readouterr().err
    assert "TD" in err and "nope" in err

    good = tmp_path / "m.json"
    good.write_text(json.dumps([{"Brand": "A", "Name": "R", "Color": "#ff0000", "TD": 2}]), encoding="utf-8")
    colors, tds, names, hexes = load_materials(argparse.Namespace(csv_file="", json_file=str(good)))
    assert hexes == ["#FF0000"] and tds.tolist() == [2.0]


def test_l13_swatch_colors_with_a_hash():
    from autoforge.Helper.FilamentHelper import extract_colors_from_swatches

    colors, _tds, _names, hexes = extract_colors_from_swatches(
        [{"td": 1.0, "manufacturer": {"name": "B"}, "color_name": "R", "hex_color": "#ff0000"}]
    )
    assert hexes == ["#FF0000"]
    assert np.allclose(colors[0], [1.0, 0.0, 0.0])


def test_l14_plot_results_imports_without_running():
    import importlib

    mod = importlib.import_module("autoforge.Init_comparer.plot_results")
    assert callable(mod.main)


def test_l15_zero_layer_height_is_reported(capsys):
    from autoforge.Helper.OtherHelper import perform_basic_check

    with pytest.raises(SystemExit):
        perform_basic_check(argparse.Namespace(background_height=0.24, layer_height=0.0))
    assert "layer_height" in capsys.readouterr().err
