"""Colour/swap limits held by the optimizer itself (--constrained_opt).

Covers the feasible projection (ConstraintHelper), the constrained optimizer
(limits, Lagrangian, feasible snapshots, feasible local search) and the CLI
paths the prune harness uses (--constrained_opt, --prune_sweep).
"""
import itertools
import json
import types

import numpy as np
import pytest
import torch

from autoforge.Helper.ConstraintHelper import (
    _palette_masks,
    count_colors_swaps,
    feasible,
    neighbour_stacks,
    project_assignment,
    top_palettes,
)
from autoforge.Modules.Optimizer import FilamentOptimizer


def _brute_force(score, max_colors, max_swaps):
    L, M = score.shape
    best, best_v = None, -float("inf")
    for dg in itertools.product(range(M), repeat=L):
        t = torch.tensor(dg)
        if not feasible(t, max_colors, max_swaps):
            continue
        v = score[torch.arange(L), t].sum().item()
        if v > best_v:
            best, best_v = t, v
    return best, best_v


def _value(score, dg):
    return score[torch.arange(score.shape[0]), dg].sum().item()


# --------------------------------------------------------------------------
# projection


@pytest.mark.parametrize("max_colors,max_swaps", [(None, 0), (None, 1), (None, 3), (1, None), (2, None), (3, None)])
def test_projection_is_exact_for_a_single_limit(max_colors, max_swaps):
    gen = torch.Generator().manual_seed(1)
    for _ in range(25):
        score = torch.randn(6, 4, generator=gen)
        out = project_assignment(score, max_colors, max_swaps)
        assert feasible(out, max_colors, max_swaps)
        _, best_v = _brute_force(score, max_colors, max_swaps)
        assert _value(score, out) == pytest.approx(best_v, abs=1e-4)


def test_projection_with_both_limits_is_feasible():
    gen = torch.Generator().manual_seed(2)
    for _ in range(25):
        score = torch.randn(6, 4, generator=gen)
        out = project_assignment(score, 2, 2)
        assert feasible(out, 2, 2)


def test_projection_without_limits_is_the_argmax():
    score = torch.randn(10, 5)
    assert torch.equal(project_assignment(score, None, None), score.argmax(-1))


def test_projection_is_batched():
    score = torch.randn(4, 20, 6)
    out = project_assignment(score, None, 3)
    assert out.shape == (4, 20)
    for b in range(4):
        assert torch.equal(out[b], project_assignment(score[b], None, 3))


def test_projection_survives_nan_scores():
    # A NaN score used to backtrack the Viterbi into a swap count of -1
    # (device-side assert on CUDA, index error on CPU).
    score = torch.randn(30, 5)
    score[3, 2] = float("nan")
    score[10:12] = float("nan")
    out = project_assignment(score, None, 4)
    assert feasible(out, None, 4)
    assert ((out >= 0) & (out < 5)).all()


def test_top_palettes_ranks_the_best_palette_first():
    score = torch.randn(12, 7)
    tops = top_palettes(score, 3, 5)
    assert tops.shape == (5, 7)
    assert (tops.sum(1) == 3).all()
    assert torch.equal(tops[0], _palette_masks(score.unsqueeze(0), 3)[0])
    neg = torch.tensor(-1e9)
    vals = [torch.where(m, score, neg).amax(-1).sum().item() for m in tops]
    assert vals == sorted(vals, reverse=True)


# --------------------------------------------------------------------------
# neighbourhood


@pytest.mark.parametrize("max_colors,max_swaps", [(None, None), (3, None), (None, 4), (3, 3)])
def test_neighbours_are_feasible_distinct_and_new(max_colors, max_swaps):
    dg = torch.tensor([0, 0, 1, 1, 1, 2, 2, 0])
    assert feasible(dg, max_colors, max_swaps)
    cands = neighbour_stacks(dg, 5, max_colors, max_swaps)
    assert cands
    seen = set()
    for c in cands:
        assert feasible(c, max_colors, max_swaps)
        t = tuple(c.tolist())
        assert t != tuple(dg.tolist())
        assert t not in seen
        seen.add(t)


def test_neighbours_include_band_merges_and_boundary_shifts():
    dg = torch.tensor([0, 0, 1, 1, 1, 2, 2])
    cands = {tuple(c.tolist()) for c in neighbour_stacks(dg, 3, None, 2)}
    assert (0, 0, 0, 0, 0, 2, 2) in cands  # middle band merged into the one below
    assert (0, 0, 0, 1, 1, 2, 2) in cands  # boundary shifted up by one layer
    assert (0, 0, 2, 2, 2, 2, 2) in cands  # colour merged everywhere


def test_count_colors_swaps():
    assert count_colors_swaps(torch.tensor([3, 3, 1, 1, 3])) == (2, 2)
    assert count_colors_swaps(torch.tensor([0])) == (1, 0)


# --------------------------------------------------------------------------
# optimizer


def _args(**overrides):
    args = types.SimpleNamespace(
        max_layers=12,
        layer_height=0.04,
        learning_rate=0.015,
        final_tau=0.01,
        init_tau=1.0,
        iterations=60,
        warmup_fraction=1.0,
        learning_rate_warmup_fraction=0.01,
        tensorboard=False,
        run_name="",
        visualize=False,
        disable_visualization_for_gradio=1,
        output_folder=".",
        background_height=0.24,
        cuda_graph=False,
        flatforge=False,
        constrained_opt=True,
        pruning_max_colors=4,
        pruning_max_swaps=100,
        constraint_rho=0.0,
        constraint_start=0.1,
        constraint_full=0.6,
        constraint_dual_lr=0.01,
        constraint_swap_margin=1.0,
        constraint_linear_moves=False,
        constraint_pin=True,
        constraint_search_rounds=3,
        optimize_background=False,
        auto_background_color=True,
    )
    for k, v in overrides.items():
        setattr(args, k, v)
    return args


def _optimizer(**overrides):
    torch.manual_seed(0)
    np.random.seed(0)
    H = W = 12
    gen = np.random.default_rng(0)
    target = torch.tensor(gen.integers(0, 256, (H, W, 3)), dtype=torch.float32)
    labels = gen.integers(0, 4, (H, W)).astype(np.float32)
    logits = np.log((labels / 4 + 0.1) / (1 - labels / 4 - 0.1 + 1e-6)).astype(np.float32)
    n_mat = 6
    colors = torch.tensor(gen.random((n_mat, 3)), dtype=torch.float32)
    tds = torch.tensor(gen.uniform(0.5, 5.0, n_mat), dtype=torch.float32)
    return FilamentOptimizer(
        _args(**overrides), target, logits, labels, None, colors, tds,
        torch.tensor([0.0, 0.0, 0.0]), torch.device("cpu"), None,
    )


def test_limits_count_the_background_slot_like_pruning():
    opt = _optimizer(pruning_max_colors=4, pruning_max_swaps=5)
    assert (opt.limit_colors, opt.limit_swaps) == (3, 5)
    assert opt.constrained
    opt = _optimizer(pruning_max_colors=4, flatforge=True)
    assert opt.limit_colors == 2  # background + clear reserved


def test_limits_at_or_above_what_exists_are_no_constraint():
    opt = _optimizer(pruning_max_colors=100, pruning_max_swaps=100)
    assert (opt.limit_colors, opt.limit_swaps) == (None, None)
    assert not opt.constrained


def test_unconstrained_path_is_untouched_without_the_flag():
    opt = _optimizer(constrained_opt=False, pruning_max_colors=3, pruning_max_swaps=2)
    assert not opt.constrained
    assert not hasattr(opt, "_lam")
    for i in range(3):
        opt.step(record_best=True)
    assert torch.isfinite(opt.loss)


def test_expected_counts_match_a_hard_stack_and_have_gradients():
    opt = _optimizer(pruning_max_swaps=3)
    dg = torch.tensor([0, 0, 1, 1, 2, 2, 2, 0, 0, 0, 1, 1])
    hard = torch.full((12, 6), -50.0)
    hard[torch.arange(12), dg] = 50.0
    e = opt._expected_counts(hard)
    # 4 changes between layers + the change from the (custom) base to layer 0.
    assert e.tolist() == pytest.approx([3.0, 5.0], abs=1e-3)
    soft = torch.zeros(12, 6, requires_grad=True)
    opt._expected_counts(soft).sum().backward()
    assert soft.grad is not None and torch.isfinite(soft.grad).all()


def test_dual_ascent_raises_the_multiplier_while_over_the_limit():
    opt = _optimizer(pruning_max_colors=100, pruning_max_swaps=1, iterations=20, constraint_start=0.0)
    for _ in range(5):
        opt.step(record_best=False)
    assert opt._lam[1] > 0  # swaps multiplier
    assert opt._lam[0] == 0  # colours are not limited


@pytest.mark.parametrize("colors,swaps", [(3, 100), (100, 3), (4, 5)])
def test_training_keeps_every_snapshot_and_the_result_within_the_limits(colors, swaps):
    opt = _optimizer(pruning_max_colors=colors, pruning_max_swaps=swaps)
    for i in range(opt.args.iterations):
        opt.step(record_best=i % 10 == 0)
        if opt.best_params is not None:
            n_colors, n_swaps, _ = opt.solution_counts()
            assert n_colors <= colors - 1 and n_swaps <= swaps
    opt.constrained_local_search(compound=True)
    opt.end_check_scope()
    n_colors, n_swaps, _ = opt.solution_counts()
    assert n_colors <= colors - 1 and n_swaps <= swaps
    assert np.isfinite(opt.solution_loss())


def test_local_search_never_worsens_and_stays_feasible():
    from autoforge.Helper.PruningHelper import _eval_candidates_batch, _make_shared_eff_thick

    opt = _optimizer(pruning_max_colors=4, pruning_max_swaps=4)
    for i in range(20):
        opt.step(record_best=i % 5 == 0)
    dg_before, _ = opt.get_discretized_solution(best=True)
    shared = _make_shared_eff_thick(opt)
    before, _ = _eval_candidates_batch(opt, [dg_before], eff_thick=shared)
    opt.constrained_local_search(compound=True)
    dg_after, _ = opt.get_discretized_solution(best=True)
    after, _ = _eval_candidates_batch(opt, [dg_after], eff_thick=shared)
    opt.end_check_scope()
    assert after <= before + 1e-6
    assert feasible(dg_after, opt.limit_colors, opt.limit_swaps)


# --------------------------------------------------------------------------
# CLI paths used by benchmarks/prune_harness.py


@pytest.fixture
def _cli_inputs(tmp_path):
    import cv2

    img = np.zeros((32, 32, 3), dtype=np.uint8)
    img[:, :16] = (30, 120, 220)
    img[:16, 16:] = (240, 240, 240)
    img[16:, 16:] = (20, 160, 40)
    img_path = tmp_path / "img.png"
    cv2.imwrite(str(img_path), img)
    csv = tmp_path / "filaments.csv"
    rows = [
        ("#000000", "Black", 0.6), ("#ffffff", "White", 3.0), ("#ff0000", "Red", 2.0),
        ("#00aa00", "Green", 2.0), ("#0000ff", "Blue", 1.5), ("#ffff00", "Yellow", 4.0),
    ]
    csv.write_text(
        "Brand, Type, Color, Name, TD, Owned, Uuid\n"
        + "".join(f"B,PLA,{c},{n},{td},false,{{{i}}}\n" for i, (c, n, td) in enumerate(rows))
    )
    return img_path, csv


def _run_cli(monkeypatch, img_path, csv, out, *extra):
    from autoforge.auto_forge import parse_args, start

    monkeypatch.setattr(
        "sys.argv",
        [
            "autoforge", "--input_image", str(img_path), "--csv_file", str(csv),
            "--output_folder", str(out), "--stl_output_size", "20", "--iterations", "60",
            "--discrete_check", "10", "--no-visualize", "--random_seed", "1",
            "--minimal_postprocess", "--no-cuda_graph", *extra,
        ],
    )
    return start(parse_args())


def test_cli_constrained_opt_writes_counts_within_the_limits(monkeypatch, tmp_path, _cli_inputs):
    img_path, csv = _cli_inputs
    out = tmp_path / "out"
    loss = _run_cli(
        monkeypatch, img_path, csv, out, "--constrained_opt", "--no-perform_pruning",
        "--pruning_max_colors", "3", "--pruning_max_swaps", "3",
    )
    assert np.isfinite(loss)
    counts = json.loads((out / "final_counts.json").read_text())
    # The print's filaments, the base included.
    assert counts["colors"] <= 3 and counts["swaps"] <= 3


def test_cli_prune_sweep_prunes_one_training_to_every_limit(monkeypatch, tmp_path, _cli_inputs):
    img_path, csv = _cli_inputs
    out = tmp_path / "out"
    _run_cli(monkeypatch, img_path, csv, out, "--perform_pruning", "--prune_sweep", "100:100,3:100,100:2")
    sweep = json.loads((out / "prune_sweep.json").read_text())
    assert [(r["max_colors"], r["max_swaps"]) for r in sweep] == [(100, 100), (3, 100), (100, 2)]
    assert sweep[1]["colors"] <= 3
    assert sweep[2]["swaps"] <= 2
    assert all(np.isfinite(r["loss"]) for r in sweep)


# --------------------------------------------------------------------------
# WebUI: max_colors / max_swaps set before a run


def test_webui_limits_map_to_constrained_opt():
    from autoforge.webui.helpers.pipeline_runner import _make_settings

    none = _make_settings({})
    assert not none.constrained_opt
    both = _make_settings({"max_colors": 4, "max_swaps": 10})
    assert both.constrained_opt
    assert (both.pruning_max_colors, both.pruning_max_swaps) == (4, 10)
    colors_only = _make_settings({"max_colors": 3, "max_swaps": None})
    assert colors_only.constrained_opt
    assert (colors_only.pruning_max_colors, colors_only.pruning_max_swaps) == (3, 100)
    # Every constraint_* knob the optimizer reads has a default.
    for key in ("constraint_start", "constraint_full", "constraint_rho", "constraint_dual_lr",
                "constraint_search_rounds", "constraint_pin", "constraint_swap_margin"):
        assert hasattr(both, key)


def test_webui_limit_settings_are_validated():
    from pydantic import ValidationError

    from autoforge.webui.models import OptimizationSettings

    s = OptimizationSettings()
    assert s.max_colors is None and s.max_swaps is None
    assert OptimizationSettings(max_colors=2, max_swaps=0).max_swaps == 0
    with pytest.raises(ValidationError):
        OptimizationSettings(max_colors=1)
    with pytest.raises(ValidationError):
        OptimizationSettings(max_swaps=-1)
    # The frontend posts camelCase too.
    assert OptimizationSettings(**{"maxColors": 5}).max_colors == 5


# --------------------------------------------------------------------------
# The base filament is one of the colors, counted once


def test_projection_keeps_the_base_free_and_in_every_palette():
    gen = torch.Generator().manual_seed(3)
    for _ in range(20):
        score = torch.randn(6, 4, generator=gen)
        out = project_assignment(score, 1, None, free=0)
        # One color besides base material 0.
        assert len(set(out.tolist()) - {0}) <= 1
        # Exact: best over stacks using base + one other.
        best_v = max(
            sum(max(score[l, m].item() for m in (0, other)) for l in range(6)) for other in (1, 2, 3)
        )
        assert _value(score, out) == pytest.approx(best_v, abs=1e-4)
    assert project_assignment(torch.randn(6, 4), 3, None, free=0).shape == (6,)


def test_counts_leave_the_base_out():
    dg = torch.tensor([0, 0, 2, 2, 0, 1])
    assert count_colors_swaps(dg, free=0) == (2, 3)
    assert count_colors_swaps(dg) == (3, 3)
    assert feasible(dg, 2, None, free=0) and not feasible(dg, 2, None)


def test_neighbours_can_merge_into_the_base_and_use_it_freely():
    dg = torch.tensor([1, 1, 2, 2, 3])
    cands = {tuple(c.tolist()) for c in neighbour_stacks(dg, 4, 3, None, free=0)}
    assert (0, 0, 2, 2, 3) in cands  # colour 1 merged into the base
    assert (1, 0, 2, 2, 3) in cands  # base reused by a layer at no colour cost
    for c in cands:
        assert len(set(c) - {0}) <= 3


def test_base_material_is_found_by_color():
    from autoforge.Helper.ConstraintHelper import base_material_index

    mats = torch.tensor([[1.0, 0, 0], [0, 1.0, 0], [0.2, 0.2, 0.2]])
    assert base_material_index(torch.tensor([0, 1.0, 0]), mats) == 1
    assert base_material_index(torch.tensor([0.5, 0.5, 0.5]), mats) is None


def test_optimizer_budget_leaves_out_a_base_that_is_a_filament():
    opt = _optimizer(pruning_max_colors=4)
    assert opt.base_material is None  # black base, no black filament
    opt.material_colors[2] = torch.tensor([0.0, 0.0, 0.0])
    from autoforge.Helper.ConstraintHelper import base_material_index

    assert base_material_index(opt.background, opt.material_colors) == 2


def test_color_pruning_counts_the_base_once_and_may_merge_into_it():
    from autoforge.Helper.PruningHelper import prune_num_colors

    opt = _optimizer(constrained_opt=False)
    for i in range(20):
        opt.step(record_best=i % 5 == 0)
    dg, _ = opt.get_discretized_solution(best=True)
    opt.base_material = int(dg[0])  # pretend the base is the bottom layer's filament
    prune_num_colors(opt, 2, opt.vis_tau, None, fast=True, chunking_percent=0.25, pruning_batch_size=8)
    after, _ = opt.get_discretized_solution(best=True)
    assert len(set(after.tolist()) - {opt.base_material}) <= 2


# --------------------------------------------------------------------------
# The base filament is optimized (default with an automatic base)


def test_base_is_optimized_by_default_only_when_automatic():
    opt = _optimizer(optimize_background=True, auto_background_color=True, constrained_opt=False)
    assert opt.bg_logits is not None
    opt = _optimizer(optimize_background=True, auto_background_color=False, constrained_opt=False)
    assert opt.bg_logits is None  # a base picked by hand stays as picked
    from autoforge.auto_forge import parse_args
    import sys

    argv = sys.argv
    try:
        sys.argv = ["autoforge", "--input_image", "x.png"]
        assert parse_args().optimize_background is True
    finally:
        sys.argv = argv


def test_optimized_base_is_what_the_result_uses():
    opt = _optimizer(optimize_background=True, auto_background_color=True, pruning_max_colors=4)
    for i in range(opt.args.iterations):
        opt.step(record_best=i % 10 == 0)
    idx = int(opt.best_params["background_index"])
    args = types.SimpleNamespace(background_color="#000000")
    names = [f"F{i}" for i in range(opt.material_colors.shape[0])]
    opt.finalize_background(args, names)
    assert args.background_material_index == idx
    assert args.background_material_name == f"F{idx}"
    assert opt.base_material == idx
    torch.testing.assert_close(opt.background, opt.material_colors[idx])
    rgb = [int(round(float(c) * 255)) for c in opt.material_colors[idx]]
    assert args.background_color == "#" + "".join(f"{c:02X}" for c in rgb)
    # The limits count this base as one of the colors.
    assert opt.print_colors() <= 4


def test_cli_run_reports_the_optimized_base(monkeypatch, tmp_path, _cli_inputs):
    img_path, csv = _cli_inputs
    out = tmp_path / "out"
    _run_cli(monkeypatch, img_path, csv, out, "--no-perform_pruning", "--constrained_opt",
             "--pruning_max_colors", "3")
    text = (out / "auto_background_color.txt").read_text()
    assert "optimized_filament_color=" in text
    counts = json.loads((out / "final_counts.json").read_text())
    assert counts["colors"] <= 3


def test_webui_pipeline_uses_the_optimized_base(tmp_path, _cli_inputs):
    from autoforge.webui.helpers.pipeline_runner import run_pipeline
    from autoforge.webui.helpers.sliders import derive_base_from_result

    img_path, csv = _cli_inputs
    colors = ["#000000", "#ffffff", "#ff0000", "#00aa00", "#0000ff", "#ffff00"]
    fil = [{"color": c, "td": 2.0, "name": f"T - {c}", "brand": "T", "short_name": c, "owned": False,
            "uuid": f"u{i}", "filament_type": "PLA"} for i, c in enumerate(colors)]
    state = run_pipeline(str(img_path), fil, str(tmp_path / "w"), dict(
        iterations=60, stl_output_size=20, num_init_rounds=1, discrete_check=10, visualize=False,
        cuda_graph=False, random_seed=1, max_colors=3))
    opt, args = state["optimizer"], state["args"]
    idx = int(opt.best_params["background_index"])
    assert args.background_material_index == idx
    base = derive_base_from_result(state)
    assert base["filament_uuid"] == f"u{idx}"
    assert base["color"].lower() == colors[idx]
    torch.testing.assert_close(state["background"], opt.material_colors[idx])


# --------------------------------------------------------------------------
# Swaps count from the base up (the base is band 0)


@pytest.mark.parametrize("base", [0, 2, -1])
def test_projection_counts_the_change_from_the_base(base):
    gen = torch.Generator().manual_seed(5)
    for max_swaps in (1, 2, 3):
        for _ in range(10):
            score = torch.randn(5, 3, generator=gen)
            out = project_assignment(score, None, max_swaps, free=base)
            assert count_colors_swaps(out, base)[1] <= max_swaps
            best_v = max(
                sum(score[l, m].item() for l, m in enumerate(dg))
                for dg in itertools.product(range(3), repeat=5)
                if count_colors_swaps(torch.tensor(dg), base)[1] <= max_swaps
            )
            assert _value(score, out) == pytest.approx(best_v, abs=1e-4)


def test_first_band_in_the_base_filament_is_no_swap():
    assert count_colors_swaps(torch.tensor([1, 1, 2]), free=1)[1] == 1
    assert count_colors_swaps(torch.tensor([3, 3, 2]), free=1)[1] == 2
    assert count_colors_swaps(torch.tensor([3, 3, 2]), free=-1)[1] == 2  # custom base: always a change
    assert count_colors_swaps(torch.tensor([3, 3, 2]))[1] == 1  # no base: the old count


def test_swap_pruning_can_recolor_the_first_band_into_the_base():
    from autoforge.Helper.PruningHelper import find_color_bands, prune_num_swaps

    opt = _optimizer(constrained_opt=False)
    for i in range(20):
        opt.step(record_best=i % 5 == 0)
    dg, _ = opt.get_discretized_solution(best=True)
    opt.base_material = int((dg[0] + 1) % opt.material_colors.shape[0])  # a base unlike the first band
    prune_num_swaps(opt, 0, opt.vis_tau, None, fast=True, chunking_percent=0.25, pruning_batch_size=8)
    after, _ = opt.get_discretized_solution(best=True)
    # Zero swaps from the base up: one band, in the base filament.
    assert len(find_color_bands(after)) == 1 and int(after[0]) == opt.base_material


def test_swap_instructions_skip_a_first_band_in_the_base_filament():
    from autoforge.Helper.OutputHelper import generate_swap_instructions

    dg = np.array([1, 1, 2, 2, 0])
    height = np.full((2, 2), 5)
    names = ["Red", "White", "Blue"]
    with_base = "\n".join(generate_swap_instructions(dg, height, 0.04, 6, 0.24, names, "White", 1))
    assert "swap to White" not in with_base
    assert "At layer #4 (0.36mm) swap to Blue" in with_base
    other_base = "\n".join(generate_swap_instructions(dg, height, 0.04, 6, 0.24, names, "Red", 0))
    assert "At layer #2 (0.28mm) swap to White" in other_base


# --------------------------------------------------------------------------
# Progress: every step of a run and of pruning reports it


def test_webui_run_and_prune_report_progress_for_every_step(tmp_path, _cli_inputs):
    import copy

    from autoforge.webui.helpers.pipeline_runner import export_results, run_pipeline

    img_path, csv = _cli_inputs
    colors = ["#000000", "#ffffff", "#ff0000", "#00aa00", "#0000ff", "#ffff00"]
    fil = [{"color": c, "td": 2.0, "name": f"T - {c}", "brand": "T", "short_name": c, "owned": False,
            "uuid": f"u{i}", "filament_type": "PLA"} for i, c in enumerate(colors)]
    seen = []
    state = run_pipeline(str(img_path), fil, str(tmp_path / "w"), dict(
        iterations=60, stl_output_size=20, num_init_rounds=2, discrete_check=10, visualize=False,
        cuda_graph=False, random_seed=1, max_colors=3, max_swaps=4),
        phase_callback=lambda phase, f: seen.append((phase, f)))
    for phase in ("Estimating heights", "Refining within the limits"):
        fr = [f for p, f in seen if p == phase]
        assert fr and fr == sorted(fr) and fr[-1] == pytest.approx(1.0), (phase, fr)
    exported = []
    export_results(state, export_progress=exported.append)
    assert exported == sorted(exported) and exported[-1] == 1.0

    # Pruning: its closing steps report 0-100 as they go, not only at the end.
    prune = dict(state)
    args = copy.copy(state["args"])
    args.perform_pruning = True
    args.pruning_max_colors, args.pruning_max_swaps, args.pruning_max_layer = 3, 4, 12
    args.output_folder = str(tmp_path / "p")
    (tmp_path / "p").mkdir()
    prune["args"] = args
    reports = []
    opt = state["optimizer"]
    opt.preview_callback = lambda _o, pct, phase=None, loss=None: reports.append((phase, float(pct)))
    export_results(prune)
    for phase in ("Searching layer stacks", "Refining pixel heights", "Removing spikes"):
        pcts = [p for ph, p in reports if ph == phase]
        assert len(pcts) >= 3 and pcts[-1] == 100.0, (phase, pcts)
