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
    assert e.tolist() == pytest.approx([3.0, 4.0], abs=1e-3)
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
    assert counts["colors"] <= 2 and counts["swaps"] <= 3


def test_cli_prune_sweep_prunes_one_training_to_every_limit(monkeypatch, tmp_path, _cli_inputs):
    img_path, csv = _cli_inputs
    out = tmp_path / "out"
    _run_cli(monkeypatch, img_path, csv, out, "--perform_pruning", "--prune_sweep", "100:100,3:100,100:2")
    sweep = json.loads((out / "prune_sweep.json").read_text())
    assert [(r["max_colors"], r["max_swaps"]) for r in sweep] == [(100, 100), (3, 100), (100, 2)]
    assert sweep[1]["colors"] <= 2
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
