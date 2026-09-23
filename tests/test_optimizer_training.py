"""Behavioural tests for FilamentOptimizer's training / discretization /
post-processing pipeline (autoforge.Modules.Optimizer).

Everything runs on CPU with the perception loss disabled so the tests are
fast and deterministic.
"""

import argparse

import numpy as np
import pytest
import torch

from autoforge.Helper.FilamentHelper import count_distinct_colors, count_swaps
from autoforge.Modules.Optimizer import FilamentOptimizer


def _args(**overrides):
    base = dict(
        max_layers=8,
        layer_height=0.2,
        learning_rate=2e-2,
        init_tau=1.0,
        final_tau=0.02,
        iterations=80,
        warmup_fraction=0.2,
        learning_rate_warmup_fraction=0.2,
        visualize=False,
        tensorboard=False,
        run_name="",
        disable_visualization_for_gradio=1,
        output_folder="/tmp",
        background_height=0.4,
        spike_threshold_layers=1,
        spike_removal_passes=4,
    )
    base.update(overrides)
    ns = argparse.Namespace()
    for k, v in base.items():
        setattr(ns, k, v)
    return ns


def _make_optimizer(H=24, W=24, M=4, seed=0, labels=None, **args_overrides):
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device("cpu")
    # A target that is a smooth colour gradient -> the optimizer has real
    # structure to fit rather than pure noise.
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    target = torch.tensor(
        np.stack([255 * xx / W, 255 * yy / H, 255 * (1 - xx / W)], axis=-1),
        dtype=torch.float32,
    )
    pixel_height_logits_init = np.zeros((H, W), dtype=np.float32)
    pixel_height_labels = (
        np.zeros((H, W), dtype=np.int32) if labels is None else labels.astype(np.int32)
    )
    global_logits_init = np.random.randn(args_overrides.get("max_layers", 8), M).astype(np.float32)
    material_colors = torch.rand(M, 3)
    material_TDs = torch.rand(M) * 4.0 + 1.0
    background = torch.zeros(3)
    return FilamentOptimizer(
        _args(**args_overrides),
        target,
        pixel_height_logits_init,
        pixel_height_labels,
        global_logits_init,
        material_colors,
        material_TDs,
        background,
        device,
        perception_loss_module=None,
    )


# --------------------------------------------------------------------------
# tau / lr schedule
# --------------------------------------------------------------------------


def test_tau_schedule_holds_during_warmup_then_decays_and_clamps():
    opt = _make_optimizer(iterations=50, init_tau=1.0, final_tau=0.1, warmup_fraction=0.2)
    taus = []
    for _ in range(70):  # run past `iterations` to see the clamp engage
        taus.append(opt._get_tau()[0])
        opt.step()
    assert taus[0] == pytest.approx(1.0)
    assert taus[int(opt.warmup_steps) - 1] == pytest.approx(1.0)   # flat through warmup
    assert all(b <= a + 1e-6 for a, b in zip(taus, taus[1:]))      # non-increasing
    assert min(taus) >= 0.1 - 1e-6                                 # never below final_tau
    assert taus[-1] == pytest.approx(0.1, abs=1e-6)                # clamped at final_tau


# --------------------------------------------------------------------------
# training reduces the loss
# --------------------------------------------------------------------------


def test_training_reduces_soft_loss_at_constant_tau():
    # With a fixed tau the step() loss is a plain differentiable objective and
    # gradient descent must bring it down. (Under the annealing schedule the
    # soft loss can rise as tau->0 even while the discrete solution improves,
    # so that path is checked via best_discrete_loss below instead.)
    opt = _make_optimizer(seed=1, iterations=140, init_tau=0.6, final_tau=0.6)
    early = np.mean([opt.step().item() for _ in range(10)])
    for _ in range(110):
        opt.step()
    late = np.mean([opt.step().item() for _ in range(10)])
    assert late < early


def test_best_discrete_loss_never_increases_and_improves_over_a_run():
    opt = _make_optimizer(seed=1, iterations=140)
    history = []
    for _ in range(140):
        opt.step(record_best=True)
        history.append(opt.best_discrete_loss)
    assert all(b <= a + 1e-9 for a, b in zip(history, history[1:]))
    assert history[-1] < history[9]


def test_step_returns_detached_finite_tensor():
    opt = _make_optimizer()
    loss = opt.step(record_best=False)
    assert torch.is_tensor(loss) and not loss.requires_grad
    assert np.isfinite(loss.item())


def test_parameters_actually_move():
    opt = _make_optimizer(seed=2)
    before = opt.params["global_logits"].detach().clone()
    for _ in range(15):
        opt.step()
    after = opt.params["global_logits"].detach()
    assert not torch.allclose(before, after)


# --------------------------------------------------------------------------
# discretization
# --------------------------------------------------------------------------


def test_get_discretized_solution_is_deterministic_and_shaped():
    opt = _make_optimizer()
    for _ in range(10):
        opt.step(record_best=True)
    dg1, dh1 = opt.get_discretized_solution(best=True)
    dg2, dh2 = opt.get_discretized_solution(best=True)
    assert torch.equal(dg1, dg2) and torch.equal(dh1, dh2)
    assert dg1.shape[0] == opt.params["global_logits"].shape[0]
    assert dh1.shape == (opt.H, opt.W)
    assert int(dh1.max()) <= opt.max_layers


def test_discretize_solution_matches_get_discretized_solution():
    opt = _make_optimizer()
    for _ in range(10):
        opt.step(record_best=True)
    dg, dh = opt.discretize_solution(
        opt.best_params, opt.vis_tau, opt.h, opt.max_layers, rng_seed=opt.best_seed
    )
    dg2, dh2 = opt.get_discretized_solution(best=True)
    assert torch.equal(dg, dg2) and torch.equal(dh, dh2)
    assert dg.dtype == torch.int64 or dg.dtype == torch.long
    assert dh.min() >= 0


# --------------------------------------------------------------------------
# height-offset apply / remove
# --------------------------------------------------------------------------


def _labels_with_centre_cluster(H, W):
    lab = np.zeros((H, W), dtype=np.int32)
    lab[H // 4 : 3 * H // 4, W // 4 : 3 * W // 4] = 1
    return lab


def test_apply_then_remove_height_offset_is_identity():
    opt = _make_optimizer(H=24, W=24, labels=_labels_with_centre_cluster(24, 24))
    with torch.no_grad():
        opt.height_offsets.copy_(torch.randn_like(opt.height_offsets))
    logits = torch.randn(opt.H, opt.W)
    back = opt._remove_height_offset(opt._apply_height_offset(logits))
    assert torch.allclose(back, logits, atol=1e-5)


def test_apply_height_offset_leaves_background_pixels_untouched():
    opt = _make_optimizer(H=24, W=24, labels=_labels_with_centre_cluster(24, 24))
    with torch.no_grad():
        opt.height_offsets.copy_(torch.full_like(opt.height_offsets, 3.0))
    out = opt._apply_height_offset(torch.zeros(opt.H, opt.W))
    assert torch.allclose(out[0, 0], torch.tensor(0.0))          # background
    assert out[12, 12].item() == pytest.approx(3.0)              # cluster 1 shifted


def test_apply_height_offset_resizes_offset_field_to_the_export_resolution():
    opt = _make_optimizer(H=32, W=32, labels=_labels_with_centre_cluster(32, 32))
    with torch.no_grad():
        opt.height_offsets.copy_(torch.full_like(opt.height_offsets, 2.0))

    full = torch.zeros(64, 128)  # different aspect + resolution, like full-res export
    out = opt._apply_height_offset(pixel_logits=full).detach()
    assert out.shape == (64, 128)
    # the offset region is centred, so the corners stay background (== 0)
    assert float(out[0, 0]) == pytest.approx(0.0, abs=1e-4)
    assert float(out[-1, -1]) == pytest.approx(0.0, abs=1e-4)
    assert out.abs().max() > 0.5  # the centre did get shifted


def test_get_current_parameters_returns_detached_clones():
    opt = _make_optimizer()
    opt.step()
    params = opt.get_current_parameters()
    assert set(params) == {"pixel_height_logits", "global_logits", "height_offsets"}
    for v in params.values():
        assert not v.requires_grad
    params["global_logits"].add_(999.0)
    assert not torch.allclose(params["global_logits"], opt.params["global_logits"].detach())


# --------------------------------------------------------------------------
# best-params bookkeeping
# --------------------------------------------------------------------------


def test_record_best_tracks_best_discrete_loss_on_the_run_device():
    opt = _make_optimizer()
    assert opt.best_discrete_loss == float("inf")
    for _ in range(12):
        opt.step(record_best=True)
    assert np.isfinite(opt.best_discrete_loss)
    assert opt.best_params is not None
    for v in opt.best_params.values():
        assert v.device == opt.target.device


# --------------------------------------------------------------------------
# pruning + post-processing never regress the discrete loss
# --------------------------------------------------------------------------


def test_prune_runs_and_does_not_worsen_loss_or_increase_colours():
    opt = _make_optimizer(seed=3, iterations=80)
    for _ in range(60):
        opt.step(record_best=True)
    loss_before = opt.best_discrete_loss
    dg_before, _ = opt.get_discretized_solution(best=True)
    colours_before = count_distinct_colors(dg_before)

    completed = opt.prune(
        max_colors_allowed=2,
        max_swaps_allowed=3,
        min_layers_allowed=1,
        max_layers_allowed=opt.max_layers,
        search_seed=False,
    )
    assert completed is True

    dg_after, _ = opt.get_discretized_solution(best=True)
    assert count_distinct_colors(dg_after) <= colours_before
    assert opt.best_discrete_loss <= loss_before + 1e-3


def test_prune_apply_spike_removal_false_skips_the_spike_phase(monkeypatch):
    """webui auto-repeat pruning calls prune() once per pass but must only
    pay for (and trade accuracy on, see post_remove_spikes) spike removal on
    the pass it settles on — passing apply_spike_removal=False must make
    prune() skip post_remove_spikes entirely, not just no-op it."""
    opt = _make_optimizer(seed=9, iterations=80, spike_removal=True)
    for _ in range(40):
        opt.step(record_best=True)

    calls = []
    monkeypatch.setattr(opt, "post_remove_spikes", lambda **kw: calls.append(kw))

    completed = opt.prune(
        max_colors_allowed=2,
        max_swaps_allowed=3,
        min_layers_allowed=1,
        max_layers_allowed=opt.max_layers,
        search_seed=False,
        fine_tune_height=False,
        apply_spike_removal=False,
    )
    assert completed is True
    assert calls == []

    completed = opt.prune(
        max_colors_allowed=2,
        max_swaps_allowed=3,
        min_layers_allowed=1,
        max_layers_allowed=opt.max_layers,
        search_seed=False,
        fine_tune_height=False,
        apply_spike_removal=True,
    )
    assert completed is True
    assert len(calls) == 1


def test_prune_num_colors_phase_respects_the_colour_budget():
    from autoforge.Helper.PruningHelper import prune_num_colors

    opt = _make_optimizer(seed=7, iterations=80)
    for _ in range(60):
        opt.step(record_best=True)
    dg_before, _ = opt.get_discretized_solution(best=True)
    loss_before = opt.best_discrete_loss

    prune_num_colors(opt, 2, opt.vis_tau, None)

    dg_after, _ = opt.get_discretized_solution(best=True)
    assert count_distinct_colors(dg_after) <= max(2, 1)
    assert count_distinct_colors(dg_after) <= count_distinct_colors(dg_before)
    assert opt.best_discrete_loss <= loss_before + 1e-3


def test_prune_num_swaps_phase_respects_the_swap_budget():
    from autoforge.Helper.PruningHelper import prune_num_swaps

    opt = _make_optimizer(seed=8, iterations=80)
    for _ in range(60):
        opt.step(record_best=True)
    dg_before, _ = opt.get_discretized_solution(best=True)
    loss_before = opt.best_discrete_loss

    prune_num_swaps(opt, 2, opt.vis_tau, None)

    dg_after, _ = opt.get_discretized_solution(best=True)
    assert count_swaps(dg_after) <= max(count_swaps(dg_before), 2)
    assert opt.best_discrete_loss <= loss_before + 1e-3


def test_fine_tune_height_offsets_returns_bool_and_holds_or_improves():
    opt = _make_optimizer(seed=4)
    for _ in range(40):
        opt.step(record_best=True)
    loss_before = opt.best_discrete_loss
    kept = opt.fine_tune_height_offsets(num_steps=15, lr=5e-3)
    assert isinstance(kept, bool)
    assert opt.best_discrete_loss <= loss_before + 1e-3


def test_post_remove_spikes_runs_after_training():
    opt = _make_optimizer(seed=5)
    for _ in range(30):
        opt.step(record_best=True)
    loss_before = opt.best_discrete_loss
    opt.post_remove_spikes()
    assert opt.best_discrete_loss <= loss_before + 1e-3


def test_rng_seed_search_does_not_regress():
    opt = _make_optimizer(seed=6)
    for _ in range(30):
        opt.step(record_best=True)
    start = opt.best_discrete_loss
    best_seed, best_loss = opt.rng_seed_search(start, num_seeds=4, autoset_seed=True)
    assert best_loss <= start * 1.02
