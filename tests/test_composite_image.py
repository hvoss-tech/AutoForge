"""Semantic tests for the differentiable / discrete compositing model in
autoforge.Helper.OptimizerHelper.

These go beyond shape checks: they pin down the physical behaviour the
optimizer relies on (an empty stack shows the background, a tall opaque
stack shows its top material, lower transmission distance = more opaque,
the low-memory and chunked variants agree with the reference, gradients
flow).
"""

import pytest
import torch

from autoforge.Helper.OptimizerHelper import (
    PrecisionManager,
    _runs_from_materials,
    adaptive_round,
    batched_layer_material_indices,
    bleed_layer_effect,
    composite_image_cont,
    composite_image_cont_lowmem,
    composite_image_disc,
    deterministic_gumbel_noise,
    deterministic_gumbel_softmax,
    deterministic_rand_like,
)

RED, GREEN, BLUE = (
    torch.tensor([1.0, 0.0, 0.0]),
    torch.tensor([0.0, 1.0, 0.0]),
    torch.tensor([0.0, 0.0, 1.0]),
)


def _inputs(H=8, W=8, L=4, top_material=0, height_logit=0.0):
    pixel_logits = torch.full((H, W), float(height_logit))
    global_logits = torch.full((L, 3), -10.0)
    # make every layer choose `top_material` decisively
    global_logits[:, top_material] = 10.0
    material_colors = torch.stack([RED, GREEN, BLUE])
    material_TDs = torch.tensor([4.0, 4.0, 4.0])
    background = torch.tensor([0.0, 0.0, 0.0])
    return pixel_logits, global_logits, material_colors, material_TDs, background


# --------------------------------------------------------------------------
# adaptive_round
# --------------------------------------------------------------------------


def test_adaptive_round_hard_at_low_tau_is_exact_round():
    x = torch.tensor([0.2, 0.7, 1.4, 2.6, -0.4])
    out = adaptive_round(x, tau=0.0, high_tau=1.0, low_tau=0.0, temp=0.1)
    assert torch.equal(out, torch.round(x))


def test_adaptive_round_soft_stays_within_floor_ceil():
    x = torch.tensor([0.2, 0.7, 1.4, 2.6])
    out = adaptive_round(x, tau=1.0, high_tau=1.0, low_tau=0.0, temp=0.1)
    assert torch.all(out >= torch.floor(x) - 1e-6)
    assert torch.all(out <= torch.ceil(x) + 1e-6)


def test_adaptive_round_is_differentiable_in_soft_regime():
    x = torch.tensor([0.3, 1.6, 2.2], requires_grad=True)
    adaptive_round(x, tau=1.0, high_tau=1.0, low_tau=0.0, temp=0.1).sum().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()
    assert (x.grad > 0).all()  # sigmoid of (diff-0.5)/temp is increasing in diff


# --------------------------------------------------------------------------
# deterministic_gumbel_softmax
# --------------------------------------------------------------------------


def test_deterministic_gumbel_softmax_is_reproducible_and_seed_sensitive():
    logits = torch.tensor([0.1, 2.0, 0.3, 0.5])
    a = deterministic_gumbel_softmax(logits, tau=1.0, hard=False, rng_seed=42)
    b = deterministic_gumbel_softmax(logits, tau=1.0, hard=False, rng_seed=42)
    c = deterministic_gumbel_softmax(logits, tau=1.0, hard=False, rng_seed=999)
    assert torch.allclose(a, b)
    assert not torch.allclose(a, c)
    # hard=True returns a straight-through one-hot selection
    hard = deterministic_gumbel_softmax(logits, tau=1.0, hard=True, rng_seed=42)
    assert hard.sum() == pytest.approx(1.0, abs=1e-4)
    assert ((hard.detach() == 0) | (hard.detach() == 1)).all()


# --------------------------------------------------------------------------
# bleed_layer_effect
# --------------------------------------------------------------------------


def test_bleed_layer_effect_spreads_into_neighbours():
    mask = torch.zeros(1, 5, 5)
    mask[0, 2, 2] = 1.0
    out = bleed_layer_effect(mask, strength=0.5)
    assert out.shape == mask.shape
    assert out[0, 2, 2] == pytest.approx(1.0)  # centre tap is zero
    assert out[0, 2, 1] > 0.0 and out[0, 1, 2] > 0.0  # neighbours gained bleed


# --------------------------------------------------------------------------
# composite_image_cont — semantics
# --------------------------------------------------------------------------


def test_empty_stack_shows_background():
    pl, gl, mc, td, _bg = _inputs(height_logit=-30.0)
    bg = torch.tensor([0.2, 0.4, 0.6])
    out = composite_image_cont(pl, gl, 0.01, 0.01, 0.2, gl.shape[0], mc, td, bg)
    assert torch.allclose(out / 255.0, bg.expand_as(out / 255.0), atol=2e-2)


def test_tall_opaque_stack_shows_top_material_colour():
    # very tall stack, low TD (opaque), every layer red
    pl, gl, mc, _td, bg = _inputs(height_logit=30.0, top_material=0)
    td = torch.tensor([0.5, 0.5, 0.5])
    out = composite_image_cont(pl, gl, 0.01, 0.01, 0.4, gl.shape[0], mc, td, bg) / 255.0
    assert out[..., 0].mean() > 0.8
    assert out[..., 1].mean() < 0.2 and out[..., 2].mean() < 0.2


def test_lower_transmission_distance_is_more_opaque():
    pl, gl, mc, _td, _bg = _inputs(height_logit=2.0, top_material=0)
    bg = torch.tensor([0.0, 0.0, 1.0])  # blue background behind red stack
    opaque = composite_image_cont(
        pl, gl, 0.01, 0.01, 0.2, gl.shape[0], mc, torch.tensor([0.5, 0.5, 0.5]), bg
    ) / 255.0
    sheer = composite_image_cont(
        pl, gl, 0.01, 0.01, 0.2, gl.shape[0], mc, torch.tensor([40.0, 40.0, 40.0]), bg
    ) / 255.0
    # more opaque -> more red, less of the blue background bleeding through
    assert opaque[..., 0].mean() > sheer[..., 0].mean()
    assert opaque[..., 2].mean() < sheer[..., 2].mean()


def test_output_is_finite_and_roughly_bounded():
    pl, gl, mc, td, bg = _inputs(height_logit=1.0)
    out = composite_image_cont(pl, gl, 0.5, 0.5, 0.2, gl.shape[0], mc, td, bg)
    assert torch.isfinite(out).all()
    assert out.min() >= -1.0 and out.max() <= 256.0


def test_gradients_flow_to_both_parameter_tensors():
    # tau_height=1.0 keeps the height path in its smooth regime; at low tau
    # the model is deliberately near-discrete and the height gradient vanishes.
    pl, gl, mc, td, bg = _inputs(height_logit=0.5)
    # spread the per-layer material choice out so gumbel-softmax isn't saturated
    gl = (torch.randn_like(gl)).clone().requires_grad_(True)
    pl = pl.clone().requires_grad_(True)
    target = torch.rand(pl.shape[0], pl.shape[1], 3) * 255.0
    out = composite_image_cont(pl, gl, 1.0, 1.0, 0.2, gl.shape[0], mc, td, bg)
    torch.nn.functional.mse_loss(out, target).backward()
    assert pl.grad is not None and torch.isfinite(pl.grad).all()
    assert gl.grad is not None and torch.isfinite(gl.grad).all()
    assert pl.grad.abs().sum() > 0
    assert gl.grad.abs().sum() > 0


# --------------------------------------------------------------------------
# variants agree
# --------------------------------------------------------------------------


def test_lowmem_matches_reference_composite():
    torch.manual_seed(0)
    pl, gl, mc, td, bg = _inputs(H=12, W=10, L=6, height_logit=1.0)
    kw = dict(tau_height=0.4, tau_global=0.4, h=0.2, max_layers=6)
    # gumbel_exp fixed so both calls use the same material noise
    gumbel_exp = torch.empty(6, 3).exponential_(1.0)
    ref = composite_image_cont(
        pl, gl, kw["tau_height"], kw["tau_global"], kw["h"], kw["max_layers"],
        mc, td, bg, None, gumbel_exp,
    )
    low = composite_image_cont_lowmem(
        pl, gl, kw["tau_height"], kw["tau_global"], kw["h"], kw["max_layers"],
        mc, td, bg, None, gumbel_exp, 2,
    )
    assert torch.allclose(ref, low, atol=1e-3)


def test_disc_is_deterministic_for_a_fixed_seed():
    pl, gl, mc, td, bg = _inputs(height_logit=1.0)
    a = composite_image_disc(pl, gl, 0.2, 0.2, 0.2, gl.shape[0], mc, td, bg, rng_seed=5)
    b = composite_image_disc(pl, gl, 0.2, 0.2, 0.2, gl.shape[0], mc, td, bg, rng_seed=5)
    assert torch.equal(a, b)
    assert a.shape == (pl.shape[0], pl.shape[1], 3)


def test_disc_layer_chunking_does_not_change_the_result():
    pl, gl, mc, td, bg = _inputs(H=10, W=10, L=9, height_logit=1.0)
    gl = torch.randn_like(gl)
    whole = composite_image_disc(pl, gl, 0.2, 0.2, 0.2, 9, mc, td, bg, rng_seed=3)
    chunked = composite_image_disc(
        pl, gl, 0.2, 0.2, 0.2, 9, mc, td, bg, rng_seed=3, layer_chunk=3
    )
    assert torch.allclose(whole, chunked, atol=1e-3)


# --------------------------------------------------------------------------
# deterministic RNG / indexing helpers
# --------------------------------------------------------------------------


def test_deterministic_rand_like_is_seed_addressable():
    t = torch.zeros(6)
    assert torch.allclose(deterministic_rand_like(t, 123), deterministic_rand_like(t, 123))
    assert not torch.allclose(deterministic_rand_like(t, 123), deterministic_rand_like(t, 124))
    vals = deterministic_rand_like(t, 7)
    assert vals.shape == t.shape
    assert vals.min() >= 0.0 and vals.max() < 1.0


def test_deterministic_gumbel_noise_matches_shape_and_is_reproducible():
    seeds = torch.tensor([100, 101, 102, 103])
    a = deterministic_gumbel_noise(seeds, 5)
    b = deterministic_gumbel_noise(seeds, 5)
    assert a.shape == (4, 5)
    assert torch.equal(a, b)


def test_batched_layer_material_indices_reproduces_the_scalar_loop():
    # the batched kernel documents itself as an exact replacement for the
    # per-layer argmax(deterministic_gumbel_softmax(..., hard=True)) loop.
    torch.manual_seed(0)
    for tau in (0.2, 0.8, 2.0):
        gl = torch.randn(15, 6)
        seed_base = 500
        batched = batched_layer_material_indices(gl, tau, seed_base)
        scalar = torch.stack([
            torch.argmax(
                deterministic_gumbel_softmax(gl[j], tau, True, seed_base + j)
            )
            for j in range(gl.shape[0])
        ])
        assert torch.equal(batched, scalar)


def test_runs_from_materials_groups_contiguous_layers():
    starts, ends, mats = _runs_from_materials(torch.tensor([0, 0, 1, 1, 1, 2, 0]))
    assert starts.tolist() == [0, 2, 5, 6]
    assert ends.tolist() == [2, 5, 6, 7]
    assert mats.tolist() == [0, 1, 2, 0]


# --------------------------------------------------------------------------
# PrecisionManager (CPU path)
# --------------------------------------------------------------------------


def test_precision_manager_is_disabled_and_fp32_on_cpu():
    prec = PrecisionManager(torch.device("cpu"))
    assert prec.enabled is False
    assert prec.scaler is None
    with prec.autocast():
        assert (torch.randn(4, 4) @ torch.randn(4, 4)).dtype == torch.float32


def test_precision_manager_backward_and_step_updates_parameters_on_cpu():
    prec = PrecisionManager(torch.device("cpu"))
    w = torch.zeros(3, requires_grad=True)
    opt = torch.optim.SGD([w], lr=0.1)
    target = torch.tensor([1.0, -2.0, 3.0])
    with prec.autocast():
        loss = (w - target).pow(2).sum()
    prec.backward_and_step(loss, opt)
    # one SGD step of lr=0.1 on d/dw (w-t)^2 = 2(w-t): w <- 0 - 0.1*2*(0-t)
    assert torch.allclose(w.detach(), 0.2 * target, atol=1e-6)
