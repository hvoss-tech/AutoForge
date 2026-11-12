"""Tests for autoforge.Loss.LossFunctions (compute_loss + loss_fn)."""

import pytest
import torch

from autoforge.Loss.LossFunctions import compute_loss, loss_fn


def _mats():
    material_colors = torch.tensor(
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], dtype=torch.float32
    )
    material_TDs = torch.tensor([0.5, 0.7, 0.9], dtype=torch.float32)
    background = torch.tensor([1.0, 1.0, 1.0], dtype=torch.float32)
    return material_colors, material_TDs, background


# --------------------------------------------------------------------------
# compute_loss
# --------------------------------------------------------------------------


def test_identical_images_give_near_zero_scalar_loss():
    target = torch.randint(0, 256, (16, 16, 3), dtype=torch.float32)
    loss = compute_loss(comp=target.clone(), target=target)
    assert loss.dim() == 0
    assert torch.isfinite(loss)
    assert loss.item() < 1e-6


def test_larger_colour_error_gives_larger_loss():
    target = torch.zeros(8, 8, 3)
    near = compute_loss(comp=torch.full((8, 8, 3), 10.0), target=target)
    far = compute_loss(comp=torch.full((8, 8, 3), 200.0), target=target)
    assert far.item() > near.item()


@pytest.mark.parametrize("focus_map_factory", [
    lambda: torch.ones(8, 8),
    lambda: torch.ones(8, 8, 1),
])
def test_focus_map_all_ones_equals_unweighted(focus_map_factory):
    comp = torch.randint(0, 256, (8, 8, 3), dtype=torch.float32)
    target = torch.randint(0, 256, (8, 8, 3), dtype=torch.float32)
    plain = compute_loss(comp=comp, target=target)
    weighted = compute_loss(comp=comp, target=target, focus_map=focus_map_factory())
    assert torch.allclose(plain, weighted, atol=1e-4)


def test_focus_map_concentrates_loss_on_prioritised_region():
    # error only in the top half; a focus map that highlights the top half
    # must yield a higher normalised loss than one that highlights the bottom.
    comp = torch.zeros(8, 8, 3)
    target = torch.zeros(8, 8, 3)
    target[:4] = 255.0

    top = torch.zeros(8, 8)
    top[:4] = 1.0
    bottom = torch.zeros(8, 8)
    bottom[4:] = 1.0

    loss_top = compute_loss(comp=comp, target=target, focus_map=top)
    loss_bottom = compute_loss(comp=comp, target=target, focus_map=bottom)
    assert loss_top.item() > loss_bottom.item()


def test_focus_map_values_are_clamped_to_unit_range():
    comp = torch.zeros(4, 4, 3)
    target = torch.full((4, 4, 3), 128.0)
    within = compute_loss(comp=comp, target=target, focus_map=torch.ones(4, 4))
    over = compute_loss(comp=comp, target=target, focus_map=torch.full((4, 4), 5.0))
    assert torch.allclose(within, over, atol=1e-4)


def test_alpha_mask_excludes_transparent_pixels():
    comp = torch.zeros(8, 8, 3)
    target = torch.zeros(8, 8, 3)
    target[:4] = 255.0  # error confined to the top half

    alpha_hide_error = torch.full((8, 8), 255.0)
    alpha_hide_error[:4] = 0.0  # mask out exactly the erroring pixels
    masked = compute_loss(comp=comp, target=target, alpha=alpha_hide_error)
    assert masked.item() < 1e-6

    alpha_show_error = torch.zeros(8, 8)
    alpha_show_error[:4] = 255.0
    shown = compute_loss(comp=comp, target=target, alpha=alpha_show_error)
    assert shown.item() > 1.0


def test_smoothness_penalty_adds_cost_for_rough_height_map():
    comp = torch.zeros(8, 8, 3)
    target = torch.zeros(8, 8, 3)
    smooth = torch.zeros(8, 8)
    rough = torch.randn(8, 8) * 5.0
    base = compute_loss(comp=comp, target=target, pixel_height_logits=smooth,
                        add_penalty_loss=1.0)
    penalised = compute_loss(comp=comp, target=target, pixel_height_logits=rough,
                             add_penalty_loss=1.0)
    assert penalised.item() > base.item()


def test_target_lab_conversion_is_cached_on_the_tensor():
    comp = torch.randint(0, 256, (8, 8, 3), dtype=torch.float32)
    target = torch.randint(0, 256, (8, 8, 3), dtype=torch.float32)
    assert not hasattr(target, "_af_lab_cache")
    compute_loss(comp=comp, target=target)
    assert hasattr(target, "_af_lab_cache")
    assert target._af_lab_cache.shape == target.shape


# --------------------------------------------------------------------------
# loss_fn (full forward: composite + loss)
# --------------------------------------------------------------------------


def test_loss_fn_returns_finite_scalar():
    material_colors, material_TDs, background = _mats()
    target = torch.randint(0, 256, (16, 16, 3), dtype=torch.float32)
    params = {
        "pixel_height_logits": torch.zeros(16, 16),
        "global_logits": torch.zeros(8, 3),
    }
    out = loss_fn(
        params, target, tau_height=0.5, tau_global=0.5, h=0.2, max_layers=8,
        material_colors=material_colors, material_TDs=material_TDs, background=background,
    )
    assert out.dim() == 0
    assert torch.isfinite(out)


def test_loss_fn_is_differentiable_wrt_params():
    material_colors, material_TDs, background = _mats()
    target = torch.rand(12, 12, 3) * 255.0
    params = {
        "pixel_height_logits": torch.zeros(12, 12, requires_grad=True),
        "global_logits": torch.randn(6, 3, requires_grad=True),
    }
    out = loss_fn(
        params, target, tau_height=1.0, tau_global=1.0, h=0.2, max_layers=6,
        material_colors=material_colors, material_TDs=material_TDs, background=background,
    )
    out.backward()
    assert torch.isfinite(params["pixel_height_logits"].grad).all()
    assert torch.isfinite(params["global_logits"].grad).all()
    assert params["global_logits"].grad.abs().sum() > 0


def test_loss_fn_low_memory_path_matches_default():
    torch.manual_seed(0)
    material_colors, material_TDs, background = _mats()
    target = torch.rand(10, 10, 3) * 255.0
    params = {
        "pixel_height_logits": torch.zeros(10, 10),
        "global_logits": torch.full((5, 3), -5.0),
    }
    params["global_logits"][:, 0] = 5.0
    gumbel_exp = torch.empty(5, 3).exponential_(1.0)
    kw = dict(
        tau_height=0.4, tau_global=0.4, h=0.2, max_layers=5,
        material_colors=material_colors, material_TDs=material_TDs,
        background=background, gumbel_exp=gumbel_exp,
    )
    default = loss_fn(params, target, low_memory=False, **kw)
    low = loss_fn(params, target, low_memory=True, **kw)
    assert torch.allclose(default, low, atol=1e-3)


def test_target_lab_cache_is_full_precision_even_when_first_built_under_autocast():
    # Training on CPU runs under bf16 autocast, and the first compute_loss
    # call cached the target's Lab at bf16 precision (~0.9 Lab off). Every
    # later loss - the reported final loss, pruning - reused that, so a result
    # rebuilt from disk (fresh cache) scored differently than the live one.
    g = torch.Generator().manual_seed(0)
    target = torch.rand(16, 16, 3, generator=g) * 255
    comp = torch.rand(16, 16, 3, generator=g) * 255
    expected = compute_loss(comp, target.clone()).item()

    with torch.autocast("cpu", dtype=torch.bfloat16):
        compute_loss(comp, target)
    assert compute_loss(comp, target).item() == pytest.approx(expected, rel=1e-6)
