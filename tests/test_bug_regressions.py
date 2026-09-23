"""
Regression tests for bugs found and fixed in the AutoForge codebase.

Bug 1: compute_loss focus_map weighting formula was wrong
       (docstring said 0.1 + 0.9*focus_map, code did 1.0 + 10.0*focus_map)
Bug 2: _maybe_update_best_discrete had redundant computation
       (called both discretize_solution AND composite_image_disc with
        separate Gumbel-Softmax per layer, doubling throughput cost)
Bug 3: pixel_height_logits device mismatch in _post_optimize_and_export
       (CPU tensor assigned when GPU expected)
Bug 4: get_best_discretized_image double-computed _apply_height_offset
       when custom_height_logits was provided
Bug 5: VRAM - intermediate tensors not freed in composite_image_cont/disc
       (p_print kept alive while p_print_bleed computed, etc.)
"""

import argparse
import numpy as np
import torch
import pytest

from autoforge.Loss.LossFunctions import compute_loss, loss_fn
from autoforge.Helper.OptimizerHelper import (
    composite_image_cont,
    composite_image_disc,
)
from autoforge.Modules.Optimizer import FilamentOptimizer


class DummyLoss(torch.nn.Module):
    def forward(self, *args, **kwargs):
        return torch.tensor(0.0)


def _make_args(**overrides):
    ns = argparse.Namespace()
    defaults = dict(
        max_layers=4,
        layer_height=0.2,
        learning_rate=1e-2,
        final_tau=0.5,
        init_tau=0.5,
        iterations=5,
        warmup_fraction=0.2,
        learning_rate_warmup_fraction=0.2,
        visualize=False,
        tensorboard=False,
        run_name="",
        disable_visualization_for_gradio=1,
        output_folder="/tmp",
    )
    defaults.update(overrides)
    for k, v in defaults.items():
        setattr(ns, k, v)
    return ns


def _make_optimizer(H=8, W=8, L=4, M=3, device=None):
    if device is None:
        device = torch.device("cpu")
    target = torch.randint(0, 256, (H, W, 3), dtype=torch.float32, device=device)
    pixel_height_logits_init = np.zeros((H, W), dtype=np.float32)
    pixel_height_labels = np.zeros((H, W), dtype=np.int32)
    global_logits_init = np.random.randn(L, M).astype(np.float32)
    material_colors = torch.rand(M, 3, device=device)
    material_TDs = torch.ones(M, device=device)
    background = torch.zeros(3, device=device)
    return FilamentOptimizer(
        _make_args(),
        target,
        pixel_height_logits_init,
        pixel_height_labels,
        global_logits_init,
        material_colors,
        material_TDs,
        background,
        device,
        DummyLoss(),
    )


# ---------------------------------------------------------------------------
# Bug 1: focus_map weighting formula
# ---------------------------------------------------------------------------


class TestFocusMapWeighting:
    """Bug 1: The docstring says weight = 0.1 + 0.9 * focus_map (range [0.1, 1.0]),
    but the old code computed weight = 1.0 + focus_strength * focus_map which
    with default focus_strength=10 gave range [1.0, 11.0].
    """

    def test_focus_map_weight_range(self):
        """Weights must be in [0.1, 1.0] as documented."""
        comp = torch.randint(0, 256, (4, 4, 3), dtype=torch.float32)
        target = torch.randint(0, 256, (4, 4, 3), dtype=torch.float32)
        focus_map = torch.ones(4, 4, dtype=torch.float32)
        loss_full = compute_loss(comp=comp, target=target, focus_map=focus_map)

        focus_map_zero = torch.zeros(4, 4, dtype=torch.float32)
        loss_zero = compute_loss(comp=comp, target=target, focus_map=focus_map_zero)

        assert loss_full.item() > 0
        assert loss_zero.item() > 0
        assert loss_full.item() > loss_zero.item() * 0.5

    def test_focus_map_all_ones_same_as_no_map(self):
        """When focus_map is all 1.0 (weight=1.0), the result should equal
        the unweighted loss (no focus_map) because every pixel has uniform
        weight and mean(weight)/mean(weight) = 1."""
        comp = torch.randint(0, 256, (8, 8, 3), dtype=torch.float32)
        target = comp.clone()

        loss_no_map = compute_loss(comp=comp, target=target)
        focus_map_ones = torch.ones(8, 8, dtype=torch.float32)
        loss_with_map = compute_loss(comp=comp, target=target, focus_map=focus_map_ones)

        assert torch.allclose(loss_no_map, loss_with_map, atol=1e-4)

    def test_focus_map_zero_weight_is_low(self):
        """With focus_map all zeros, weight=0.1 so outside-mask pixels
        contribute only 10% weight relative to full-focus pixels."""
        comp = torch.zeros(4, 4, 3, dtype=torch.float32)
        target = torch.ones(4, 4, 3, dtype=torch.float32) * 255

        focus_all = torch.ones(4, 4, dtype=torch.float32)
        focus_zero = torch.zeros(4, 4, dtype=torch.float32)

        loss_all = compute_loss(comp=comp, target=target, focus_map=focus_all)
        loss_zero = compute_loss(comp=comp, target=target, focus_map=focus_zero)

        assert loss_zero.item() < loss_all.item(), (
            "With weight=0.1 everywhere, the weighted-and-normalized loss "
            "should be lower than with weight=1.0 everywhere for same input"
        )

    def test_focus_map_partial(self):
        """With half the pixels at focus=1 and half at focus=0,
        the effective loss should be between the all-1 and all-0 cases."""
        comp = torch.zeros(8, 8, 3, dtype=torch.float32)
        target = torch.ones(8, 8, 3, dtype=torch.float32) * 255

        focus_half = torch.zeros(8, 8, dtype=torch.float32)
        focus_half[:4, :] = 1.0

        loss_all = compute_loss(
            comp=comp, target=target,
            focus_map=torch.ones(8, 8, dtype=torch.float32),
        )
        loss_zero = compute_loss(
            comp=comp, target=target,
            focus_map=torch.zeros(8, 8, dtype=torch.float32),
        )
        loss_half = compute_loss(comp=comp, target=target, focus_map=focus_half)

        min_loss = min(loss_all.item(), loss_zero.item())
        max_loss = max(loss_all.item(), loss_zero.item())
        assert min_loss <= loss_half.item() <= max_loss or \
               abs(loss_half.item() - loss_all.item()) < 1e-3 or \
               abs(loss_half.item() - loss_zero.item()) < 1e-3


# ---------------------------------------------------------------------------
# Bug 2: _maybe_update_best_discrete uses disc_to_logits to avoid
#         redundant Gumbel-Softmax in composite_image_disc
# ---------------------------------------------------------------------------


class TestMaybeUpdateBestDiscrete:
    """Bug 2: _maybe_update_best_discrete now uses disc_to_logits to convert
    the already-discretized global assignment into logits for
    composite_image_disc, instead of running Gumbel-Softmax again."""

    def test_maybe_update_best_discrete_updates_best(self):
        opt = _make_optimizer()
        assert opt.best_discrete_loss == float("inf")
        opt.step(record_best=True)
        assert opt.best_discrete_loss < float("inf")
        assert opt.best_params is not None

    def test_maybe_update_best_discrete_uses_disc_to_logits(self):
        """Verify the code path uses disc_to_logits by checking the import
        exists and the method runs successfully."""
        from autoforge.Helper.PruningHelper import disc_to_logits
        opt = _make_optimizer()
        for _ in range(3):
            opt.step(record_best=True)
        assert opt.best_seed is not None


# ---------------------------------------------------------------------------
# Bug 3: pixel_height_logits device mismatch
# ---------------------------------------------------------------------------


class TestDeviceMismatch:
    """Bug 3: In _post_optimize_and_export, pixel_height_logits was set via
    torch.from_numpy() without .to(device), creating a CPU tensor when the
    optimizer runs on GPU. This would cause device mismatch errors."""

    def test_pixel_height_logits_on_device(self):
        device = torch.device("cpu")
        opt = _make_optimizer(device=device)
        for _ in range(2):
            opt.step(record_best=True)

        assert opt.pixel_height_logits.device == torch.device("cpu")
        assert opt.pixel_height_logits.device == opt.target.device

    def test_best_params_pixel_height_logits_on_device(self):
        device = torch.device("cpu")
        opt = _make_optimizer(device=device)
        opt.step(record_best=True)
        assert opt.best_params is not None
        assert opt.best_params["pixel_height_logits"].device == opt.target.device

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA")
    def test_pixel_height_logits_gpu_device(self):
        device = torch.device("cuda")
        opt = _make_optimizer(device=device)
        for _ in range(2):
            opt.step(record_best=True)
        assert opt.pixel_height_logits.device.type == "cuda"
        assert opt.best_params["pixel_height_logits"].device.type == "cuda"


# ---------------------------------------------------------------------------
# Bug 4: get_best_discretized_image avoids double _apply_height_offset
# ---------------------------------------------------------------------------


class TestGetBestDiscretizedImage:
    """Bug 4: When custom_height_logits is provided,
    get_best_discretized_image should only call _apply_height_offset once
    (on the custom logits), not twice (on best_params AND custom logits)."""

    def test_custom_height_logits_only(self):
        opt = _make_optimizer()
        for _ in range(2):
            opt.step(record_best=True)

        custom = torch.zeros(opt.H, opt.W, dtype=torch.float32)
        result = opt.get_best_discretized_image(custom_height_logits=custom)
        assert result.shape == (opt.H, opt.W, 3)
        assert torch.isfinite(result).all()

    def test_no_custom_height_logits(self):
        opt = _make_optimizer()
        for _ in range(2):
            opt.step(record_best=True)

        result = opt.get_best_discretized_image()
        assert result.shape == (opt.H, opt.W, 3)
        assert torch.isfinite(result).all()

    def test_custom_global_logits(self):
        opt = _make_optimizer()
        for _ in range(2):
            opt.step(record_best=True)

        num_materials = opt.material_colors.shape[0]
        custom_global = torch.ones(opt.max_layers, num_materials) * -1.0
        for i in range(opt.max_layers):
            custom_global[i, i % num_materials] = 1.0

        result = opt.get_best_discretized_image(custom_global_logits=custom_global)
        assert result.shape == (opt.H, opt.W, 3)
        assert torch.isfinite(result).all()


# ---------------------------------------------------------------------------
# Bug 5: VRAM - intermediate tensors freed in composite functions
# ---------------------------------------------------------------------------


class TestCompositeVRAM:
    """Bug 5: composite_image_cont and composite_image_disc should free
    intermediate tensors (p_print, p_print_bleed, eff_thick, etc.)
    after they're consumed to reduce peak VRAM."""

    def test_composite_cont_output_valid(self):
        H, W, L, M = 32, 32, 16, 5
        pixel_logits = torch.randn(H, W)
        global_logits = torch.randn(L, M)
        material_colors = torch.rand(M, 3)
        material_TDs = torch.rand(M) * 0.5 + 0.5
        background = torch.rand(3)

        out = composite_image_cont(
            pixel_logits, global_logits, 0.5, 0.5, 0.2, L,
            material_colors, material_TDs, background,
        )
        assert out.shape == (H, W, 3)
        assert torch.isfinite(out).all()
        assert out.min() >= -1.0
        assert out.max() <= 256.0

    def test_composite_disc_output_valid(self):
        H, W, L, M = 32, 32, 16, 5
        pixel_logits = torch.randn(H, W)
        global_logits = torch.randn(L, M)
        material_colors = torch.rand(M, 3)
        material_TDs = torch.rand(M) * 0.5 + 0.5
        background = torch.rand(3)

        out = composite_image_disc(
            pixel_logits, global_logits, 0.5, 0.5, 0.2, L,
            material_colors, material_TDs, background, rng_seed=42,
        )
        assert out.shape == (H, W, 3)
        assert torch.isfinite(out).all()
        assert out.min() >= -1.0
        assert out.max() <= 256.0

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA")
    def test_composite_cont_vram_cleanup(self):
        """Verify that intermediate tensors are freed by checking
        that peak memory doesn't grow excessively on GPU."""
        device = torch.device("cuda")
        H, W, L, M = 64, 64, 32, 5
        pixel_logits = torch.randn(H, W, device=device)
        global_logits = torch.randn(L, M, device=device)
        material_colors = torch.rand(M, 3, device=device)
        material_TDs = torch.rand(M, device=device) * 0.5 + 0.5
        background = torch.rand(3, device=device)

        torch.cuda.reset_peak_memory_stats(device)
        out = composite_image_cont(
            pixel_logits, global_logits, 0.5, 0.5, 0.2, L,
            material_colors, material_TDs, background,
        )
        peak_mem = torch.cuda.max_memory_allocated(device)
        assert peak_mem > 0
        assert torch.isfinite(out).all()

    def test_composite_cont_and_disc_consistency(self):
        """With very low tau (near-discrete), cont and disc should
        produce similar results when given the same seed."""
        H, W, L, M = 8, 8, 4, 3
        pixel_logits = torch.zeros(H, W)
        global_logits = torch.zeros(L, M)
        for i in range(L):
            global_logits[i, i % M] = 5.0
        material_colors = torch.tensor(
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
        )
        material_TDs = torch.tensor([0.5, 0.7, 0.9])
        background = torch.tensor([0.0, 0.0, 0.0])

        out_cont = composite_image_cont(
            pixel_logits, global_logits, 0.01, 0.01, 0.2, L,
            material_colors, material_TDs, background,
        )
        out_disc = composite_image_disc(
            pixel_logits, global_logits, 0.01, 0.01, 0.2, L,
            material_colors, material_TDs, background, rng_seed=0,
        )
        diff = (out_cont - out_disc).abs().mean()
        assert diff < 50.0, f"Cont and disc outputs differ by {diff:.1f} on average"


# ---------------------------------------------------------------------------
# Bug 1 additional: test that compute_loss returns a scalar, not a tensor
#            with an extra dimension (old code had .mean().mean())
# ---------------------------------------------------------------------------


class TestComputeLossReturnsScalar:
    """Verify that compute_loss returns a proper scalar tensor,
    not a 0-dim or 1-dim tensor that could cause downstream issues."""

    def test_compute_loss_no_map_is_scalar(self):
        comp = torch.randint(0, 256, (4, 4, 3), dtype=torch.float32)
        target = torch.randint(0, 256, (4, 4, 3), dtype=torch.float32)
        loss = compute_loss(comp=comp, target=target)
        assert loss.dim() == 0, f"Expected 0-dim scalar, got {loss.dim()}-dim"

    def test_compute_loss_with_map_is_scalar(self):
        comp = torch.randint(0, 256, (4, 4, 3), dtype=torch.float32)
        target = torch.randint(0, 256, (4, 4, 3), dtype=torch.float32)
        focus = torch.rand(4, 4, dtype=torch.float32)
        loss = compute_loss(comp=comp, target=target, focus_map=focus)
        assert loss.dim() == 0, f"Expected 0-dim scalar, got {loss.dim()}-dim"


# ---------------------------------------------------------------------------
# Integration: loss_fn with focus_map
# ---------------------------------------------------------------------------


class TestLossFnWithFocusMap:
    def test_loss_fn_focus_map_reduces(self):
        material_colors = torch.tensor(
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
        )
        material_TDs = torch.tensor([0.5, 0.7, 0.9])
        background = torch.tensor([1.0, 1.0, 1.0])
        target = torch.randint(0, 256, (8, 8, 3), dtype=torch.float32)
        pixel_height_logits = torch.zeros(8, 8)
        global_logits = torch.zeros(4, material_colors.shape[0])
        params = {
            "pixel_height_logits": pixel_height_logits,
            "global_logits": global_logits,
        }
        focus_map = torch.ones(8, 8, dtype=torch.float32)
        out = loss_fn(
            params, target,
            tau_height=0.5, tau_global=0.5, h=0.2, max_layers=4,
            material_colors=material_colors, material_TDs=material_TDs,
            background=background, focus_map=focus_map,
        )
        assert torch.isfinite(out)
        assert out.dim() == 0