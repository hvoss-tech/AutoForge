"""Regression: the JIT-fused Gumbel noise used to turn NaN (from the second
call on) where sin(x)*43758 landed on an exact integer, and the argmax over a
NaN row picked material 0 - a discrete stack changed colour in single layers."""
import pytest
import torch

from autoforge.Helper.OptimizerHelper import (
    batched_layer_material_indices,
    deterministic_gumbel_noise,
)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="fused kernel differs only on CUDA")
def test_noise_finite_on_repeated_calls():
    seeds = torch.arange(75, device="cuda") + 347449  # seed hitting an exact integer
    for _ in range(4):
        assert torch.isfinite(deterministic_gumbel_noise(seeds, 14)).all()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="fused kernel differs only on CUDA")
def test_hard_logits_always_pick_their_material():
    lg = torch.full((75, 14), -1e5, device="cuda")
    lg[:, 3] = 1e5
    for _ in range(4):
        assert (batched_layer_material_indices(lg, 0.01, 347449) == 3).all()
