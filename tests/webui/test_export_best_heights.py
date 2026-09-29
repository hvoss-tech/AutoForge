"""export_results carries the trained per-pixel heights to the output
resolution instead of restoring the plain init (which dropped the heights the
intermediate stack search refines, while keeping the stack chosen for them:
wrong colors in the result and every preview after it)."""
import numpy as np
import torch

from autoforge.webui.helpers.pipeline_runner import _best_heights_at_output_res


def test_same_resolution_keeps_the_trained_heights():
    init = np.random.default_rng(0).normal(size=(6, 8)).astype(np.float32)
    best = torch.from_numpy(init) + torch.randn(6, 8)
    out = _best_heights_at_output_res(best, init, torch.from_numpy(init), None)
    assert torch.allclose(out, best)


def test_untouched_logits_give_the_init():
    full = np.random.default_rng(1).normal(size=(8, 10)).astype(np.float32)
    proc = full[::2, ::2].copy()
    out = _best_heights_at_output_res(torch.from_numpy(proc), proc, torch.from_numpy(full), None)
    assert torch.equal(out, torch.from_numpy(full))


def test_refined_pixels_carry_over_nearest_and_alpha_stays_masked():
    full = np.zeros((4, 4), np.float32)
    proc = full[::2, ::2].copy()
    best = torch.from_numpy(proc).clone()
    best[0, 1] = 3.0
    alpha = np.full((4, 4), 255, np.uint8)
    alpha[0, 3] = 0
    out = _best_heights_at_output_res(best, proc, torch.from_numpy(full), alpha)
    assert out[0:2, 2:4].tolist() == [[3.0, 0.0], [3.0, 3.0]]
    assert float(out[0:2, 0:2].abs().sum()) == 0.0
