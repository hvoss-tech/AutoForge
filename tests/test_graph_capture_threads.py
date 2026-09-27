"""Regression: CUDA graph capture of repeated composites from a fresh thread.

Pruning scores candidates from joblib worker threads. The graph replay in
``_replay_captured`` used to capture as soon as a call had been seen a few
times - on whichever thread made the next call. A thread that had never run
the composite had no cuBLAS handle yet, so the handle was created during
capture, which fails (CUBLAS_STATUS_NOT_INITIALIZED) and invalidates the
capture - the webui's pruning crashed right after an optimization.
"""
import threading

import pytest
import torch

from autoforge.Helper.OptimizerHelper import composite_graph_scope, composite_image_disc

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


def _args():
    dev = "cuda"
    g = torch.Generator(device="cpu").manual_seed(0)
    L, M, H, W = 20, 5, 32, 40
    logits = torch.randn(H, W, generator=g).to(dev)
    glog = torch.randn(L, M, generator=g).to(dev)
    colors = torch.rand(M, 3, generator=g).to(dev)
    tds = (torch.rand(M, generator=g) * 5 + 0.5).to(dev)
    bg = torch.zeros(3, device=dev)
    return logits, glog, 0.01, 0.01, 0.04, L, colors, tds, bg


def test_capture_from_a_thread_that_never_ran_the_composite():
    args = _args()
    with composite_graph_scope():
        # Warm-up calls on this thread only.
        ref = [composite_image_disc(*args, rng_seed=1) for _ in range(3)][-1]
        out, err = {}, {}

        def worker():
            try:
                # First call on this thread: the old code captured right here.
                out["a"] = composite_image_disc(*args, rng_seed=1)
                out["b"] = composite_image_disc(*args, rng_seed=1)
                torch.cuda.synchronize()
            except Exception as e:  # pragma: no cover - the regression
                err["e"] = e

        t = threading.Thread(target=worker)
        t.start()
        t.join()
    assert "e" not in err, err.get("e")
    torch.testing.assert_close(out["a"], ref)
    torch.testing.assert_close(out["b"], ref)
