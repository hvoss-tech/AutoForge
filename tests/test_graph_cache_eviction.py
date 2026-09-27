"""Regression: CUDA graph replay of composites is scoped to one thread.

The scope, cache and memory pool used to be process-wide. While pruning held
a scope in its thread, the webui's other threads (preview callback, slider
re-renders from the page) went through the same unlocked cache and pool, and
the threads' captures and evictions corrupted each other: "it->second->
use_count > 0 INTERNAL ASSERT FAILED" in capture_begin, after which every
later run in the server died with "Offset increment outside graph capture
encountered unexpectedly"."""
import threading

import pytest
import torch

from autoforge.Helper import OptimizerHelper as OH

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA graphs need a GPU")


def _fn(x):
    return (x * 2.0 + 1.0).sum(dim=0)


def _hammer(n_rounds, offset, errors):
    """Many shapes, each called often enough to be captured, cycled so the
    thread's cache keeps evicting captured graphs."""
    try:
        for _ in range(n_rounds):
            for n in range(2, 2 + OH._GRAPH_CACHE_SIZE + 3):
                # No CUDA RNG in here: PyTorch's generator is process-wide and
                # may not be drawn from while another thread captures (the
                # webui's other threads never do).
                x = torch.arange((n + offset) * 8, dtype=torch.float32, device="cuda").view(-1, 8)
                for _ in range(OH._GRAPH_WARM_CALLS + 2):
                    out = OH._replay_captured(_fn, "thread-test", [x])
                torch.testing.assert_close(out, _fn(x))
    except Exception as exc:  # pragma: no cover - the failure being guarded
        errors.append(exc)


def test_a_scope_only_affects_the_thread_that_opened_it():
    seen = {}

    def other_thread():
        # No scope here, although the main thread has one open.
        seen["depth"] = OH._graph_thread_state().depth
        x = torch.ones(3, 8, device="cuda")
        for _ in range(OH._GRAPH_WARM_CALLS + 3):
            OH._replay_captured(_fn, "other", [x])
        seen["cache"] = len(OH._graph_thread_state().cache)

    with OH.composite_graph_scope():
        t = threading.Thread(target=other_thread)
        t.start()
        t.join()
        assert OH._graph_thread_state().depth == 1
        assert len(OH._graph_thread_state().cache) == 0  # untouched by the other thread
    assert seen == {"depth": 0, "cache": 0}  # it ran eagerly
    assert OH._graph_thread_state().depth == 0


def test_concurrent_scopes_in_two_threads_do_not_corrupt_each_other():
    errors = []

    def scoped(offset):
        with OH.composite_graph_scope():
            _hammer(3, offset, errors)

    def unscoped():
        _hammer(3, 50, errors)

    threads = [threading.Thread(target=scoped, args=(0,)), threading.Thread(target=scoped, args=(20,)),
               threading.Thread(target=unscoped)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert not errors, errors
    # The process is still usable afterwards (the RNG was the casualty).
    torch.randn(4, device="cuda").exponential_()
    torch.cuda.synchronize()


def test_capture_after_every_captured_graph_was_evicted():
    errors = []
    with OH.composite_graph_scope():
        _hammer(3, 0, errors)
        torch.cuda.empty_cache()
        _hammer(1, 0, errors)
    assert not errors, errors


def test_capture_after_uncaptured_shapes_evicted_the_only_captured_graph():
    """Layer pruning's pattern: one shape gets captured, the next few shapes
    are only seen a couple of times (no capture) but push it out of the
    cache, then a new shape gets captured. In the webui server that capture
    went into the evicted graph's pool and hit the use_count assert (seen in
    tests/sliders-regression.spec.ts); standalone PyTorch happens to release
    the pool cleanly, so this checks the fresh-pool path works rather than
    reproducing the assert."""
    with OH.composite_graph_scope():
        x = torch.ones(20, 8, device="cuda")
        for _ in range(OH._GRAPH_WARM_CALLS + 2):
            OH._replay_captured(_fn, "layers", [x])
        for n in range(10, 10 + OH._GRAPH_CACHE_SIZE + 1):
            y = torch.ones(n, 8, device="cuda")
            for _ in range(OH._GRAPH_WARM_CALLS):  # warm only, never captured
                OH._replay_captured(_fn, "layers", [y])
        assert not any(e["graph"] is not None for e in OH._graph_thread_state().cache.values())
        z = torch.ones(3, 8, device="cuda")
        for _ in range(OH._GRAPH_WARM_CALLS + 2):
            out = OH._replay_captured(_fn, "layers", [z])
        torch.testing.assert_close(out, _fn(z))
    torch.randn(4, device="cuda").exponential_()
    torch.cuda.synchronize()
