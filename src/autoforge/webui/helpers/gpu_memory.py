"""Releasing the GPU memory a finished pipeline result still holds.

A pipeline result (see ``pipeline_runner.build_pipeline_state``) is not a
small record: it keeps the whole ``FilamentOptimizer`` — its parameters,
Adam state, the full- and processing-resolution target images, the material
tensors — plus several more standalone tensors, all of them on the training
device. One completed run therefore pins hundreds of MB of VRAM for as long
as anything holds a reference to its result dict.

``OptimizationService`` deliberately keeps results around between Run and
Prune, but it used to keep *every* one of them, forever: optimizing image A,
switching to image B and optimizing again left A's optimizer resident, so
VRAM climbed with every image and eventually OOM'd mid-run. Only the result
being (or about to be) pruned is worth keeping, so everything older is
released through here.
"""

from __future__ import annotations

import gc
from typing import Any, Optional


def release_pipeline_result(result: Optional[dict[str, Any]]) -> None:
    """Drop every tensor a pipeline result holds and reclaim the memory.

    Clearing the dict itself matters: the caller's reference is rarely the
    only one (a job thread's local, a traceback frame), so dropping the name
    alone can leave the whole graph alive. Emptying the mapping breaks those
    remaining references to the tensors even when the dict outlives us.
    """
    if not result:
        return

    device = result.get("device")
    optimizer = result.get("optimizer")
    if optimizer is not None:
        # The optimizer may still own a captured CUDA graph and its private
        # memory pool; that pool is not reclaimed by empty_cache() while the
        # graph object is reachable. `release_cuda_graph` is the real method
        # (Modules/Optimizer.py); the other names are kept for the fakes the
        # tests use.
        for release in ("release_cuda_graph", "_release_graph", "release_graph"):
            fn = getattr(optimizer, release, None)
            if callable(fn):
                try:
                    fn()
                except Exception:
                    pass
                break
        if device is None:
            device = getattr(optimizer, "device", None)

    try:
        result.clear()
    except Exception:
        pass

    gc.collect()
    _empty_cache(device)


def _empty_cache(device: Any = None) -> None:
    """Best-effort allocator flush; never raises, and never imports torch
    just to do nothing (the webui's non-GPU tests run without a device)."""
    try:
        from autoforge.Helper.DeviceUtils import empty_cache
    except Exception:
        return
    try:
        empty_cache(device)
    except Exception:
        pass


def empty_device_cache(device: Any = None) -> None:
    """Public wrapper — used after a job finishes, where there is no result
    dict to release but the allocator is still holding freed blocks."""
    gc.collect()
    _empty_cache(device)
