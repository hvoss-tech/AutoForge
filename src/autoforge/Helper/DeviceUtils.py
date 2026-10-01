"""Backend-agnostic device selection and per-backend capability helpers.

AutoForge runs on four PyTorch backends:

* **CUDA** - NVIDIA GPUs.
* **ROCm** - AMD GPUs. PyTorch exposes ROCm *through the ``torch.cuda``
  namespace*: ``torch.cuda.is_available()`` is True, the device type string is
  still ``"cuda"``, and ``torch.version.hip`` is what distinguishes the two.
  So "is this device.type == 'cuda'" is the right question for almost
  everything, and ``is_rocm()`` only matters for the handful of features whose
  support genuinely differs (graph capture, TF32).
* **MPS** - Apple Metal on Apple Silicon. Its own device type, no
  ``torch.cuda`` anything.
* **CPU** - the fallback.

Everything that needs to branch on "which accelerator is this" goes through
here, so backend knowledge lives in one file instead of being spread over bare
``torch.cuda.*`` calls that silently no-op on the other three backends (that
silence is the trap: ``torch.cuda.empty_cache()`` on an MPS run does nothing
at all and frees no Metal memory, but it also raises nothing, so the bug is
invisible).
"""

from __future__ import annotations

import os
from typing import Optional

import torch

# Device specs that mean "figure it out yourself".
_AUTO_SPECS = {"", "auto", "default", "none"}


def is_rocm() -> bool:
    """True when this PyTorch build targets AMD ROCm/HIP rather than CUDA."""
    return getattr(torch.version, "hip", None) is not None


def cuda_is_available() -> bool:
    """``torch.cuda.is_available()``, but never raises on a broken install."""
    try:
        return bool(torch.cuda.is_available())
    except Exception:
        return False


def mps_is_available() -> bool:
    """True when Apple Metal is both compiled in and usable on this machine.

    ``is_built()`` is checked first: on a non-macOS wheel ``torch.backends.mps``
    exists but ``is_available()`` is the only thing that answers correctly, and
    older wheels lack one or the other entirely.
    """
    backend = getattr(torch.backends, "mps", None)
    if backend is None:
        return False
    try:
        if hasattr(backend, "is_built") and not backend.is_built():
            return False
        return bool(backend.is_available())
    except Exception:
        return False


def backend_of(device: torch.device) -> str:
    """Return one of ``"cuda"``, ``"rocm"``, ``"mps"``, ``"cpu"``.

    Unlike ``device.type`` this separates CUDA from ROCm, which share the
    ``"cuda"`` device type.
    """
    if device.type == "cuda":
        return "rocm" if is_rocm() else "cuda"
    return device.type


def has_float64(device: torch.device) -> bool:
    """False on MPS, which has no float64: creating or casting to a float64
    tensor there raises, so float64 work has to stay on the host."""
    return device.type != "mps"


def is_accelerator(device: torch.device) -> bool:
    return device.type in ("cuda", "mps")


def describe_device(device: torch.device) -> str:
    """Human-readable one-liner for startup logging."""
    backend = backend_of(device)
    if device.type == "cuda":
        try:
            name = torch.cuda.get_device_name(device)
        except Exception:
            name = "unknown GPU"
        label = "ROCm/HIP" if backend == "rocm" else "CUDA"
        return f"{device} ({label}, {name})"
    if device.type == "mps":
        return f"{device} (Apple Metal / MPS)"
    return str(device)


def resolve_device(spec: Optional[str] = None, args=None) -> torch.device:
    """Pick the device to run on.

    Resolution order:

    1. ``spec`` (from ``--device``), else the ``AUTOFORGE_DEVICE`` env var.
    2. The legacy ``--mps`` flag, kept working but no longer *required* - MPS
       is auto-detected below, so the flag is only a no-op nudge now.
    3. Auto-detect: CUDA/ROCm, then MPS, then CPU.

    An explicitly requested device that is unavailable falls back to
    auto-detection with a warning rather than dying, so a stale shell alias or
    a config file carried between machines never hard-fails a run.
    """
    if spec is None:
        spec = getattr(args, "device", None)
    if spec is None:
        spec = os.getenv("AUTOFORGE_DEVICE")

    if spec is not None and str(spec).strip().lower() not in _AUTO_SPECS:
        spec = str(spec).strip()
        try:
            requested = torch.device(spec)
        except (RuntimeError, ValueError):
            print(f"Warning: unrecognized device '{spec}'; auto-detecting instead.")
        else:
            if _device_is_usable(requested):
                return requested
            print(
                f"Warning: requested device '{spec}' is not available; "
                "auto-detecting instead."
            )

    if cuda_is_available():
        return torch.device("cuda")
    if mps_is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _device_is_usable(device: torch.device) -> bool:
    if device.type == "cuda":
        if not cuda_is_available():
            return False
        # An explicit index past the last GPU would only fail later, at the
        # first allocation, with a much less obvious message.
        if device.index is not None and device.index >= torch.cuda.device_count():
            return False
        return True
    if device.type == "mps":
        return mps_is_available()
    if device.type == "cpu":
        return True
    # Some other backend (xpu, hpu, ...): trust the user if torch knows it.
    return hasattr(torch, device.type)


def empty_cache(device: Optional[torch.device] = None) -> None:
    """Release cached-but-unused allocator blocks on ``device``'s backend.

    ``torch.cuda.empty_cache()`` is a silent no-op when CUDA was never
    initialized, so the old unconditional calls did nothing whatsoever on
    Metal - the one backend where reclaiming between phases matters most,
    since MPS shares its memory with the system.
    """
    if device is None:
        # No device in hand: clear whatever backend is actually initialized.
        if cuda_is_available():
            torch.cuda.empty_cache()
        if mps_is_available():
            _mps_empty_cache()
        return
    if device.type == "cuda":
        torch.cuda.empty_cache()
    elif device.type == "mps":
        _mps_empty_cache()


def _mps_empty_cache() -> None:
    mps = getattr(torch, "mps", None)
    if mps is not None and hasattr(mps, "empty_cache"):
        try:
            mps.empty_cache()
        except Exception:
            pass


def synchronize(device: Optional[torch.device] = None) -> None:
    """Block until every queued kernel on ``device`` has finished."""
    if device is None:
        # No device in hand: sync whatever backend is actually initialized.
        if cuda_is_available():
            torch.cuda.synchronize()
        if mps_is_available():
            _mps_synchronize()
        return
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        _mps_synchronize()


def _mps_synchronize() -> None:
    mps = getattr(torch, "mps", None)
    if mps is not None and hasattr(mps, "synchronize"):
        try:
            mps.synchronize()
        except Exception:
            pass


def supports_graph_capture(device: torch.device) -> bool:
    """True when the training step can be captured into a replayable graph.

    Only the CUDA family has graph capture at all. On ROCm the same
    ``torch.cuda.CUDAGraph`` API maps onto HIP graphs, which work on current
    ROCm builds but have a rockier history - capture is attempted there too,
    but the caller validates the replay numerically before trusting it (see
    ``FilamentOptimizer._maybe_capture_graph``), and ``AUTOFORGE_GRAPH=off``
    turns it off everywhere without touching the CLI flag.

    Metal has no equivalent, so MPS always takes the eager path.
    """
    if os.getenv("AUTOFORGE_GRAPH", "").strip().lower() in {"off", "0", "false", "no"}:
        return False
    if device.type != "cuda":
        return False
    return hasattr(torch.cuda, "CUDAGraph")


def accelerator_device() -> Optional[torch.device]:
    """The best available GPU device, or None on a CPU-only machine.

    For helpers that want to opportunistically offload a standalone chunk of
    math (e.g. the pixel->centroid assignment in the height-map initializers)
    without being handed the run's device.
    """
    if cuda_is_available():
        return torch.device("cuda")
    if mps_is_available():
        return torch.device("mps")
    return None


def max_memory_allocated(device: Optional[torch.device] = None) -> float:
    """Peak bytes allocated on the accelerator, or 0.0 when unsupported."""
    try:
        if (device is None or device.type == "cuda") and cuda_is_available():
            return float(torch.cuda.max_memory_allocated())
        if (device is None or device.type == "mps") and mps_is_available():
            mps = getattr(torch, "mps", None)
            if mps is not None and hasattr(mps, "current_allocated_memory"):
                return float(mps.current_allocated_memory())
    except Exception:
        pass
    return 0.0


def reset_peak_memory_stats(device: Optional[torch.device] = None) -> None:
    try:
        if (device is None or device.type == "cuda") and cuda_is_available():
            torch.cuda.reset_peak_memory_stats()
    except Exception:
        pass
