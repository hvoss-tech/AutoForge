#!/usr/bin/env python3
"""Run a single AutoForge pipeline invocation in-process and report timing/VRAM.

Usage mirrors the `autoforge` CLI: all args are forwarded to autoforge's own
argparser (via sys.argv). Prints a single line:

    EXPERIMENT_RESULT {"loss": ..., "elapsed_s": ..., "peak_vram_reserved_gb": ...,
                        "peak_vram_allocated_gb": ..., "peak_nvidia_smi_gb": ...,
                        "vram_samples_gb": [...]}

Peak VRAM is tracked from multiple angles since usage changes across phases
(init, solve loop, post-processing / discretization) due to resizing and
batching:
  - torch.cuda.max_memory_reserved / max_memory_allocated: allocator
    high-water marks, updated continuously by torch itself (no polling gaps).
  - A background thread samples `nvidia-smi` every 150ms for this PID and
    keeps every sample plus the max, as an external, allocator-independent
    cross-check that also captures non-torch (CUDA context, cuDNN workspace)
    memory.
"""
import json
import os
import subprocess
import sys
import threading
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

# Mirror what auto_forge.py itself sets (see its module docstring-adjacent
# comment) - must happen before `import torch`, since a real `autoforge` CLI
# invocation imports torch for the first time inside auto_forge.py, and this
# harness's own `import torch` below would otherwise lock in the allocator
# config before that line ever runs.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

if os.environ.get("AF_DETERMINISTIC") == "1":
    # Benchmark-only: make repeated runs bit-reproducible so small loss
    # differences between code versions are measurable.
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch

if os.environ.get("AF_DETERMINISTIC") == "1":
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

from autoforge.Helper.DeviceUtils import synchronize
# `autoforge.auto_forge` (and its transitive imports, e.g. Optimizer.py) is
# intentionally imported inside main(), after t0 starts - some of those
# imports are lazy/conditional on CLI flags (matplotlib, tensorboard), and a
# real CLI invocation pays that cost as part of "time to get the STL", so the
# benchmark should count it too instead of hiding it in module-import time.


def _poll_nvidia_smi(stop_event: threading.Event, samples: list) -> None:
    pid = str(os.getpid())
    while not stop_event.is_set():
        try:
            out = subprocess.check_output(
                [
                    "nvidia-smi",
                    "--query-compute-apps=pid,used_memory",
                    "--format=csv,noheader,nounits",
                ],
                timeout=2,
            ).decode()
            for line in out.strip().splitlines():
                parts = [p.strip() for p in line.split(",")]
                if len(parts) == 2 and parts[0] == pid:
                    mb = float(parts[1])
                    samples.append(mb / 1e3)
        except Exception:
            pass
        stop_event.wait(0.15)


def main() -> None:
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    nvidia_smi_samples: list = []
    stop_event = threading.Event()
    poller = threading.Thread(
        target=_poll_nvidia_smi, args=(stop_event, nvidia_smi_samples), daemon=True
    )
    if torch.cuda.is_available():
        poller.start()

    t0 = time.perf_counter()
    from autoforge import auto_forge
    from autoforge.auto_forge import parse_args, start

    # Keep the final height map (layers) for the roughness check.
    _generate_stl = auto_forge.generate_stl
    heights = {}

    def _capture_stl(height_map_mm, *a, **k):
        heights["mm"] = height_map_mm
        return _generate_stl(height_map_mm, *a, **k)

    auto_forge.generate_stl = _capture_stl

    args = parse_args()
    final_loss = start(args)

    # Sync before stopping the clock, or an async backend's queued work lands
    # outside the measurement. (The VRAM instrumentation below stays
    # NVIDIA-specific on purpose - it polls nvidia-smi.)
    synchronize()
    elapsed = time.perf_counter() - t0
    stop_event.set()
    if torch.cuda.is_available():
        poller.join(timeout=3)

    peak_reserved_gb = (
        torch.cuda.max_memory_reserved() / 1e9 if torch.cuda.is_available() else 0.0
    )
    peak_allocated_gb = (
        torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else 0.0
    )
    peak_nvidia_smi_gb = max(nvidia_smi_samples) if nvidia_smi_samples else 0.0

    result = {
        "loss": float(final_loss),
        "elapsed_s": elapsed,
        "peak_vram_reserved_gb": peak_reserved_gb,
        "peak_vram_allocated_gb": peak_allocated_gb,
        "peak_nvidia_smi_gb": peak_nvidia_smi_gb,
        "num_nvidia_smi_samples": len(nvidia_smi_samples),
    }
    if "mm" in heights:
        # Printability: share of pixels >= 3 layers off their 3x3 median.
        import numpy as np
        from scipy.ndimage import median_filter

        z = np.round(heights["mm"] / args.layer_height)
        result["roughness"] = float((np.abs(z - median_filter(z, size=3, mode="nearest")) >= 3).mean())
        np.save(os.path.join(args.output_folder, "final_heights.npy"), z.astype(np.int16))
    print("EXPERIMENT_RESULT " + json.dumps(result))


if __name__ == "__main__":
    main()
