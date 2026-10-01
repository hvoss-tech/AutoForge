"""Tests for the backend-agnostic device layer.

These have to pass on a CUDA box, a ROCm box, an Apple Silicon Mac and a
CPU-only CI runner, so they assert on the *logic* (fallbacks, overrides,
no-ops) rather than on which backend happens to be present.
"""

import argparse

import numpy as np
import pytest
import torch

from autoforge.Helper import DeviceUtils as DU
from autoforge.Helper.OtherHelper import get_device


def _args(**kw):
    base = {"mps": False, "device": None, "random_seed": 0}
    base.update(kw)
    return argparse.Namespace(**base)


class TestBackendDetection:
    def test_backend_of_separates_cuda_and_rocm(self, monkeypatch):
        cuda = torch.device("cuda")
        monkeypatch.setattr(DU, "is_rocm", lambda: False)
        assert DU.backend_of(cuda) == "cuda"
        monkeypatch.setattr(DU, "is_rocm", lambda: True)
        assert DU.backend_of(cuda) == "rocm"

    def test_backend_of_passes_through_other_types(self):
        assert DU.backend_of(torch.device("cpu")) == "cpu"
        assert DU.backend_of(torch.device("mps")) == "mps"

    def test_is_accelerator(self):
        assert DU.is_accelerator(torch.device("cuda"))
        assert DU.is_accelerator(torch.device("mps"))
        assert not DU.is_accelerator(torch.device("cpu"))

    def test_availability_probes_never_raise(self):
        assert isinstance(DU.cuda_is_available(), bool)
        assert isinstance(DU.mps_is_available(), bool)
        assert isinstance(DU.is_rocm(), bool)

    def test_mps_unavailable_when_not_built(self, monkeypatch):
        class FakeBackend:
            @staticmethod
            def is_built():
                return False

            @staticmethod
            def is_available():
                return True  # would lie on a non-macOS wheel

        monkeypatch.setattr(torch.backends, "mps", FakeBackend)
        assert DU.mps_is_available() is False

    def test_describe_device_is_a_string_for_every_backend(self):
        for spec in ("cpu", "cuda", "mps"):
            assert isinstance(DU.describe_device(torch.device(spec)), str)


class TestResolveDevice:
    def test_prefers_cuda_then_mps_then_cpu(self, monkeypatch):
        monkeypatch.delenv("AUTOFORGE_DEVICE", raising=False)

        monkeypatch.setattr(DU, "cuda_is_available", lambda: True)
        monkeypatch.setattr(DU, "mps_is_available", lambda: True)
        assert DU.resolve_device().type == "cuda"

        monkeypatch.setattr(DU, "cuda_is_available", lambda: False)
        assert DU.resolve_device().type == "mps"

        monkeypatch.setattr(DU, "mps_is_available", lambda: False)
        assert DU.resolve_device().type == "cpu"

    def test_explicit_spec_wins(self, monkeypatch):
        monkeypatch.delenv("AUTOFORGE_DEVICE", raising=False)
        monkeypatch.setattr(DU, "cuda_is_available", lambda: True)
        assert DU.resolve_device("cpu").type == "cpu"

    def test_env_var_is_honored(self, monkeypatch):
        monkeypatch.setenv("AUTOFORGE_DEVICE", "cpu")
        monkeypatch.setattr(DU, "cuda_is_available", lambda: True)
        assert DU.resolve_device().type == "cpu"

    def test_auto_spec_falls_through_to_detection(self, monkeypatch):
        monkeypatch.delenv("AUTOFORGE_DEVICE", raising=False)
        monkeypatch.setattr(DU, "cuda_is_available", lambda: False)
        monkeypatch.setattr(DU, "mps_is_available", lambda: False)
        for spec in ("", "auto", "  AUTO  ", None):
            assert DU.resolve_device(spec).type == "cpu"

    def test_unavailable_device_falls_back_instead_of_raising(self, monkeypatch):
        """A config carried over from another machine must not kill the run."""
        monkeypatch.delenv("AUTOFORGE_DEVICE", raising=False)
        monkeypatch.setattr(DU, "cuda_is_available", lambda: False)
        monkeypatch.setattr(DU, "mps_is_available", lambda: False)
        assert DU.resolve_device("mps").type == "cpu"
        assert DU.resolve_device("cuda:3").type == "cpu"

    def test_garbage_device_string_falls_back(self, monkeypatch):
        monkeypatch.delenv("AUTOFORGE_DEVICE", raising=False)
        monkeypatch.setattr(DU, "cuda_is_available", lambda: False)
        monkeypatch.setattr(DU, "mps_is_available", lambda: False)
        assert DU.resolve_device("not-a-device").type == "cpu"


class TestGetDevice:
    def test_no_mps_flag_needed(self, monkeypatch):
        """The whole point: Metal is picked up without --mps."""
        monkeypatch.delenv("AUTOFORGE_DEVICE", raising=False)
        monkeypatch.setattr(DU, "cuda_is_available", lambda: False)
        monkeypatch.setattr(DU, "mps_is_available", lambda: True)
        assert get_device(_args(mps=False)).type == "mps"

    def test_works_without_args_at_all(self, monkeypatch):
        monkeypatch.delenv("AUTOFORGE_DEVICE", raising=False)
        assert isinstance(get_device(), torch.device)

    def test_legacy_mps_flag_still_forces_metal(self, monkeypatch):
        """--mps on a box that also has a GPU keeps its old meaning."""
        monkeypatch.delenv("AUTOFORGE_DEVICE", raising=False)
        monkeypatch.setattr(DU, "cuda_is_available", lambda: True)
        monkeypatch.setattr(DU, "mps_is_available", lambda: True)
        import autoforge.Helper.OtherHelper as OH

        monkeypatch.setattr(OH, "mps_is_available", lambda: True)
        assert get_device(_args(mps=True)).type == "mps"

    def test_legacy_mps_flag_is_ignored_without_metal(self, monkeypatch):
        monkeypatch.delenv("AUTOFORGE_DEVICE", raising=False)
        monkeypatch.setattr(DU, "cuda_is_available", lambda: False)
        monkeypatch.setattr(DU, "mps_is_available", lambda: False)
        import autoforge.Helper.OtherHelper as OH

        monkeypatch.setattr(OH, "mps_is_available", lambda: False)
        assert get_device(_args(mps=True)).type == "cpu"

    def test_device_arg_overrides_everything(self, monkeypatch):
        monkeypatch.delenv("AUTOFORGE_DEVICE", raising=False)
        monkeypatch.setattr(DU, "cuda_is_available", lambda: True)
        assert get_device(_args(device="cpu")).type == "cpu"


class TestGraphCapture:
    def test_never_captures_on_metal_or_cpu(self):
        """CUDA graphs must not be attempted on backends that lack them."""
        assert not DU.supports_graph_capture(torch.device("mps"))
        assert not DU.supports_graph_capture(torch.device("cpu"))

    def test_env_kill_switch(self, monkeypatch):
        monkeypatch.setenv("AUTOFORGE_GRAPH", "off")
        assert not DU.supports_graph_capture(torch.device("cuda"))

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA/ROCm")
    def test_enabled_on_cuda_family(self, monkeypatch):
        monkeypatch.delenv("AUTOFORGE_GRAPH", raising=False)
        assert DU.supports_graph_capture(torch.device("cuda"))


class TestCacheHelpers:
    def test_empty_cache_and_sync_are_safe_no_ops(self):
        """Must not raise on any backend, including ones not present here."""
        for spec in ("cpu", "mps", "cuda"):
            device = torch.device(spec)
            if spec == "cuda" and not torch.cuda.is_available():
                continue
            if spec == "mps" and not DU.mps_is_available():
                continue
            DU.empty_cache(device)
            DU.synchronize(device)
        DU.empty_cache()
        DU.synchronize()

    def test_memory_helpers_return_floats(self):
        assert isinstance(DU.max_memory_allocated(), float)
        DU.reset_peak_memory_stats()


class TestPrecisionManagerPortability:
    def test_no_amp_on_cpu(self):
        """CPU must stay fp32 so a CPU fallback run reproduces CPU results."""
        from autoforge.Helper.OptimizerHelper import PrecisionManager

        prec = PrecisionManager(torch.device("cpu"))
        assert prec.enabled is False
        assert prec.scaler is None
        with prec.autocast():
            x = torch.randn(4, 4)
            assert (x @ x).dtype == torch.float32

    def test_gpu_capability_is_zero_off_cuda(self):
        from autoforge.Helper.OptimizerHelper import _gpu_capability, _has_fp16

        assert _gpu_capability(torch.device("cpu")) == 0
        assert _gpu_capability(torch.device("mps")) == 0
        assert not _has_fp16(torch.device("cpu"))


class CudaTouched(BaseException):
    """Raised by the trap below.

    Deliberately a BaseException: graph capture is wrapped in
    ``except Exception`` so it can fall back to eager steps on odd drivers,
    and an ordinary exception would be swallowed there - making the test pass
    for the wrong reason even with the device guard removed.
    """


def _trap_cuda(monkeypatch):
    """Make every torch.cuda entry point the graph machinery uses explode."""

    def boom(*a, **kw):
        raise CudaTouched("torch.cuda touched on a non-CUDA device")

    for name in ("CUDAGraph", "graph", "empty_cache", "synchronize"):
        monkeypatch.setattr(torch.cuda, name, boom)


class TestNonCudaRunNeverTouchesCuda:
    """The CUDA-graph work must be inert on Metal and CPU.

    Rather than trusting the ``device.type`` guard by inspection, this runs a
    real optimizer on a non-CUDA device with every ``torch.cuda`` entry point
    the graph machinery uses booby-trapped to raise. Anything that reaches for
    CUDA on an MPS/CPU run fails the test loudly instead of failing on a Mac.
    """

    @staticmethod
    def _build_optimizer(device):
        import types

        import numpy as np

        from autoforge.Modules.Optimizer import FilamentOptimizer

        args = types.SimpleNamespace(
            iterations=8,
            warmup_fraction=1.0,
            learning_rate_warmup_fraction=0.0,
            init_tau=1.0,
            final_tau=0.5,
            learning_rate=0.01,
            layer_height=0.04,
            max_layers=4,
            visualize=False,
            tensorboard=False,
            disable_visualization_for_gradio=1,
            output_folder="/tmp",
            cuda_graph=True,  # explicitly on: the guard, not the flag, must stop it
        )
        H = W = 16
        return FilamentOptimizer(
            args=args,
            # The caller puts these on the device (as auto_forge does);
            # FilamentOptimizer doesn't move them.
            target=(torch.rand(H, W, 3) * 255.0).to(device),
            pixel_height_logits_init=np.zeros((H, W), dtype=np.float32),
            pixel_height_labels=np.zeros((H, W), dtype=np.int32),
            global_logits_init=np.zeros((4, 3), dtype=np.float32),
            material_colors=torch.rand(3, 3).to(device),
            material_TDs=torch.ones(3).to(device),
            background=torch.zeros(3).to(device),
            device=device,
            perception_loss_module=None,
        )

    def test_cpu_run_makes_no_cuda_calls(self, monkeypatch):
        _trap_cuda(monkeypatch)
        opt = self._build_optimizer(torch.device("cpu"))
        # Past _graph_capture_after (3), so capture would have been attempted.
        for _ in range(6):
            opt.step()
        opt.release_cuda_graph()

        assert opt._graph is None
        assert torch.isfinite(opt.loss).all()

    @pytest.mark.skipif(not DU.mps_is_available(), reason="Requires Apple Metal")
    def test_mps_run_makes_no_cuda_calls(self, monkeypatch):
        _trap_cuda(monkeypatch)
        opt = self._build_optimizer(torch.device("mps"))
        for _ in range(6):
            opt.step()
        opt.release_cuda_graph()

        assert opt._graph is None
        assert torch.isfinite(opt.loss).all()


def _emulate_mps_on_cuda(monkeypatch):
    """Select the no-float64 (MPS) code paths on a CUDA device."""
    monkeypatch.setattr(DU, "has_float64", lambda device: device.type == "cpu")


def _no_float64_on_gpu():
    """Dispatch mode that fails on any float64 CUDA tensor, as MPS would."""
    from torch.utils._python_dispatch import TorchDispatchMode
    from torch.utils._pytree import tree_flatten

    class NoFloat64OnGpu(TorchDispatchMode):
        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            out = func(*args, **(kwargs or {}))
            for t in tree_flatten(out)[0]:
                if isinstance(t, torch.Tensor) and t.is_cuda and t.dtype == torch.float64:
                    raise AssertionError(f"float64 tensor on the GPU from {func}")
            return out

    return NoFloat64OnGpu()


class TestStackSearchWithoutFloat64:
    """MPS has no float64 at all: any float64 tensor on the device raises
    there. The stack searches used to keep their bookkeeping in float64 on the
    device, which crashed the pixel-refine stage on Apple Silicon.

    Off a Mac, MPS's restriction is emulated on CUDA: a dispatch hook fails on
    any float64 GPU tensor, with the MPS code paths selected (no Triton,
    fp32 decisions)."""

    @staticmethod
    def _trained_optimizer(device):
        torch.manual_seed(0)
        opt = TestNonCudaRunNeverTouchesCuda._build_optimizer(device)
        for _ in range(3):
            opt.step()
        return opt

    @staticmethod
    def _run_searches(opt, device):
        from autoforge.Helper.PixelHeightRefine import palette_search_stack, search_stack

        dg, _ = opt.get_discretized_solution(best=False)
        dg = search_stack(opt, rounds=3, batch=4, init_dg=dg, verbose=False)
        dg = palette_search_stack(opt, dg, rounds=1, sweeps=1)
        assert dg.device.type == device.type
        return dg

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires a CUDA-family GPU")
    def test_no_float64_on_device_with_mps_paths(self, monkeypatch):
        from autoforge.Helper import FusedComposite as fc

        monkeypatch.setattr(fc, "_HAS_TRITON", False)
        monkeypatch.setenv("AUTOFORGE_GRAPH", "off")
        _emulate_mps_on_cuda(monkeypatch)

        device = torch.device("cuda")
        opt = self._trained_optimizer(device)
        # Only the searches run under the hook: the TorchScript training step
        # doesn't run under a dispatch mode.
        with _no_float64_on_gpu():
            self._run_searches(opt, device)

    @pytest.mark.skipif(not DU.mps_is_available(), reason="Requires Apple Metal")
    def test_searches_run_on_mps(self):
        device = torch.device("mps")
        self._run_searches(self._trained_optimizer(device), device)

    def test_decision_dtype(self):
        from autoforge.Helper.PixelHeightRefine import _decision_dtype

        assert _decision_dtype(torch.device("mps")) == torch.float32
        assert _decision_dtype(torch.device("cuda")) == torch.float64
        assert _decision_dtype(torch.device("cpu")) == torch.float64


class TestInitWithoutFloat64:
    """The heightmap init's k-means put the float64 pixels on the device
    (``torch.as_tensor(..., device=mps)``) and summed clusters in float64
    there, so every run on Apple Silicon crashed at "Computing
    over-clustering"; the deterministic gather's backward did the same."""

    @staticmethod
    def _kmeans(device):
        from autoforge.Helper.Heightmaps._cluster import kmeans

        rng = np.random.default_rng(0)
        pixels = rng.random((5000, 3)) * 100.0  # float64, like the Lab pixels
        return kmeans(pixels, 20, seed=1, device=device)

    @staticmethod
    def _gather_backward(device):
        from autoforge.Helper.DeterministicOps import gather_plan, segment_gather

        gen = torch.Generator().manual_seed(0)
        labels = torch.randint(0, 7, (64, 64), generator=gen).to(device)
        values = torch.randn(7, generator=gen).to(device).requires_grad_()
        segment_gather(values, labels, gather_plan(labels, 7)).square().sum().backward()
        return values.grad

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires a CUDA-family GPU")
    def test_no_float64_on_device_with_mps_paths(self, monkeypatch):
        device = torch.device("cuda")
        ref_c, ref_l = self._kmeans(device)
        ref_g = self._gather_backward(device)
        _emulate_mps_on_cuda(monkeypatch)
        with _no_float64_on_gpu():
            c, l = self._kmeans(device)
            g = self._gather_backward(device)
        assert c.dtype == np.float64 and c.shape == (20, 3)
        assert (l == ref_l).mean() > 0.99
        np.testing.assert_allclose(c, ref_c, atol=1e-3)
        torch.testing.assert_close(g, ref_g)

    @pytest.mark.skipif(not DU.mps_is_available(), reason="Requires Apple Metal")
    def test_runs_on_mps(self):
        device = torch.device("mps")
        c, _ = self._kmeans(device)
        assert np.isfinite(c).all()
        assert torch.isfinite(self._gather_backward(device)).all()

    def test_has_float64(self):
        assert not DU.has_float64(torch.device("mps"))
        assert DU.has_float64(torch.device("cuda"))
        assert DU.has_float64(torch.device("cpu"))
