"""Tests for autoforge.Helper.AmpUtils (device-aware autocast selection)."""

import pytest
import torch

from autoforge.Helper import AmpUtils
from autoforge.Helper.AmpUtils import (
    _device_key,
    get_selected_autocast,
    safe_autocast,
)

CPU = torch.device("cpu")


@pytest.fixture(autouse=True)
def clear_selection_cache():
    """get_selected_autocast memoises per device key in a module global."""
    AmpUtils._SELECTED_MAP.clear()
    yield
    AmpUtils._SELECTED_MAP.clear()


# --------------------------------------------------------------------------
# _device_key
# --------------------------------------------------------------------------


def test_device_key_distinguishes_backends():
    assert _device_key(torch.device("cpu")) == "cpu"
    assert _device_key(torch.device("mps")) == "mps"
    assert _device_key(torch.device("cuda:1")) == "cuda:1"


# --------------------------------------------------------------------------
# env override
# --------------------------------------------------------------------------


def test_amp_off_disables_autocast(monkeypatch):
    monkeypatch.setenv("AUTOFORGE_AMP", "off")
    dtype, reason = get_selected_autocast(CPU)
    assert dtype is None
    assert "off" in reason

    # matmul inside safe_autocast stays fp32
    with safe_autocast(CPU):
        out = torch.randn(4, 4) @ torch.randn(4, 4)
    assert out.dtype == torch.float32


def test_selection_is_cached_after_first_probe(monkeypatch):
    calls = []
    real = AmpUtils._select_dtype_for_device

    def counting(device):
        calls.append(device.type)
        return real(device)

    monkeypatch.setattr(AmpUtils, "_select_dtype_for_device", counting)
    a = get_selected_autocast(CPU)
    b = get_selected_autocast(CPU)
    assert a == b
    assert calls == ["cpu"]  # probed once, not twice


# --------------------------------------------------------------------------
# automatic CPU selection
# --------------------------------------------------------------------------


def test_cpu_auto_selection_is_bf16_or_fp32_never_fp16(monkeypatch):
    monkeypatch.delenv("AUTOFORGE_AMP", raising=False)
    dtype, _reason = get_selected_autocast(CPU)
    assert dtype in (None, torch.bfloat16)


def test_safe_autocast_forward_backward_never_raises_on_cpu(monkeypatch):
    monkeypatch.delenv("AUTOFORGE_AMP", raising=False)
    w = torch.randn(8, 8, requires_grad=True)
    with safe_autocast(CPU):
        loss = torch.nn.functional.mse_loss(torch.randn(8, 8) @ w, torch.randn(8, 8))
    loss.backward()
    assert w.grad is not None and torch.isfinite(w.grad).all()


@pytest.mark.parametrize("mode", ["off", "bf16", "fp16"])
def test_env_forced_modes_stay_usable_end_to_end(monkeypatch, mode):
    monkeypatch.setenv("AUTOFORGE_AMP", mode)
    w = torch.randn(8, 8, requires_grad=True)
    with safe_autocast(CPU):
        loss = torch.nn.functional.mse_loss(torch.randn(8, 8) @ w, torch.randn(8, 8))
    loss.backward()
    assert w.grad is not None and torch.isfinite(w.grad).all()
