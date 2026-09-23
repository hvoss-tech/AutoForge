"""Tests for autoforge.Helper.OtherHelper (set_seed, perform_basic_check).

get_device is covered separately in test_device_utils.py.
"""

import types

import numpy as np
import pytest
import torch

from autoforge.Helper.OtherHelper import perform_basic_check, set_seed


# --------------------------------------------------------------------------
# set_seed
# --------------------------------------------------------------------------


def test_set_seed_makes_rng_reproducible():
    set_seed(types.SimpleNamespace(random_seed=1234))
    a_np, a_torch = np.random.rand(5), torch.rand(5)
    set_seed(types.SimpleNamespace(random_seed=1234))
    b_np, b_torch = np.random.rand(5), torch.rand(5)
    assert np.allclose(a_np, b_np)
    assert torch.allclose(a_torch, b_torch)


def test_set_seed_returns_the_seed_it_used():
    assert set_seed(types.SimpleNamespace(random_seed=77)) == 77


def test_set_seed_zero_picks_a_nonzero_random_seed():
    seeds = {set_seed(types.SimpleNamespace(random_seed=0)) for _ in range(3)}
    assert all(s != 0 for s in seeds)


# --------------------------------------------------------------------------
# perform_basic_check
# --------------------------------------------------------------------------


def _check_args(tmp_path, **overrides):
    img = tmp_path / "in.png"
    img.write_bytes(b"not-really-a-png-but-exists")
    base = dict(
        background_height=0.4,
        layer_height=0.2,
        input_image=str(img),
        csv_file="",
        json_file="",
        priority_mask="",
    )
    base.update(overrides)
    return types.SimpleNamespace(**base)


def test_perform_basic_check_passes_for_valid_args(tmp_path):
    perform_basic_check(_check_args(tmp_path))  # must not raise / exit


def test_perform_basic_check_rejects_non_multiple_background_height(tmp_path):
    with pytest.raises(SystemExit):
        perform_basic_check(_check_args(tmp_path, background_height=0.5, layer_height=0.2))


def test_perform_basic_check_rejects_missing_input_image(tmp_path):
    with pytest.raises(SystemExit):
        perform_basic_check(_check_args(tmp_path, input_image=str(tmp_path / "nope.png")))


def test_perform_basic_check_rejects_missing_csv(tmp_path):
    with pytest.raises(SystemExit):
        perform_basic_check(_check_args(tmp_path, csv_file=str(tmp_path / "missing.csv")))


def test_perform_basic_check_rejects_missing_priority_mask(tmp_path):
    with pytest.raises(SystemExit):
        perform_basic_check(
            _check_args(tmp_path, priority_mask=str(tmp_path / "missing.png"))
        )
