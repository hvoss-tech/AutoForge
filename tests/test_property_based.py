import pytest
import torch

pytest.importorskip("hypothesis")

from hypothesis import given, strategies as st, settings

from autoforge.Helper.FilamentHelper import hex_to_rgb
from autoforge.Helper.OptimizerHelper import adaptive_round, deterministic_rand_like


@given(st.text(alphabet="0123456789ABCDEFabcdef", min_size=6, max_size=6))
@settings(max_examples=25)
def test_hex_to_rgb_bounds(hex_part):
    rgb = hex_to_rgb("#" + hex_part)
    assert len(rgb) == 3
    for v in rgb:
        assert 0.0 <= v <= 1.0


@given(
    x=st.lists(
        st.floats(min_value=-5, max_value=5, allow_nan=False, allow_infinity=False),
        min_size=1,
        max_size=20,
    ),
    tau=st.floats(min_value=0.0, max_value=1.0),
)
@settings(max_examples=40)
def test_adaptive_round_brackets(x, tau):
    t = torch.tensor(x, dtype=torch.float32)
    out = adaptive_round(t, tau=tau, high_tau=1.0, low_tau=0.0, temp=0.1)
    floor = torch.floor(t)
    ceil = torch.ceil(t)
    # allow small numerical tolerance and soft overshoot (<0.01)
    assert torch.all(out + 1e-5 >= floor)
    assert torch.all(out <= ceil + 1e-2)


@given(
    shape=st.tuples(st.integers(1, 5), st.integers(1, 5)), seed=st.integers(0, 10000)
)
@settings(max_examples=30)
def test_deterministic_rand_like_repro(shape, seed):
    t = torch.empty(shape)
    a = deterministic_rand_like(t, seed)
    b = deterministic_rand_like(t, seed)
    assert torch.allclose(a, b)
    c = deterministic_rand_like(t, seed + 1)
    # Very low probability they are identical; allow inequality
    assert not torch.allclose(a, c)
