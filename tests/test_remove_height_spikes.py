import itertools
import numpy as np
import torch
import pytest

from autoforge.Helper.PruningHelper import remove_height_spikes


@pytest.mark.parametrize("threshold_layers", [1, 2, 3])
@pytest.mark.parametrize("max_outliers", [1, 2])
def test_no_change_when_no_outliers(threshold_layers, max_outliers):
    base = np.array([
        [5, 5, 5],
        [5, 5, 5],
        [5, 5, 5],
    ], dtype=np.float32)
    dh = torch.tensor(base)
    cleaned, spikes = remove_height_spikes(dh, threshold_layers=threshold_layers, max_outliers=max_outliers)
    assert spikes == 0
    assert torch.equal(cleaned, dh)


def _all_heightmaps_with_k_outliers(k: int, high: float, low: float = 0.0):
    """Generate all 3x3 patterns with exactly k highs and the rest lows."""
    highs = [high] * k
    lows = [low] * (9 - k)
    for perm in set(itertools.permutations(highs + lows)):
        grid = np.array(perm, dtype=np.float32).reshape(3, 3)
        yield grid


@pytest.mark.parametrize("max_outliers", [1, 2])
@pytest.mark.parametrize("threshold_layers", [1, 2, 3])
def test_single_center_outlier_is_fixed(max_outliers, threshold_layers):
    high = threshold_layers + 5
    grid = np.zeros((3, 3), dtype=np.float32)
    grid[1, 1] = high
    dh = torch.tensor(grid)
    cleaned, spikes = remove_height_spikes(dh, threshold_layers=threshold_layers, max_outliers=max_outliers)
    assert spikes == 1
    # New center should be median of surrounding lows (which are all zero here)
    assert cleaned[1, 1].item() == pytest.approx(0.0)


@pytest.mark.parametrize("max_outliers", [1, 2])
@pytest.mark.parametrize("threshold_layers", [1, 2, 3])
def test_two_centered_outliers_allowed(max_outliers, threshold_layers):
    if max_outliers < 2:
        pytest.skip("max_outliers too small for two-outlier case")
    high = threshold_layers + 5
    # Place two highs: center and one neighbor
    for dy, dx in [(0, 1), (1, 0), (0, -1), (-1, 0), (1, 1), (-1, -1), (1, -1), (-1, 1)]:
        grid = np.zeros((3, 3), dtype=np.float32)
        grid[1, 1] = high
        grid[1 + dy, 1 + dx] = high
        dh = torch.tensor(grid)
        cleaned, spikes = remove_height_spikes(dh, threshold_layers=threshold_layers, max_outliers=max_outliers)
        # Center is always fixed
        assert cleaned[1, 1].item() == pytest.approx(0.0)
        # Center's 3x3 window has 2 outliers (≤max_outliers) so it's a spike.
        # Pass 2 may or may not fix the neighbor depending on padding:
        # - Cardinal neighbors: window has ≤2 outliers → fixed (spikes=2)
        # - Diagonal neighbors: replicate padding introduces extra outliers → not fixed (spikes=1)
        assert spikes in (1, 2)
        if dy == 0 or dx == 0:
            assert cleaned[1 + dy, 1 + dx].item() == pytest.approx(0.0)  # cardinal: fixed
        else:
            assert cleaned[1 + dy, 1 + dx].item() == pytest.approx(high)  # diagonal: not fixed


@pytest.mark.parametrize("threshold_layers", [1, 2, 3])
def test_more_than_allowed_outliers_not_changed(threshold_layers):
    high = threshold_layers + 5
    grid = np.full((3, 3), high, dtype=np.float32)
    dh = torch.tensor(grid)
    cleaned, spikes = remove_height_spikes(dh, threshold_layers=threshold_layers, max_outliers=2)
    # Too many outliers (9), so nothing should change
    assert spikes == 0
    assert torch.equal(cleaned, dh)


@pytest.mark.parametrize("threshold_layers", [1, 2, 3])
def test_center_not_outlier_not_changed(threshold_layers):
    grid = np.zeros((3, 3), dtype=np.float32)
    grid[0, 0] = threshold_layers + 5  # corner high, center low
    dh = torch.tensor(grid)
    cleaned, spikes = remove_height_spikes(dh, threshold_layers=threshold_layers, max_outliers=2)
    assert spikes == 0
    assert torch.equal(cleaned, dh)

