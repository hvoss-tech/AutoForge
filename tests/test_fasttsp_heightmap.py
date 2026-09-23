"""Tests for the MST-path height-map initializer
(autoforge.Helper.Heightmaps.FastTSPHeightMap)."""

import numpy as np
import pytest

pytest.importorskip("sklearn")
pytest.importorskip("skimage")
pytest.importorskip("scipy")

from autoforge.Helper.Heightmaps.FastTSPHeightMap import (
    _two_opt_refine,
    init_height_map,
    run_init_threads,
    tsp_order_mst_path,
)


def _blocky_image(H=40, W=40):
    """Four solid colour quadrants -> clean, well-separated clusters."""
    img = np.zeros((H, W, 3), dtype=np.float32)
    img[: H // 2, : W // 2] = (20, 20, 20)
    img[: H // 2, W // 2 :] = (200, 30, 30)
    img[H // 2 :, : W // 2] = (30, 200, 30)
    img[H // 2 :, W // 2 :] = (230, 230, 230)
    return img


# --------------------------------------------------------------------------
# tsp_order_mst_path
# --------------------------------------------------------------------------


def test_tsp_order_mst_path_pins_endpoints_and_keeps_all_nodes():
    labs = np.array(
        [[0, 0, 0], [30, 0, 0], [60, 0, 0], [90, 0, 0], [45, 10, 0]], dtype=float
    )
    order = tsp_order_mst_path([0, 1, 2, 3, 4], labs, bg=0, fg=3)
    assert order[0] == 0 and order[-1] == 3
    assert set(order) == {0, 1, 2, 3, 4}


def test_tsp_order_mst_path_recovers_the_monotone_chain():
    labs = np.array([[float(i), 0.0, 0.0] for i in range(6)], dtype=float)
    order = tsp_order_mst_path(list(range(6)), labs, bg=0, fg=5)
    assert order == [0, 1, 2, 3, 4, 5]


def test_tsp_order_mst_path_adds_missing_endpoint_nodes():
    labs = np.array([[0, 0, 0], [10, 0, 0], [20, 0, 0]], dtype=float)
    order = tsp_order_mst_path([1], labs, bg=0, fg=2)
    assert order[0] == 0 and order[-1] == 2
    assert set(order) == {0, 1, 2}


def test_two_opt_refine_does_not_worsen_a_tour():
    rng = np.random.default_rng(0)
    pts = rng.random((8, 2))
    from scipy.spatial.distance import cdist

    D = cdist(pts, pts)

    def length(path):
        return sum(D[path[i], path[i + 1]] for i in range(len(path) - 1))

    start = [0, 5, 2, 7, 1, 4, 3, 6]
    refined = _two_opt_refine(list(start), D, max_passes=20)
    assert set(refined) == set(start)
    assert length(refined) <= length(start) + 1e-9


# --------------------------------------------------------------------------
# init_height_map
# --------------------------------------------------------------------------


def test_init_height_map_shapes_and_range():
    img = _blocky_image()
    logits, global_logits, ordering_metric, cluster_layers, sil, labels = init_height_map(
        img,
        max_layers=8,
        h=0.2,
        background_tuple=[20, 20, 20],
        random_seed=0,
        cluster_layers=4,
    )
    assert logits.shape == img.shape[:2]
    assert labels.shape == img.shape[:2]
    assert np.isfinite(logits).all()
    recon = 1.0 / (1.0 + np.exp(-logits))
    assert recon.min() >= 0.0 and recon.max() <= 1.0
    assert ordering_metric >= 0.0
    assert -1.0 <= sil <= 1.0
    assert set(np.unique(labels)).issubset(set(range(4)))


def test_init_height_map_background_quadrant_is_lowest():
    img = _blocky_image()
    logits, *_ = init_height_map(
        img, max_layers=8, h=0.2, background_tuple=[20, 20, 20],
        random_seed=0, cluster_layers=4,
    )
    recon = 1.0 / (1.0 + np.exp(-logits))
    bg_quadrant = recon[:20, :20].mean()          # the (20,20,20) block == background
    white_quadrant = recon[20:, 20:].mean()       # the (230,230,230) block, most distinct
    assert bg_quadrant < white_quadrant
    assert bg_quadrant == pytest.approx(recon.min(), abs=1e-3)


def test_init_height_map_is_deterministic_for_a_fixed_seed():
    img = _blocky_image()
    a = init_height_map(img, 8, 0.2, [20, 20, 20], random_seed=3, cluster_layers=4)[0]
    b = init_height_map(img, 8, 0.2, [20, 20, 20], random_seed=3, cluster_layers=4)[0]
    assert np.allclose(a, b)


def test_run_init_threads_picks_a_valid_best_result():
    img = _blocky_image(36, 36)
    logits, global_logits, labels = run_init_threads(
        img,
        max_layers=8,
        h=0.2,
        background_tuple=(20 / 255, 20 / 255, 20 / 255),
        random_seed=0,
        num_threads=1,
        num_runs=3,
        init_method="kmeans",
        cluster_layers=4,
    )
    assert logits.shape == img.shape[:2]
    assert labels.shape == img.shape[:2]
    assert np.isfinite(logits).all()
    recon = 1.0 / (1.0 + np.exp(-logits))
    assert recon.min() >= 0.0 and recon.max() <= 1.0


def test_init_height_map_emits_per_layer_material_logits_when_palette_given():
    img = _blocky_image()
    material_colors = np.array(
        [[0.1, 0.1, 0.1], [0.8, 0.1, 0.1], [0.1, 0.8, 0.1], [0.9, 0.9, 0.9]],
        dtype=np.float32,
    )
    out = init_height_map(
        img, max_layers=8, h=0.2, background_tuple=[20, 20, 20],
        random_seed=0, cluster_layers=4, material_colors=material_colors,
    )
    global_logits = out[1]
    assert global_logits is not None
    assert np.asarray(global_logits).shape == (8, 4)
