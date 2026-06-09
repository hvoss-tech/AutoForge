import random
from typing import Optional

import numpy as np
from joblib import Parallel, delayed
from scipy.spatial.distance import cdist
from skimage.color import rgb2lab
from sklearn.cluster import MiniBatchKMeans

from autoforge.Helper.Heightmaps.ChristofidesHeightMap import (
    _compute_distinctiveness,
    segmentation_quality,
    compute_ordering_metric,
    create_mapping,
    interpolate_arrays,
)


# ---------------------------------------------------------------------------
# Split two‑stage K‑Means so the expensive over‑clustering (Stage 1) runs
# only once, while the cheap Stage 2 + ordering runs N times in parallel.
# ---------------------------------------------------------------------------


def _compute_overclustering(
    pixels: np.ndarray,
    overcluster_k: int = 500,
    random_state: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Stage 1: heavy over‑segmentation on the full pixel array.

    Returns (over_cluster_centroids, over_cluster_labels) where labels
    are per-pixel assignments (avoids redoing the expensive cdist in Stage 2).
    """
    kmeans = MiniBatchKMeans(
        n_clusters=overcluster_k,
        random_state=random_state,
        max_iter=300,
    )
    kmeans.fit(pixels)
    return kmeans.cluster_centers_, kmeans.labels_  # type: ignore[return-value]


def _refine_clusters(
    pixels: np.ndarray,
    H: int,
    W: int,
    centroids1: np.ndarray,
    labels1: np.ndarray,
    final_k: int,
    beta_distinct: float = 4.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Stage 2: weighted K‑Means on over‑cluster centroids + pixel assignment.

    *labels1* is the per-pixel over-cluster assignment from Stage 1
    (avoids re-computing the expensive pixel→centroid distances).

    Returns (final_centroids, labels) where labels is (H, W).
    """
    from sklearn.cluster import KMeans

    counts1 = np.bincount(labels1, minlength=centroids1.shape[0]).astype(np.float64)

    distinct = _compute_distinctiveness(centroids1)
    if distinct.max() > 0:
        distinct /= distinct.max()
    weights = counts1 * (1.0 + beta_distinct * distinct)

    kmeans2 = KMeans(n_clusters=final_k, random_state=0, n_init="auto")
    kmeans2.fit(centroids1, sample_weight=weights)
    centroids_final = kmeans2.cluster_centers_

    chunk = 2**18
    labels_final = np.empty(pixels.shape[0], dtype=np.int32)
    for start in range(0, pixels.shape[0], chunk):
        end = start + chunk
        d = cdist(pixels[start:end], centroids_final, metric="euclidean")
        labels_final[start:end] = np.argmin(d, axis=1)

    return centroids_final, labels_final.reshape(H, W)


def _minimum_spanning_tree_prim(n: int, D: np.ndarray) -> tuple[list[list[tuple[int, float]]], list[int]]:
    """Prim's MST on distance matrix D (n×n). Returns (adj, parent)."""
    visited = [False] * n
    key = [np.inf] * n
    parent = [-1] * n
    key[0] = 0.0

    for _ in range(n):
        u = min((i for i in range(n) if not visited[i]), key=lambda i: key[i])
        visited[u] = True
        for v in range(n):
            if not visited[v] and D[u, v] < key[v]:
                key[v] = D[u, v]
                parent[v] = u

    adj: list[list[tuple[int, float]]] = [[] for _ in range(n)]
    for v in range(1, n):
        u = parent[v]
        w = D[u, v]
        adj[u].append((v, w))
        adj[v].append((u, w))
    return adj, parent


def _tree_path(adj: list[list[tuple[int, float]]], start: int, end: int) -> list[int]:
    """Unique path between start and end in a tree via DFS."""
    parent = {start: None}
    stack = [start]
    while stack:
        u = stack.pop()
        if u == end:
            break
        for v, _ in adj[u]:
            if v not in parent:
                parent[v] = u
                stack.append(v)

    path = []
    u = end
    while u is not None:
        path.append(u)
        u = parent[u]
    return path[::-1]


def _insert_remaining(path_global: list[int], D: np.ndarray, unvisited: set[int]) -> list[int]:
    """Greedy best-insertion of remaining nodes, ordered by furthest-first."""
    remaining = list(unvisited)
    min_dists = [min(D[u, v] for v in path_global) for u in remaining]
    order = sorted(zip(remaining, min_dists), key=lambda x: -x[1])

    for u, _ in order:
        best_pos = -1
        best_cost = float("inf")
        for i in range(len(path_global) - 1):
            cost = D[path_global[i], u] + D[u, path_global[i + 1]] - D[path_global[i], path_global[i + 1]]
            if cost < best_cost:
                best_cost = cost
                best_pos = i + 1
        path_global.insert(best_pos, u)

    return path_global


def _two_opt_refine(path: list[int], D: np.ndarray, max_passes: int = 20) -> list[int]:
    """2-opt local search; never moves first (bg) or last (fg) node."""
    n = len(path)
    for _ in range(max_passes):
        improved = False
        for i in range(1, n - 2):
            for j in range(i + 1, n - 1):
                old = D[path[i - 1], path[i]] + D[path[j], path[j + 1]]
                new = D[path[i - 1], path[j]] + D[path[i], path[j + 1]]
                if new < old - 1e-12:
                    path[i:j + 1] = path[i:j + 1][::-1]
                    improved = True
        if not improved:
            break
    return path


def tsp_order_mst_path(nodes: list[int], labs: np.ndarray, bg: int, fg: int) -> list[int]:
    """Order clusters from bg to fg using MST-Path + 2-Opt.

    Fast replacement for Christofides-based TSP ordering.
    1. Compute MST, extract the unique bg→fg spine
    2. Insert remaining clusters at optimal positions
    3. Apply 2-opt local search for refinement
    """
    nodes = list(set(nodes) | {bg, fg})
    idx_map = {v: i for i, v in enumerate(nodes)}
    n = len(nodes)

    pts = labs[nodes]
    D = cdist(pts, pts, metric="euclidean")

    adj, _ = _minimum_spanning_tree_prim(n, D)

    bg_i = idx_map[bg]
    fg_i = idx_map[fg]

    spine_idx = _tree_path(adj, bg_i, fg_i)

    remaining = [i for i in range(n) if i not in spine_idx]

    if remaining:
        spine_idx = _insert_remaining(spine_idx, D, set(remaining))

    spine_idx = _two_opt_refine(spine_idx, D)

    spine = [nodes[i] for i in spine_idx]

    if spine[0] != bg:
        spine.remove(bg)
        spine.insert(0, bg)
    if spine[-1] != fg:
        spine.remove(fg)
        spine.append(fg)

    if len(spine) > 2:
        rev = [spine[0]] + spine[1:-1][::-1] + [spine[-1]]
        if compute_ordering_metric(rev, labs) < compute_ordering_metric(spine, labs):
            spine = rev

    return spine


def _prepare_lab_image(
    target: np.ndarray,
    lab_weights: tuple[float, float, float],
    lab_space: bool,
) -> np.ndarray:
    """Convert target RGB uint8 image to weighted Lab (N, 3)."""
    target_np = target.astype(np.float32) / 255.0
    if lab_space:
        lab = rgb2lab(target_np)
        lab[..., 0] *= lab_weights[0]
        lab[..., 1] *= lab_weights[1]
        lab[..., 2] *= lab_weights[2]
    else:
        lab = target_np
    return lab.reshape(-1, 3)


def init_height_map(
    target,
    max_layers,
    h,
    background_tuple,
    eps=1e-6,
    random_seed=None,
    lab_weights=(1.0, 1.0, 1.0),
    init_method="quantize_maxcoverage",
    cluster_layers=None,
    lab_space=True,
    material_colors=None,
    focus_map: Optional[np.ndarray] = None,
    focus_boost: float = 0.5,
    overcluster_centroids: Optional[np.ndarray] = None,
    overcluster_labels: Optional[np.ndarray] = None,
    overcluster_seed: int = 0,
):
    """Initialize pixel height logits using MST-Path + 2-Opt ordering.

    If *overcluster_centroids* and *overcluster_labels* are provided, the
    expensive Stage 1 over‑clustering is skipped.
    """
    if cluster_layers is None:
        cluster_layers = max_layers

    if random_seed is not None:
        np.random.seed(random_seed)
        random.seed(random_seed)

    H, W, _ = target.shape
    target_lab_reshaped = _prepare_lab_image(target, lab_weights, lab_space)

    if overcluster_centroids is not None and overcluster_labels is not None:
        labs, labels = _refine_clusters(
            target_lab_reshaped, H, W, overcluster_centroids, overcluster_labels,
            cluster_layers, beta_distinct=4.0,
        )
    else:
        # Fallback when no pre-computed over-clustering — run full two-stage
        from autoforge.Helper.Heightmaps.ChristofidesHeightMap import (
            two_stage_weighted_kmeans as _twosk,
        )
        labs, labels = _twosk(
            target_lab_reshaped, H, W,
            overcluster_k=500, final_k=cluster_layers,
            beta_distinct=4.0, random_state=random_seed,
        )

    if lab_space:
        target_lab_for_quality = rgb2lab((target.astype(np.float32) / 255.0))
        target_lab_for_quality[..., 0] *= lab_weights[0]
        target_lab_for_quality[..., 1] *= lab_weights[1]
        target_lab_for_quality[..., 2] *= lab_weights[2]
    else:
        target_lab_for_quality = target.astype(np.float32) / 255.0

    sil_score = segmentation_quality(
        target_lab_for_quality.reshape(-1, 3),
        labels,
        sample_size=5000,
        random_state=random_seed,
    )

    bg_rgb = np.array(background_tuple).astype(np.float32) / 255.0
    if lab_space:
        bg_lab = rgb2lab(np.array([[bg_rgb]]))[0, 0, :]
        bg_lab[0] *= lab_weights[0]
        bg_lab[1] *= lab_weights[1]
        bg_lab[2] *= lab_weights[2]
    else:
        bg_lab = bg_rgb

    distances = np.linalg.norm(labs - bg_lab, axis=1)
    bg_cluster = int(np.argmin(distances))
    fg_cluster = int(np.argmax(distances))

    unique_clusters = sorted(np.unique(labels))
    nodes = unique_clusters

    final_ordering = tsp_order_mst_path(nodes, labs, bg_cluster, fg_cluster)

    new_values = create_mapping(final_ordering, labs, unique_clusters)
    new_labels = np.vectorize(lambda x: new_values[x])(labels).astype(np.float32)

    if focus_map is not None:
        fm = np.asarray(focus_map, dtype=np.float32)
        if fm.max() > 1.0 or fm.min() < 0.0:
            fm = np.clip(fm, 0, 255) / 255.0
        if fm.shape != (H, W):
            src_h, src_w = fm.shape[:2]
            iy = (np.arange(H) * src_h / H).astype(np.int32)
            ix = (np.arange(W) * src_w / W).astype(np.int32)
            iy = np.clip(iy, 0, src_h - 1)
            ix = np.clip(ix, 0, src_w - 1)
            fm = fm[np.ix_(iy, ix)]
        new_labels = np.clip(new_labels * (1.0 + focus_boost * fm), 0.0, 1.0)

    pixel_height_logits = np.log((new_labels + eps) / (1 - new_labels + eps))
    ordering_metric = compute_ordering_metric(final_ordering, labs)
    ordering_metric /= cluster_layers

    global_logits_out = None
    if material_colors is not None:
        if lab_space:
            material_lab = rgb2lab(material_colors.reshape(1, -1, 3)).reshape(-1, 3)
            material_lab[:, 0] *= lab_weights[0]
            material_lab[:, 1] *= lab_weights[1]
            material_lab[:, 2] *= lab_weights[2]
            materials = material_colors
        else:
            materials = material_colors

        num_materials = materials.shape[0]

        global_logits = []
        for idx, label in enumerate(unique_clusters):
            t = new_values[label]
            cluster_lab = labs[label]
            dists = np.linalg.norm(materials - cluster_lab, axis=1)
            best_j = np.argmin(dists)
            out_logit = np.ones(num_materials) * -1.0
            out_logit[best_j] = 1.0
            global_logits.append((t, out_logit))

        global_logits = sorted(global_logits, key=lambda x: x[0])
        global_logits_out = interpolate_arrays(global_logits, max_layers)

    return (
        pixel_height_logits,
        global_logits_out,
        ordering_metric,
        cluster_layers,
        sil_score,
        labels.reshape(H, W),
    )


def run_init_threads(
    target,
    max_layers,
    h,
    background_tuple,
    eps=1e-6,
    random_seed=None,
    num_threads=4,
    num_runs=32,
    init_method="kmeans",
    cluster_layers=None,
    material_colors=None,
    focus_map: Optional[np.ndarray] = None,
    focus_boost: float = 0.5,
):
    background_tuple = (np.asarray(background_tuple) * 255).tolist()
    if random_seed is None:
        random_seed = np.random.randint(1e6)

    if cluster_layers is None:
        cluster_layers = max_layers

    lab_space = True

    pixels = _prepare_lab_image(target, (1.0, 1.0, 1.0), lab_space)
    print("Computing over‑clustering (Stage 1) …")
    centroids1, labels1 = _compute_overclustering(pixels, overcluster_k=500, random_state=random_seed)
    print(f"  → {centroids1.shape[0]} over‑cluster centroids computed.")
    del pixels

    def _run_one(seed_offset: int) -> tuple:
        return init_height_map(
            target, max_layers, h, background_tuple, eps,
            random_seed=random_seed + seed_offset,
            init_method=init_method, cluster_layers=cluster_layers,
            lab_space=lab_space, material_colors=material_colors,
            focus_map=focus_map, focus_boost=focus_boost,
            overcluster_centroids=centroids1,
            overcluster_labels=labels1,
            overcluster_seed=seed_offset,
        )

    if num_threads > 1:
        tasks = [delayed(_run_one)(i) for i in range(num_runs)]
        results = Parallel(n_jobs=num_threads, verbose=10)(tasks)
    else:
        results = [_run_one(i) for i in range(num_runs)]

    metrics = [(r[2] / r[3]) / (r[4] + 1e-6) for r in results]
    mean_metric = np.mean(metrics)
    std_metric = np.std(metrics)
    min_metric = np.min(metrics)
    max_metric = np.max(metrics)
    print(
        f"mean: {mean_metric}, std: {std_metric}, min: {min_metric}, max: {max_metric}"
    )
    print(f"Choosing best ordering with metric: {min_metric}")
    best_result = min(results, key=lambda x: x[2])
    print(f"Best result number of cluster layers: {best_result[3]}")
    return best_result[0], best_result[1], best_result[5]
