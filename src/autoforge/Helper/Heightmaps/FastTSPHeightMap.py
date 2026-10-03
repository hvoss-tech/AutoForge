import random
from typing import Optional

import numpy as np
import torch

from autoforge.Helper.Heightmaps._cluster import cdist, kmeans, rgb2lab, weighted_kmeans

from autoforge.Helper.DeviceUtils import accelerator_device
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
    seed = random_state if random_state is not None else np.random.randint(2**31)
    return kmeans(pixels, overcluster_k, seed=seed)


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
    counts1 = np.bincount(labels1, minlength=centroids1.shape[0]).astype(np.float64)

    distinct = _compute_distinctiveness(centroids1)
    if distinct.max() > 0:
        distinct /= distinct.max()
    weights = counts1 * (1.0 + beta_distinct * distinct)

    centroids_final = weighted_kmeans(centroids1, weights, final_k, seed=0)

    labels_final = _assign_to_centroids(pixels, centroids_final)
    return centroids_final, labels_final.reshape(H, W)


def _assign_to_centroids(
    pixels: np.ndarray, centroids: np.ndarray
) -> np.ndarray:
    """Nearest-centroid assignment for every pixel, on the GPU when there is one.

    GPU pixel->centroid assignment is ~4x faster than scipy's cdist at this
    problem size (full pixel count x ~max_layers centroids) - verified to
    produce identical argmin assignments on synthetic data. Any accelerator
    will do, not just CUDA: this is a plain cdist+argmin, so Metal and ROCm
    run it just as well, and picking the device via ``accelerator_device()``
    is what stops Apple Silicon from silently taking the slow scipy path.

    Falls back to the chunked CPU path when there is no GPU (e.g.
    ``--num_init_rounds > 1`` workers on a CPU-only machine) and also when the
    GPU attempt raises, so a backend missing a ``cdist`` kernel degrades to a
    slower correct answer instead of failing the run.
    """
    gpu = accelerator_device()
    if gpu is not None:
        try:
            pixels_t = torch.as_tensor(pixels, device=gpu, dtype=torch.float32)
            centroids_t = torch.as_tensor(centroids, device=gpu, dtype=torch.float32)
            return (
                torch.argmin(torch.cdist(pixels_t, centroids_t), dim=1)
                .to(torch.int32)
                .cpu()
                .numpy()
            )
        except Exception:
            pass

    chunk = 2**18
    labels_final = np.empty(pixels.shape[0], dtype=np.int32)
    for start in range(0, pixels.shape[0], chunk):
        end = start + chunk
        d = cdist(pixels[start:end], centroids, metric="euclidean")
        labels_final[start:end] = np.argmin(d, axis=1)
    return labels_final


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
    target_lab: Optional[np.ndarray] = None,
    rank_quality: bool = True,
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
    # ``target_lab``: the caller's _prepare_lab_image of the same target
    # (with the default weights), so it is not converted a second time.
    target_lab_reshaped = target_lab if target_lab is not None else _prepare_lab_image(target, lab_weights, lab_space)

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

    # `target_lab_reshaped` above is already the (H*W, 3) weighted-Lab (or
    # weighted-RGB, if lab_space=False) array segmentation_quality wants -
    # recomputing rgb2lab() on the full image a second time here was pure
    # redundant work, done once per parallel init round (num_init_rounds x).
    # silhouette_score is O(sample_size^2); profiling showed this single call
    # dominating >80% of a round's wall time (sample_size=5000 vs. images
    # this small often having well under 5000 pixels total to begin with).
    # It's only used to *rank* num_init_rounds random restarts against each
    # other (never surfaced as an absolute score), so a smaller sample is a
    # fine trade: still representative enough to rank consistently, at a
    # fraction of the quadratic cost.
    # Only ever used to rank several init rounds against each other.
    sil_score = segmentation_quality(
        target_lab_reshaped,
        labels,
        sample_size=1500,
        random_state=random_seed,
    ) if rank_quality else 0.0

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
    # `new_values` only has ~cluster_layers entries (unique_clusters), but
    # np.vectorize(lambda x: new_values[x])(labels) called that Python
    # lambda once per *pixel* (profiling showed this as the single biggest
    # cost of a heightmap-init round at realistic resolutions - 0.6s+ of a
    # ~1.7s round on a 750x750 image). `labels` values are dense small
    # cluster ids, so a plain lookup-array + fancy indexing does the exact
    # same remap in one vectorized pass instead of ~550k individual Python
    # dict-lookup/function calls.
    lookup = np.empty(int(labels.max()) + 1, dtype=np.float32)
    for cluster_id, value in new_values.items():
        lookup[cluster_id] = value
    new_labels = lookup[labels]

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
            # Cluster colours are (weighted) Lab, so the filaments have to be
            # compared in the same space - matching their RGB (0-1) against
            # Lab (0-100) values picked the brightest filaments for nearly
            # every layer.
            materials = material_lab
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
        background_first_labels(labels.reshape(H, W), bg_cluster),
    )


def background_first_labels(labels: np.ndarray, bg_label: int) -> np.ndarray:
    """``labels`` with ``bg_label`` and 0 swapped. The optimizer reads label 0
    as the background (its height offset stays 0, see
    FilamentOptimizer._apply_height_offset); a raw k-means id 0 is just
    whichever cluster came first, which then never got a trainable offset."""
    bg_label = int(bg_label)
    if bg_label == 0:
        return labels
    out = labels.copy()
    out[labels == bg_label] = 0
    out[labels == 0] = bg_label
    return out


# How long an idle init worker process lives (see run_init_threads).
INIT_WORKER_IDLE_TIMEOUT_S = 1


def run_init_threads(
    target,
    max_layers,
    h,
    background_tuple,
    eps=1e-6,
    random_seed=None,
    num_threads=4,
    num_runs=1,
    init_method="kmeans",
    cluster_layers=None,
    material_colors=None,
    focus_map: Optional[np.ndarray] = None,
    focus_boost: float = 0.5,
    progress=None,
):
    background_tuple = (np.asarray(background_tuple) * 255).tolist()
    if num_runs > 1:
        # Every round currently produces the identical clustering and
        # ordering (the refinement k-means is seeded with a constant and the
        # ordering is deterministic; see auto_forge's --num_init_rounds), so
        # extra rounds only cost time - and on 4 worker processes, a CUDA
        # context each. Revisit once rounds really differ.
        print(f"num_init_rounds={num_runs}: extra rounds give identical results; running one.")
        num_runs = 1
    if random_seed is None:
        random_seed = np.random.randint(1e6)

    if cluster_layers is None:
        cluster_layers = max_layers

    lab_space = True

    pixels = _prepare_lab_image(target, (1.0, 1.0, 1.0), lab_space)
    print("Computing over‑clustering (Stage 1) …")
    centroids1, labels1 = _compute_overclustering(pixels, overcluster_k=500, random_state=random_seed)
    print(f"  → {centroids1.shape[0]} over‑cluster centroids computed.")
    if progress is not None:
        progress(0.3)

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
            target_lab=pixels,
            rank_quality=num_runs > 1,
        )

    if num_threads > 1 and num_runs > 1:
        # Only worth spinning up a worker-process pool (real, measurable
        # spawn overhead) when there's actually more than one task to
        # spread across it.
        from joblib import Parallel, delayed, parallel_config

        tasks = [delayed(_run_one)(i) for i in range(num_runs)]
        # In order as they finish, so progress can be reported (same results).
        results = []
        # The workers run their assignment on the GPU, so each holds its own
        # CUDA context (~750 MiB) for as long as it lives, and loky keeps idle
        # workers for 5 minutes. With runs this fast that covered the whole
        # optimization and pruning, and a full-resolution render ran out of
        # VRAM next to four idle workers. Let them exit as soon as the
        # rounds are done.
        with parallel_config(backend="loky", idle_worker_timeout=INIT_WORKER_IDLE_TIMEOUT_S):
            for r in Parallel(n_jobs=num_threads, verbose=10, return_as="generator")(tasks):
                results.append(r)
                if progress is not None:
                    progress(0.3 + 0.7 * len(results) / num_runs)
    else:
        results = []
        for i in range(num_runs):
            results.append(_run_one(i))
            if progress is not None:
                progress(0.3 + 0.7 * len(results) / num_runs)

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
