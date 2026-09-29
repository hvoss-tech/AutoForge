import random
from typing import Optional

import numpy as np

# NumPy/torch stand-ins for scipy / skimage / sklearn: importing those cost
# ~0.6 s per run (see _cluster); the sklearn k-means of the Christofides
# init itself stays, imported where it runs.
from autoforge.Helper.Heightmaps._cluster import cdist, rgb2lab, silhouette as _silhouette


def _compute_distinctiveness(centroids: np.ndarray) -> np.ndarray:
    """Return the minimum inter‑centroid distance for every centroid."""
    dmat = cdist(centroids, centroids, metric="euclidean")
    np.fill_diagonal(dmat, np.inf)
    return dmat.min(axis=1)


def two_stage_weighted_kmeans(
    target_lab: np.ndarray,
    H: int,
    W: int,
    overcluster_k: int = 200,
    final_k: int = 16,
    beta_distinct: float = 1.0,
    random_state: int | None = None,
):
    """Segment *target_lab* (reshaped (N,3)) into *final_k* clusters using a
    two‑stage weighted K‑Means.  Returns (final_centroids, final_labels).

    The pixel‑level data are *only* used in stage‑1; stage‑2 runs on the much
    smaller set of stage‑1 centroids which makes this fast and memory‑friendly.
    """

    from sklearn.cluster import KMeans, MiniBatchKMeans

    # Stage 1: heavy over‑segmentation so that even tiny colour modes appear.
    kmeans1 = MiniBatchKMeans(
        n_clusters=overcluster_k,
        random_state=random_state,
        max_iter=300,
    )
    labels1 = kmeans1.fit_predict(target_lab)
    centroids1 = kmeans1.cluster_centers_
    counts1 = np.bincount(labels1, minlength=overcluster_k).astype(np.float64)

    # Stage 2 weighting: size * (1 + beta * normalised distinctiveness)
    distinct = _compute_distinctiveness(centroids1)
    if distinct.max() > 0:
        distinct /= distinct.max()
    weights = counts1 * (1.0 + beta_distinct * distinct)

    # Weighted K‑Means on the centroid set.
    kmeans2 = KMeans(
        n_clusters=final_k,
        random_state=random_state,
        n_init="auto",
    )
    kmeans2.fit(centroids1, sample_weight=weights)
    centroids_final = kmeans2.cluster_centers_

    # Assign every pixel to its nearest final centroid.
    # Use chunks to keep memory bounded for very large images.
    chunk = 2**18  # about 256k pixels ≈ 768 kB of float32 per chunk
    labels_final = np.empty(target_lab.shape[0], dtype=np.int32)
    for start in range(0, target_lab.shape[0], chunk):
        end = start + chunk
        d = cdist(target_lab[start:end], centroids_final, metric="euclidean")
        labels_final[start:end] = np.argmin(d, axis=1)

    labels_final = labels_final.reshape(H, W)
    return centroids_final, labels_final


def build_distance_matrix(labs, nodes):
    """
    Given an array labs (with shape (N, dims)) and a list of node indices,
    return a distance matrix (NumPy array) of shape (len(nodes), len(nodes)).
    """
    pts = labs[nodes]  # extract only the points corresponding to nodes
    # Use cdist for fast vectorized distance computation.
    return cdist(pts, pts, metric="euclidean")


def sample_pixels_for_silhouette(labels, sample_size=5000, random_state=None):
    """
    Flatten the label map, draw at most sample_size random positions,
    and return their (index, label) pairs ready for silhouette_score.
    """
    rng = np.random.default_rng(random_state)
    flat = labels.reshape(-1)
    n = flat.shape[0]

    if n <= sample_size:
        idx = np.arange(n)
    else:
        idx = rng.choice(n, size=sample_size, replace=False)

    return idx, flat[idx]


def segmentation_quality(
    target_lab_reshaped, labels, sample_size=5000, random_state=None
):
    """
    Compute the silhouette coefficient on a random pixel subset.
    Works in Lab because `target_lab_reshaped` is already weighted Lab.
    """
    idx, lbl_subset = sample_pixels_for_silhouette(labels, sample_size, random_state)
    X_subset = target_lab_reshaped[idx]
    # In rare cases (k == 1) sklearn will raise; catch and return -1
    try:
        return _silhouette(X_subset, lbl_subset)
    except ValueError:
        return -1.0


def matrix_to_graph(matrix, nodes):
    """
    Convert a 2D NumPy array (matrix) into a dictionary-of-dicts graph,
    where graph[u][v] = matrix[i][j] for u = nodes[i], v = nodes[j].
    """
    graph = {}
    n = len(nodes)
    for i in range(n):
        u = nodes[i]
        graph[u] = {}
        for j in range(n):
            v = nodes[j]
            if u != v:
                graph[u][v] = matrix[i, j]
    return graph


# --- Christofides Helpers (same as before) ---


class UnionFind:
    def __init__(self):
        self.parents = {}
        self.weights = {}

    def __getitem__(self, obj):
        if obj not in self.parents:
            self.parents[obj] = obj
            self.weights[obj] = 1
            return obj
        path = [obj]
        root = self.parents[obj]
        while root != path[-1]:
            path.append(root)
            root = self.parents[root]
        for ancestor in path:
            self.parents[ancestor] = root
        return root

    def union(self, *objects):
        roots = [self[x] for x in objects]
        heaviest = max(((self.weights[r], r) for r in roots))[1]
        for r in roots:
            if r != heaviest:
                self.weights[heaviest] += self.weights[r]
                self.parents[r] = heaviest


def minimum_spanning_tree(G):
    tree = []
    subtrees = UnionFind()
    # Build list of edges from graph dictionary.
    edges = sorted((G[u][v], u, v) for u in G for v in G[u])
    for W, u, v in edges:
        if subtrees[u] != subtrees[v]:
            tree.append((u, v, W))
            subtrees.union(u, v)
    return tree


def find_odd_vertexes(MST):
    degree = {}
    for u, v, _ in MST:
        degree[u] = degree.get(u, 0) + 1
        degree[v] = degree.get(v, 0) + 1
    return [v for v in degree if degree[v] % 2 == 1]


def minimum_weight_matching(MST, G, odd_vert):
    odd_vertices = odd_vert.copy()
    random.shuffle(odd_vertices)
    while odd_vertices:
        v = odd_vertices.pop()
        best_u = None
        best_dist = float("inf")
        for u in odd_vertices:
            if G[v][u] < best_dist:
                best_dist = G[v][u]
                best_u = u
        MST.append((v, best_u, G[v][best_u]))
        odd_vertices.remove(best_u)


def find_eulerian_tour(MST, G):
    graph = {}
    for u, v, _ in MST:
        graph.setdefault(u, []).append(v)
        graph.setdefault(v, []).append(u)
    start = next(iter(graph))
    tour = []
    stack = [start]
    while stack:
        v = stack[-1]
        if graph[v]:
            w = graph[v].pop()
            graph[w].remove(v)
            stack.append(w)
        else:
            tour.append(stack.pop())
    return tour


def christofides_tsp(graph):
    MST = minimum_spanning_tree(graph)
    odd_vertices = find_odd_vertexes(MST)
    minimum_weight_matching(MST, graph, odd_vertices)
    eulerian_tour = find_eulerian_tour(MST, graph)
    seen = set()
    path = []
    for v in eulerian_tour:
        if v not in seen:
            seen.add(v)
            path.append(v)
    path.append(path[0])
    return path


def prune_ordering(ordering, labs, bg, fg, min_length=3, improvement_factor=1.5):
    """
    Iteratively remove clusters from the ordering if doing so significantly reduces
    the total Lab-space distance. Only clusters that produce an improvement greater
    than improvement_factor * (median gap) are removed.

    Parameters:
      ordering: list of cluster indices (the current ordering)
      labs: Lab-space coordinates (indexed by cluster index)
      bg: background anchor (never removed)
      fg: foreground anchor (never removed)
      min_length: minimum allowed length of ordering
      improvement_factor: factor multiplied by the median gap to decide if a cluster is an outlier

    Returns:
      A pruned ordering that hopefully removes only extreme outliers.
    """
    current_order = ordering.copy()
    improved = True
    # print(f"Height map pruning pass: initial number of clusters = {len(ordering)}")
    while improved:
        improved = False
        # Compute gaps between consecutive clusters.
        diffs = [
            np.linalg.norm(labs[current_order[i]] - labs[current_order[i - 1]])
            for i in range(1, len(current_order))
        ]
        if len(diffs) == 0:
            break
        median_diff = np.median(diffs)
        # Try each candidate (skip first and last if they are bg and fg)
        for i in range(1, len(current_order) - 1):
            # Optionally, preserve the fg anchor.
            if current_order[i] == fg:
                continue
            # Compute the improvement if we remove current_order[i]:
            d1 = np.linalg.norm(labs[current_order[i]] - labs[current_order[i - 1]])
            d2 = np.linalg.norm(labs[current_order[i + 1]] - labs[current_order[i]])
            direct = np.linalg.norm(
                labs[current_order[i + 1]] - labs[current_order[i - 1]]
            )
            improvement = (d1 + d2) - direct
            # Remove the cluster if the improvement is large compared to median gap.
            if improvement > improvement_factor * median_diff:
                new_order = current_order[:i] + current_order[i + 1 :]
                if len(new_order) >= min_length:
                    # print(
                    #     f"Pruning outlier: removing cluster {current_order[i]} improved local gap by {improvement:.2f}"
                    # )
                    current_order = new_order
                    improved = True
                    break  # restart the scan after removal
    # print(f"Height map pruning pass: final number of clusters = {len(current_order)}")
    return current_order


def create_mapping(final_ordering, labs, all_labels):
    """
    Creates a mapping from each cluster (from all_labels) to a value in [0,1].
    Clusters in final_ordering get evenly spaced values.
    For clusters that were pruned (i.e. not in final_ordering), assign the value
    of the nearest cluster in final_ordering (based on Lab-space distance).

    Parameters:
      final_ordering: list of cluster indices (after pruning)
      labs: array of Lab-space coordinates (indexed by cluster index)
      all_labels: sorted list of all unique clusters produced by KMeans

    Returns:
      mapping: a dict mapping each cluster label in all_labels to a float in [0,1].
    """
    mapping = {}
    n_order = len(final_ordering)
    # If there's only one cluster in final_ordering, assign 0.5
    if n_order == 1:
        for label in all_labels:
            mapping[label] = 0.5
        return mapping

    # Assign evenly spaced values for clusters in final_ordering.
    for i, cluster in enumerate(final_ordering):
        mapping[cluster] = i / (n_order - 1)

    # For clusters not in final_ordering, find the nearest cluster (in Lab space)
    # from final_ordering and use its mapping value.
    for label in all_labels:
        if label not in mapping:
            lab_val = labs[label]
            best_cluster = None
            best_dist = float("inf")
            for cl in final_ordering:
                d = np.linalg.norm(labs[cl] - lab_val)
                if d < best_dist:
                    best_dist = d
                    best_cluster = cl
            mapping[label] = mapping[best_cluster]
    return mapping


def tsp_order_christofides_path(nodes, labs, bg, fg):
    """
    Ensure that the background and foreground nodes are always in the TSP cycle.
    nodes: list of cluster indices (ideally including bg and fg)
    labs: Lab-space coordinates (indexed by cluster index)
    bg, fg: background and foreground cluster indices
    Returns an ordering (list of cluster indices) that contains all nodes,
    starts with bg and ends with fg.
    """
    # Guarantee that bg and fg are included in nodes.
    nodes = list(set(nodes) | {bg, fg})

    artificial = -1
    LARGE = 1e6
    n = len(nodes)

    # Precompute the distance matrix for the given nodes.
    D = build_distance_matrix(labs, nodes)

    # Build an augmented (n+1)x(n+1) matrix.
    aug_mat = np.zeros((n + 1, n + 1))
    aug_mat[:n, :n] = D

    # For the artificial node, set cost = 0 for bg/fg and LARGE for others.
    for i, u in enumerate(nodes):
        if u in {bg, fg}:
            aug_mat[i, n] = 0.0
            aug_mat[n, i] = 0.0
        else:
            aug_mat[i, n] = LARGE
            aug_mat[n, i] = LARGE
    aug_mat[n, n] = 0.0

    # Create augmented nodes list.
    aug_nodes = nodes + [artificial]
    graph = matrix_to_graph(aug_mat, aug_nodes)

    # Run Christofides algorithm on the augmented graph.
    cycle = christofides_tsp(graph)
    # Remove the artificial node if present.
    cycle = [node for node in cycle if node != artificial]

    # Ensure bg and fg are in the cycle.
    if bg not in cycle:
        cycle.insert(0, bg)
    if fg not in cycle:
        cycle.append(fg)

    # Rotate the cycle so that bg is first.
    if cycle[0] != bg:
        idx = cycle.index(bg)
        cycle = cycle[idx:] + cycle[:idx]

    # Force fg to be the last element.
    if cycle[-1] != fg:
        cycle.remove(fg)
        cycle.append(fg)

    # Optionally, check if reversing the internal order improves the ordering metric.
    if len(cycle) > 2:
        reversed_cycle = [cycle[0]] + cycle[1:-1][::-1] + [cycle[-1]]
        if compute_ordering_metric(reversed_cycle, labs) < compute_ordering_metric(
            cycle, labs
        ):
            cycle = reversed_cycle

    return cycle


# --- Optimized Ordering Metric (Vectorized) ---


def compute_ordering_metric(ordering, labs):
    """
    Computes a metric for the ordering as the total Lab-space distance between consecutive clusters.
    Uses vectorized operations for speed.
    """
    pts = labs[ordering]
    # Compute differences between consecutive rows.
    diffs = np.diff(pts, axis=0)
    # Compute Euclidean norms along rows and sum.
    return np.sum(np.linalg.norm(diffs, axis=1))


# --- Revised init_height_map with Optimizations ---
def interpolate_arrays(value_array_pairs, num_points):
    # Sort pairs by the value (first element in each pair)
    value_array_pairs.sort(key=lambda x: x[0])

    values = np.array([pair[0] for pair in value_array_pairs])
    arrays = np.array([pair[1] for pair in value_array_pairs])

    # Generate the new interpolation positions
    new_values = np.linspace(values[0], values[-1], num_points)

    # Interpolate each element in the arrays
    interpolated_arrays = []
    for i in range(arrays.shape[1]):  # Assuming arrays are 2D or more
        interpolated_column = np.interp(new_values, values, arrays[:, i])
        interpolated_arrays.append(interpolated_column)

    return np.stack(interpolated_arrays, axis=1)  # Reconstruct the interpolated array


def init_height_map(
    target,
    max_layers,
    h,  # unused here but preserved for API compatibility
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
):
    """
    init_method should be one of quantize_median,quantize_maxcoverage,quantize_fastoctree,kmeans

    Initializes pixel height logits by:
      1. Clustering the image into max_layers clusters (via KMeans).
      2. Converting cluster centers to Lab space.
      3. Determining two anchor clusters:
         - The background cluster (closest to background_tuple) as the bottom.
         - The foreground cluster (farthest from the background) as the top.
      4. Using a Christofides TSP solution (with an artificial node) to order the clusters.
      5. Mapping the clusters to evenly spaced height values in [0, 1] and converting to logits.

    Added parameters:
      focus_map (H,W) in [0,1]: if provided boosts initial height values inside priority regions.
      focus_boost: scaling factor; pixel value multiplied by (1 + focus_boost * focus_map).
    """
    import random

    if cluster_layers is None:
        cluster_layers = max_layers  # // 2

    if random_seed is not None:
        np.random.seed(random_seed)
        random.seed(random_seed)

    H, W, _ = target.shape

    target_np = target.astype(np.float32) / 255.0
    if lab_space:
        target_lab_full = rgb2lab(target_np)
        target_lab_full[..., 0] *= lab_weights[0]
        target_lab_full[..., 1] *= lab_weights[1]
        target_lab_full[..., 2] *= lab_weights[2]
    else:
        target_lab_full = target_np
    target_lab_reshaped = target_lab_full.reshape(-1, 3)
    labs, labels = two_stage_weighted_kmeans(
        target_lab_reshaped,
        H,
        W,
        overcluster_k=500,
        final_k=cluster_layers,
        beta_distinct=4.0,
        random_state=random_seed,
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

    # Convert the background color to Lab and apply the same weighting.
    bg_rgb = np.array(background_tuple).astype(np.float32) / 255.0
    if lab_space:
        bg_lab = rgb2lab(np.array([[bg_rgb]]))[0, 0, :]
        bg_lab[0] *= lab_weights[0]
        bg_lab[1] *= lab_weights[1]
        bg_lab[2] *= lab_weights[2]
    else:
        bg_lab = bg_rgb

    # Identify the cluster closest to the background and the farthest.
    distances = np.linalg.norm(labs - bg_lab, axis=1)
    bg_cluster = int(np.argmin(distances))
    fg_cluster = int(np.argmax(distances))

    # Get the unique clusters (should be 0...max_layers-1 ideally).
    unique_clusters = sorted(np.unique(labels))
    nodes = unique_clusters

    # Get the ordering via TSP ordering function.
    final_ordering = tsp_order_christofides_path(nodes, labs, bg_cluster, fg_cluster)

    # # Optionally prune out outliers.
    # final_ordering = prune_ordering(
    #     final_ordering, labs, bg_cluster, fg_cluster, min_length=3, improvement_factor=3
    # )

    # Create a mapping that covers all clusters.
    new_values = create_mapping(final_ordering, labs, unique_clusters)
    new_labels = np.vectorize(lambda x: new_values[x])(labels).astype(np.float32)

    # Apply focus map boost before converting to logits if provided.
    if focus_map is not None:
        fm = np.asarray(focus_map, dtype=np.float32)
        # Ensure fm is in [0,1]
        if fm.max() > 1.0 or fm.min() < 0.0:
            # If passed in 0-255, normalize
            fm = np.clip(fm, 0, 255) / 255.0
        if fm.shape != (H, W):
            # Nearest-neighbor resize using index mapping (no extra deps)
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
        # Convert material_colors (assumed in [-1, 3]) to normalized [0, 1] for Lab conversion.
        # Adjust as needed depending on your color convention.
        # Reshape to (1, num_materials, 3) so that rgb2lab returns shape (1, num_materials, 3)
        if lab_space:
            material_lab = rgb2lab(material_colors.reshape(1, -1, 3)).reshape(-1, 3)
            # Apply the same lab_weights.
            material_lab[:, 0] *= lab_weights[0]
            material_lab[:, 1] *= lab_weights[1]
            material_lab[:, 2] *= lab_weights[2]
            materials = material_colors
        else:
            materials = material_colors

        num_materials = materials.shape[0]

        # Initialize global logits for each cluster in unique_clusters.
        global_logits = []

        for idx, label in enumerate(unique_clusters):
            # Use the final mapping value for this cluster.
            t = new_values[label]
            # Compute distances from this cluster’s lab value to each material (in Lab space).
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
    h,  # unused but preserved for API compatibility
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
    lab_space = True

    if num_threads > 1:
        from joblib import Parallel, delayed

        tasks = [
            delayed(init_height_map)(
                target,
                max_layers,
                h,
                background_tuple,
                eps,
                random_seed + i,
                init_method=init_method,
                cluster_layers=cluster_layers,
                lab_space=lab_space,
                material_colors=material_colors,
                focus_map=focus_map,
                focus_boost=focus_boost,
            )
            for i in range(num_runs)
        ]

        # Execute tasks in parallel; adjust n_jobs to match your available cores
        results = Parallel(n_jobs=num_threads, verbose=10)(tasks)

    else:
        results = [
            init_height_map(
                target,
                max_layers,
                h,
                background_tuple,
                eps,
                random_seed + i,
                init_method=init_method,
                cluster_layers=cluster_layers,
                lab_space=lab_space,
                material_colors=material_colors,
                focus_map=focus_map,
                focus_boost=focus_boost,
            ) for i in range(num_runs)
        ]

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
