import random
from typing import Optional

import numpy as np
from PIL import Image
from sklearn.cluster import KMeans
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import silhouette_score
from sklearn.utils._testing import ignore_warnings

from autoforge.Helper.DeviceUtils import accelerator_device


def initialize_pixel_height_logits(target):
    """
    Initialize pixel height logits based on the luminance of the target image.

    Assumes target is a jnp.array of shape (H, W, 3) in the range [0, 255].
    Uses the formula: L = 0.299*R + 0.587*G + 0.114*B.

    Args:
        target (np.ndarray): The target image array with shape (H, W, 3).

    Returns:
        np.ndarray: The initialized pixel height logits.
    """
    # Compute normalized luminance in [0,1]
    normalized_lum = (
        0.299 * target[..., 0] + 0.587 * target[..., 1] + 0.114 * target[..., 2]
    ) / 255.0
    # To avoid log(0) issues, add a small epsilon.
    eps = 1e-6
    # Convert normalized luminance to logits using the inverse sigmoid (logit) function.
    # This ensures that jax.nn.sigmoid(pixel_height_logits) approximates normalized_lum.
    pixel_height_logits = np.log((normalized_lum + eps) / (1 - normalized_lum + eps))
    return pixel_height_logits


@ignore_warnings(category=ConvergenceWarning)
def init_height_map_depth_color_adjusted(
    target,
    max_layers,
    eps=1e-6,
    random_seed=None,
    depth_strength=0.25,
    depth_threshold=0.2,
    min_cluster_value=0.1,
    w_depth=0.5,
    w_lum=0.5,
    order_blend=0.1,
    focus_map: Optional[np.ndarray] = None,
    focus_boost: float = 0.5,
):
    """
    Initialize pixel height logits by combining depth and color information while allowing a blend
    between the original luminance-based ordering and a depth-informed ordering.

    Steps:
      1. Obtain a normalized depth map using Depth Anything v2.
      2. Determine the optimal number of color clusters (between 2 and max_layers) via silhouette score.
      3. Cluster the image colors and (if needed) split clusters with large depth spreads.
      4. For each final cluster, compute its average depth and average luminance.
      5. Compute two orderings:
            - ordering_orig: Sorted purely by average luminance (approximating the original code).
            - ordering_depth: A TSP-inspired ordering using a weighted distance based on depth and luminance.
      6. For each cluster, blend its rank (normalized position) between the two orderings using order_blend.
      7. Based on the blended ordering, assign an even spacing value from min_cluster_value to 1.
      8. Finally, blend the even spacing with the cluster's average depth using depth_strength and
         convert the result to logits via an inverse sigmoid transform.

    Args:
        target (np.ndarray): Input image of shape (H, W, 3) in [0, 255].
        max_layers (int): Maximum number of clusters to consider.
        eps (float): Small constant to avoid division by zero.
        random_seed (int): Random seed for reproducibility.
        depth_strength (float): Weight (0 to 1) for blending even spacing with the cluster's average depth.
        depth_threshold (float): If a cluster’s depth spread exceeds this value, it is split.
        min_cluster_value (float): Minimum normalized value for the lowest cluster.
        w_depth (float): Weight for depth difference in ordering_depth.
        w_lum (float): Weight for luminance difference in ordering_depth.
        order_blend (float): Slider (0 to 1) blending original luminance ordering (0) and depth-informed ordering (1).
        focus_map (np.ndarray | None): Optional focus map to boost height values in certain regions.
        focus_boost (float): Scaling factor for the focus map.

    Returns:
        tuple[np.ndarray, np.ndarray]: Pixel height logits (H, W) and the final integer label map (H, W).
    """

    # ---------------------------
    # Step 1: Obtain normalized depth map using Depth Anything v2
    # ---------------------------
    # Local import to avoid making transformers a hard dependency unless this init is used.
    try:
        from transformers import pipeline  # type: ignore
    except Exception as e:
        raise ImportError(
            "Depth initializer requires 'transformers' installed. Install transformers to use --init_heightmap_method depth."
        ) from e

    target_uint8 = target.astype(np.uint8)
    image_pil = Image.fromarray(target_uint8)
    # Run the depth model on whatever accelerator is present. transformers
    # defaults to CPU, which left this init several times slower than it needs
    # to be on every GPU, Apple Silicon included. A backend that chokes on the
    # model falls back to CPU rather than failing the run.
    accel = accelerator_device()
    try:
        pipe = pipeline(
            task="depth-estimation",
            model="depth-anything/Depth-Anything-V2-Small-hf",
            device=accel,
        )
        depth_result = pipe(image_pil)
    except Exception:
        if accel is None:
            raise
        pipe = pipeline(
            task="depth-estimation", model="depth-anything/Depth-Anything-V2-Small-hf"
        )
        depth_result = pipe(image_pil)
    depth_map = depth_result["depth"]
    if hasattr(depth_map, "convert"):
        depth_map = np.array(depth_map)
    depth_map = depth_map.astype(np.float32)
    # Ensure depth map matches input size
    if depth_map.shape[:2] != (target.shape[0], target.shape[1]):
        from PIL import Image as _Image

        Resampling = getattr(_Image, "Resampling", None)
        resample = Resampling.BILINEAR if Resampling is not None else _Image.BILINEAR
        depth_map = np.array(
            _Image.fromarray(depth_map).resize(
                (target.shape[1], target.shape[0]), resample
            )
        )
    depth_min = depth_map.min()
    depth_max = depth_map.max()
    depth_norm = (depth_map - depth_min) / (depth_max - depth_min + eps)

    # ---------------------------
    # Step 2: Find optimal number of clusters (n in [2, max_layers]) for color clustering
    # ---------------------------
    H, W, _ = target.shape
    pixels = target.reshape(-1, 3).astype(np.float32)

    # sklearn refuses more clusters than samples (a tiny or 1-pixel image).
    optimal_n = max(1, min(max_layers, pixels.shape[0]))
    # ---------------------------
    # Step 3: Perform color clustering on the full image
    # ---------------------------
    # kmeans = KMeans(n_clusters=optimal_n, random_state=random_seed).fit(pixels)
    # labels = kmeans.labels_.reshape(H, W)
    labels = KMeans(n_clusters=optimal_n, random_state=random_seed).fit_predict(pixels)
    labels = labels.reshape(H, W)

    # ---------------------------
    # Step 4: Adjust clusters based on depth (split clusters with high depth spread)
    # ---------------------------
    final_labels = np.copy(labels)
    new_cluster_id = 0
    cluster_info = {}  # Mapping: final_cluster_id -> avg_depth
    unique_labels = np.unique(labels)
    for orig_label in unique_labels:
        mask = labels == orig_label
        cluster_depths = depth_norm[mask]
        avg_depth = np.mean(cluster_depths)
        depth_range = cluster_depths.max() - cluster_depths.min()
        if depth_range > depth_threshold:
            # Split this cluster into 2 subclusters based on depth values.
            depth_values = cluster_depths.reshape(-1, 1)
            k_split = 2
            kmeans_split = KMeans(n_clusters=k_split, random_state=random_seed)
            split_labels = kmeans_split.fit_predict(depth_values)
            indices = np.argwhere(mask)
            for split in range(k_split):
                sub_mask = split_labels == split
                inds = indices[sub_mask]
                if inds.size == 0:
                    continue
                for i, j in inds:
                    final_labels[i, j] = new_cluster_id
                sub_avg_depth = np.mean(depth_norm[mask][split_labels == split])
                cluster_info[new_cluster_id] = sub_avg_depth
                new_cluster_id += 1
        else:
            indices = np.argwhere(mask)
            for i, j in indices:
                final_labels[i, j] = new_cluster_id
            cluster_info[new_cluster_id] = avg_depth
            new_cluster_id += 1

    num_final_clusters = new_cluster_id

    # ---------------------------
    # Step 5: Compute average luminance for each final cluster (using standard weights)
    # ---------------------------
    cluster_colors = {}
    for cid in range(num_final_clusters):
        mask = final_labels == cid
        if np.sum(mask) == 0:
            continue
        avg_color = np.mean(
            target.reshape(-1, 3)[final_labels.reshape(-1) == cid], axis=0
        )
        lum = (
            0.299 * avg_color[0] + 0.587 * avg_color[1] + 0.114 * avg_color[2]
        ) / 255.0
        cluster_colors[cid] = lum

    # ---------------------------
    # Step 6: Build cluster feature list: (cid, avg_depth, avg_luminance)
    # ---------------------------
    cluster_features = []
    for cid in range(num_final_clusters):
        avg_depth = cluster_info[cid]
        avg_lum = cluster_colors.get(cid, 0.5)
        cluster_features.append((cid, avg_depth, avg_lum))

    # ---------------------------
    # Step 7: Compute depth-informed ordering (TSP-inspired, using w_depth and w_lum)
    # ---------------------------
    def distance(feat1, feat2):
        return w_depth * abs(feat1[1] - feat2[1]) + w_lum * abs(feat1[2] - feat2[2])

    # Greedy nearest-neighbor ordering starting from cluster with lowest avg_depth
    start_idx = np.argmin([feat[1] for feat in cluster_features])
    unvisited = cluster_features.copy()
    ordering_depth = []
    current = unvisited.pop(start_idx)
    ordering_depth.append(current)
    while unvisited:
        next_idx = np.argmin([distance(current, candidate) for candidate in unvisited])
        current = unvisited.pop(next_idx)
        ordering_depth.append(current)

    # 2-opt refinement for ordering_depth
    def total_distance(ordering):
        return sum(
            distance(ordering[i], ordering[i + 1]) for i in range(len(ordering) - 1)
        )

    improved = True
    best_order_depth = ordering_depth
    best_dist = total_distance(ordering_depth)
    while improved:
        improved = False
        for i in range(1, len(best_order_depth) - 1):
            for j in range(i + 1, len(best_order_depth)):
                new_order = (
                    best_order_depth[:i]
                    + best_order_depth[i : j + 1][::-1]
                    + best_order_depth[j + 1 :]
                )
                new_dist = total_distance(new_order)
                if new_dist < best_dist:
                    best_order_depth = new_order
                    best_dist = new_dist
                    improved = True
        ordering_depth = best_order_depth

    # ---------------------------
    # Step 8: Compute original (luminance-based) ordering: simply sort by avg_lum (darkest first)
    # ---------------------------
    ordering_orig = sorted(cluster_features, key=lambda x: x[2])

    # ---------------------------
    # Step 9: Blend the two orderings via their rank positions using order_blend
    # ---------------------------
    # Map each cluster id to its rank in each ordering.
    rank_orig = {feat[0]: idx for idx, feat in enumerate(ordering_orig)}
    rank_depth = {feat[0]: idx for idx, feat in enumerate(ordering_depth)}
    # Normalize ranks to [0, 1]
    norm_rank_orig = {
        cid: rank_orig[cid] / (len(ordering_orig) - 1) if len(ordering_orig) > 1 else 0
        for cid in rank_orig
    }
    norm_rank_depth = {
        cid: rank_depth[cid] / (len(ordering_depth) - 1)
        if len(ordering_depth) > 1
        else 0
        for cid in rank_depth
    }

    # Compute blended rank for each cluster
    blended_ranks = {}
    for cid in norm_rank_orig:
        blended_ranks[cid] = (1 - order_blend) * norm_rank_orig[
            cid
        ] + order_blend * norm_rank_depth[cid]

    # Final ordering: sort clusters by blended rank (ascending)
    final_order = sorted(cluster_features, key=lambda x: blended_ranks[x[0]])

    # ---------------------------
    # Step 10: Assign new normalized values to clusters
    # Even spacing from min_cluster_value to 1 based on final ordering
    even_spacing = np.linspace(min_cluster_value, 1, num_final_clusters)
    final_mapping = {}
    for rank, (cid, avg_depth, avg_lum) in enumerate(final_order):
        # Blend even spacing with the cluster's average depth using depth_strength.
        # (When depth_strength=0, purely even spacing; when 1, purely avg_depth.)
        blended_value = (1 - depth_strength) * even_spacing[
            rank
        ] + depth_strength * avg_depth
        blended_value = np.clip(blended_value, min_cluster_value, 1)
        final_mapping[cid] = blended_value

    # ---------------------------
    # Step 11: Create new normalized label image and convert to logits.
    # ---------------------------
    new_labels = np.vectorize(lambda x: final_mapping[x])(final_labels).astype(
        np.float32
    )
    if new_labels.max() > 0:
        new_labels = new_labels / new_labels.max()
    if focus_map is not None:
        fm = np.asarray(focus_map, dtype=np.float32)
        if fm.max() > 1.0 or fm.min() < 0.0:
            fm = np.clip(fm, 0, 255) / 255.0
        if fm.shape != new_labels.shape:
            H, W = new_labels.shape
            src_h, src_w = fm.shape[:2]
            iy = (np.arange(H) * src_h / H).astype(np.int32)
            ix = (np.arange(W) * src_w / W).astype(np.int32)
            iy = np.clip(iy, 0, src_h - 1)
            ix = np.clip(ix, 0, src_w - 1)
            fm = fm[np.ix_(iy, ix)]
        new_labels = np.clip(new_labels * (1.0 + focus_boost * fm), 0.0, 1.0)
    pixel_height_logits = np.log((new_labels + eps) / (1 - new_labels + eps))
    return pixel_height_logits, final_labels.astype(np.int32)


def tsp_simulated_annealing(
    band_reps,
    start_band,
    end_band,
    initial_order=None,
    initial_temp=1000,
    cooling_rate=0.995,
    num_iter=10000,
):
    """
    Solve the band ordering problem using simulated annealing.

    Args:
        band_reps (list or np.ndarray): List of Lab color representations.
        start_band (int): Index for the darkest band.
        end_band (int): Index for the brightest band.
        initial_order (list, optional): Initial ordering of band indices.
        initial_temp (float): Starting temperature.
        cooling_rate (float): Factor to cool the temperature.
        num_iter (int): Maximum number of iterations.

    Returns:
        list: An ordering of band indices from start_band to end_band.
    """
    if initial_order is None:
        # Use a simple ordering: start, middle bands as given, then end.
        middle_indices = [
            i for i in range(len(band_reps)) if i not in (start_band, end_band)
        ]
        order = [start_band] + middle_indices + [end_band]
    else:
        order = initial_order.copy()

    def total_distance(order):
        return sum(
            np.linalg.norm(band_reps[order[i]] - band_reps[order[i + 1]])
            for i in range(len(order) - 1)
        )

    current_distance = total_distance(order)
    best_order = order.copy()
    best_distance = current_distance
    temp = initial_temp

    for _ in range(num_iter):
        # Randomly swap two indices in the middle of the order
        new_order = order.copy()
        idx1, idx2 = random.sample(range(1, len(order) - 1), 2)
        new_order[idx1], new_order[idx2] = new_order[idx2], new_order[idx1]

        new_distance = total_distance(new_order)
        delta = new_distance - current_distance

        # Accept the new order if it improves or with a probability to escape local minima
        if delta < 0 or np.exp(-delta / temp) > random.random():
            order = new_order.copy()
            current_distance = new_distance
            if current_distance < best_distance:
                best_order = order.copy()
                best_distance = current_distance

        temp *= cooling_rate
        if temp < 1e-6:
            break
    return best_order


def choose_optimal_num_bands(centroids, min_bands=2, max_bands=15, random_seed=None):
    """
    Determine the optimal number of clusters (bands) for the centroids
    by maximizing the silhouette score.

    Args:
        centroids (np.ndarray): Array of centroid colors (e.g., shape (n_clusters, 3)).
        min_bands (int): Minimum number of clusters to try.
        max_bands (int): Maximum number of clusters to try.
        random_seed (int, optional): Random seed for reproducibility.

    Returns:
        int: Optimal number of bands.
    """
    best_num = min_bands
    best_score = -1

    for num in range(min_bands, max_bands + 1):
        # kmeans = KMeans(n_clusters=num, random_state=random_seed).fit(centroids)
        # labels = kmeans.labels_
        labels = KMeans(n_clusters=num, random_state=random_seed).fit_predict(centroids)
        # If there's only one unique label, skip to avoid errors.
        if len(np.unique(labels)) < 2:
            continue
        score = silhouette_score(centroids, labels)
        if score > best_score:
            best_score = score
            best_num = num

    return best_num
