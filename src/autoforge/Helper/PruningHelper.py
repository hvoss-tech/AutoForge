import math
import random

import torch
from tqdm import tqdm
from joblib import Parallel, delayed
import threading
import numpy as np

from autoforge.Helper.OptimizerHelper import composite_image_disc
from autoforge.Loss.LossFunctions import compute_loss
from autoforge.Modules.Optimizer import FilamentOptimizer
from collections import deque

# One global lock that serialises every call that needs GPU / VRAM
_gpu_lock = threading.Lock()


def disc_to_logits(
    dg: torch.Tensor, num_materials: int, big_pos: float = 1e5
) -> torch.Tensor:
    """
    Convert a discrete [max_layers] material assignment into [max_layers, num_materials] "logits"
    suitable for compositing with mode='discrete'.

    We place a very large positive logit on the chosen material index, and large negative on others.
    That ensures Gumbel softmax with hard=True picks that color with probability ~1.
    """
    max_layers = dg.size(0)
    # Start with a big negative for all materials:
    logits = dg.new_full(
        (max_layers, num_materials), fill_value=-big_pos, dtype=torch.float32
    )
    # Put a large positive at each chosen ID:
    for layer_idx in range(max_layers):
        c_id = dg[layer_idx].item()
        logits[layer_idx, c_id] = big_pos
    return logits


def _chunked(iterable, chunk_size):
    """Yield successive *chunk_size* chunks from *iterable*."""
    for i in range(0, len(iterable), chunk_size):
        yield iterable[i : i + chunk_size]


def prune_num_colors(
    optimizer: FilamentOptimizer,
    max_colors_allowed: int,
    tau_for_comp: float,
    perception_loss_module: torch.nn.Module,
    n_jobs: int | None = -1,
    *,
    fast: bool = True,
    chunking_percent=0.05,
    allowed_loss_increase_percent=0.00,
    preview_callback=None,
) -> torch.Tensor:
    num_materials = optimizer.material_colors.shape[0]
    disc_global, _ = optimizer.get_discretized_solution(best=True)

    def score_color(
        dg_base: torch.Tensor, c_from: int, c_to: int
    ) -> tuple[float, torch.Tensor]:
        dg_test = merge_color(dg_base, c_from, c_to)
        logits_for_disc = disc_to_logits(dg_test, num_materials, big_pos=1e5)

        with _gpu_lock, torch.no_grad():
            out_im = optimizer.get_best_discretized_image(
                custom_global_logits=logits_for_disc
            )
            loss = compute_loss(comp=out_im, target=optimizer.target, focus_map=optimizer.focus_map)

        return loss, dg_test

    def get_image_loss(dg_test: torch.Tensor) -> float:
        logits_for_disc = disc_to_logits(dg_test, num_materials, big_pos=1e5)

        with _gpu_lock, torch.no_grad():
            out_im = optimizer.get_best_discretized_image(
                custom_global_logits=logits_for_disc
            )
            loss = compute_loss(comp=out_im, target=optimizer.target, focus_map=optimizer.focus_map)

        return loss

    best_dg = disc_global.clone()
    best_loss = get_image_loss(best_dg)

    distinct_mats = torch.unique(best_dg)

    print(
        f"PRUNING: Color - initial loss={best_loss:.4f}, initial colors={len(distinct_mats)}"
    )

    tbar = tqdm(total=100, leave=False)

    while True:
        distinct_mats = torch.unique(best_dg)

        if len(distinct_mats) <= 1:
            break

        merge_pairs = [
            (c_from.item(), c_to.item())
            for c_from in distinct_mats
            for c_to in distinct_mats
            if c_from != c_to
        ]

        tbar.set_description(
            f"Colors {len(distinct_mats)} | Loss {best_loss:.4f} | Merge pairs {len(merge_pairs)}"
        )
        tbar.update(1)

        if preview_callback is not None:
            try:
                preview_callback(
                    optimizer,
                    int((1 - len(distinct_mats) / max(max_colors_allowed, 1)) * 100),
                )
            except Exception:
                pass

        if not merge_pairs:
            break

        if fast:
            random.shuffle(merge_pairs)
            chunk_size = max(1, math.ceil(len(merge_pairs) * chunking_percent))

            improved = False
            best_candidate = None
            best_candidate_loss = float("inf")

            for chunk in _chunked(merge_pairs, chunk_size):
                cand_results = Parallel(
                    n_jobs=n_jobs, backend="threading", prefer="threads"
                )(delayed(score_color)(best_dg, *pair) for pair in chunk)

                merge_loss, merge_dg = min(cand_results, key=lambda x: x[0])

                if merge_loss < best_loss * (1 + allowed_loss_increase_percent):
                    best_dg = merge_dg
                    best_loss = merge_loss
                    improved = True
                    break

                if merge_loss < best_candidate_loss:
                    best_candidate = merge_dg
                    best_candidate_loss = merge_loss

            if not improved:
                if (
                    len(distinct_mats) > max_colors_allowed
                    and best_candidate is not None
                ):
                    best_dg = best_candidate
                    best_loss = best_candidate_loss
                else:
                    break

        else:
            cand_results = Parallel(
                n_jobs=n_jobs, backend="threading", prefer="threads"
            )(delayed(score_color)(best_dg, *pair) for pair in merge_pairs)

            merge_loss, merge_dg = min(cand_results, key=lambda x: x[0])

            if merge_loss < best_loss or len(distinct_mats) > max_colors_allowed:
                best_dg = merge_dg
                best_loss = merge_loss
            else:
                break

    tbar.close()

    optimizer.best_params["global_logits"] = disc_to_logits(
        best_dg, num_materials=num_materials, big_pos=1e5
    )

    final_colors = torch.unique(best_dg)

    assert len(final_colors) <= max_colors_allowed, (
        f"Color pruning failed: {len(final_colors)} > {max_colors_allowed}"
    )

    print(
        f"PRUNING: Color - final loss={best_loss:.4f}, final colors={len(final_colors)}"
    )

    return best_dg


def prune_num_swaps(
    optimizer: FilamentOptimizer,
    max_swaps_allowed: int,
    tau_for_comp: float,
    perception_loss_module: torch.nn.Module,
    n_jobs: int | None = -1,
    *,
    fast: bool = True,
    chunking_percent=0.05,
    allowed_loss_increase_percent=0.000,
    preview_callback=None,
) -> torch.Tensor:
    """Reduce the number of color boundaries until it is <= max_swaps_allowed.
    If necessary, merges are forced even when they worsen the loss.
    """

    num_materials = optimizer.material_colors.shape[0]
    disc_global, _ = optimizer.get_discretized_solution(best=True)

    def get_image_loss(dg_test: torch.Tensor) -> float:
        logits_for_disc = disc_to_logits(dg_test, num_materials, big_pos=1e5)
        with _gpu_lock, torch.no_grad():
            out_im = optimizer.get_best_discretized_image(
                custom_global_logits=logits_for_disc
            )
            loss = compute_loss(comp=out_im, target=optimizer.target, focus_map=optimizer.focus_map)
        return loss

    def score_swap(dg_base: torch.Tensor, band_a, band_b, direction: str):
        dg_test = merge_bands(dg_base, band_a, band_b, direction=direction)
        logits_for_disc = disc_to_logits(dg_test, num_materials, big_pos=1e5)
        with _gpu_lock, torch.no_grad():
            out_im = optimizer.get_best_discretized_image(
                custom_global_logits=logits_for_disc
            )
            loss = compute_loss(comp=out_im, target=optimizer.target, focus_map=optimizer.focus_map)
        return loss, dg_test

    best_dg = disc_global.clone()
    best_loss = get_image_loss(best_dg)

    bands = find_color_bands(best_dg)
    num_swaps = len(bands) - 1

    print(f"PRUNING: Swap - initial loss={best_loss:.4f}, initial swaps={num_swaps:d}")

    tbar = tqdm(total=100, leave=False)

    while True:
        bands = find_color_bands(best_dg)
        num_swaps = len(bands) - 1

        if num_swaps == 0:
            break

        merge_specs = [
            (bands[i], bands[i + 1], dirn)
            for i in range(num_swaps)
            for dirn in ("forward", "backward")
            if bands[i][2] != bands[i + 1][2]
        ]

        tbar.set_description(
            f"Swaps {num_swaps} | Loss {best_loss:.4f} | Merge swaps {len(merge_specs)}"
        )
        tbar.update(1)

        if preview_callback is not None:
            try:
                preview_callback(
                    optimizer, int((1 - num_swaps / max(max_swaps_allowed, 1)) * 100)
                )
            except Exception:
                pass

        if not merge_specs:
            break

        if fast:
            random.shuffle(merge_specs)
            chunk_size = max(1, math.ceil(len(merge_specs) * chunking_percent))

            improved = False
            best_candidate = None
            best_candidate_loss = float("inf")

            for chunk in _chunked(merge_specs, chunk_size):
                cand_results = Parallel(
                    n_jobs=n_jobs, backend="threading", prefer="threads"
                )(
                    delayed(score_swap)(best_dg, band_a, band_b, direction)
                    for band_a, band_b, direction in chunk
                )

                merge_loss, merge_dg = min(cand_results, key=lambda x: x[0])

                # Accept merge if loss increase is allowed
                if merge_loss < best_loss * (1 + allowed_loss_increase_percent):
                    best_dg, best_loss = merge_dg, merge_loss
                    improved = True
                    break

                # Track best candidate for forced merge
                if merge_loss < best_candidate_loss:
                    best_candidate = (merge_dg, merge_loss)
                    best_candidate_loss = merge_loss

            bands = find_color_bands(best_dg)
            num_swaps = len(bands) - 1

            # If no acceptable merge but we still exceed swap budget,
            # force the best candidate.
            if not improved:
                if num_swaps > max_swaps_allowed and best_candidate is not None:
                    best_dg, best_loss = best_candidate
                    improved = True
                else:
                    break

        else:
            cand_results = Parallel(
                n_jobs=n_jobs, backend="threading", prefer="threads"
            )(
                delayed(score_swap)(best_dg, band_a, band_b, direction)
                for band_a, band_b, direction in merge_specs
            )

            merge_loss, merge_dg = min(cand_results, key=lambda x: x[0])

            if merge_loss < best_loss or num_swaps > max_swaps_allowed:
                best_dg, best_loss = merge_dg, merge_loss
            else:
                break

    print(
        f"PRUNING: Swap - final loss={best_loss:.4f}, final swaps={len(find_color_bands(best_dg)) - 1:d}"
    )

    tbar.close()

    optimizer.best_params["global_logits"] = disc_to_logits(
        best_dg, num_materials=num_materials, big_pos=1e5
    )

    # safety check
    final_swaps = len(find_color_bands(best_dg)) - 1
    assert final_swaps <= max_swaps_allowed, (
        f"Swap pruning failed: {final_swaps} swaps > {max_swaps_allowed}"
    )

    return best_dg


def merge_color(dg: torch.Tensor, c_from: int, c_to: int) -> torch.Tensor:
    """
    Return a copy of dg where every layer with material c_from is replaced by c_to.
    """
    dg_new = dg.clone()
    dg_new[dg_new == c_from] = c_to
    return dg_new


def find_color_bands(dg: torch.Tensor):
    """
    Return a list of (start_idx, end_idx, color_id) for each contiguous band
    in 'dg'. Example: if dg = [0,0,1,1,1,2,2], we get:
       [(0,1,0), (2,4,1), (5,6,2)]
    """
    bands = []
    dg_cpu = dg.detach().cpu().numpy()
    start_idx = 0
    current_color = dg_cpu[0]
    n = len(dg_cpu)

    for i in range(1, n):
        if dg_cpu[i] != current_color:
            bands.append((start_idx, i - 1, current_color))
            start_idx = i
            current_color = dg_cpu[i]
    # finish last band
    bands.append((start_idx, n - 1, current_color))

    return bands


def merge_bands(
    dg: torch.Tensor, band_a: (int, int, int), band_b: (int, int, int), direction: str
):
    """
    Merge band_a and band_b. If direction=="forward", unify band_b's color to band_a's color;
    otherwise unify band_a's color to band_b's color.
    band_a, band_b = (start_idx, end_idx, color_id)
    """
    dg_new = dg.clone()
    c_a = band_a[2]
    c_b = band_b[2]
    if direction == "forward":
        dg_new[band_b[0] : band_b[1] + 1] = c_a
    else:
        dg_new[band_a[0] : band_a[1] + 1] = c_b
    return dg_new


def remove_layer_from_solution(
    optimizer,
    params,
    layer_to_remove,
    final_tau,
    h,
    current_max_layers,
    rng_seed,
):
    """
    Remove one layer from the solution.

    Args:
        params (dict): Current parameters with keys "global_logits" and "pixel_height_logits".
        layer_to_remove (int): Candidate layer index to remove.
        final_tau (float): Final tau value used in discretization/compositing.
        h (float): Layer height.
        current_max_layers (int): Current total number of layers.
        rng_seed (int): Seed used in discretization/compositing.

    Returns:
        new_params (dict): New parameters with the candidate layer removed.
        new_max_layers (int): Updated number of layers.
    """
    # Get the current discrete height image (used to decide which pixels need adjusting)
    effective_logits = optimizer._apply_height_offset(
        params["pixel_height_logits"], params["height_offsets"]
    )
    _, disc_height = optimizer.discretize_solution(
        params,
        final_tau,
        h,
        current_max_layers,
        rng_seed,
    )

    # Remove the candidate layer from the global (color) assignment.
    new_global_logits = torch.cat(
        [
            params["global_logits"][:layer_to_remove],
            params["global_logits"][layer_to_remove + 1 :],
        ],
        dim=0,
    )
    new_max_layers = new_global_logits.shape[0]

    # Compute current effective height: height = (current_max_layers * h) * sigmoid(pixel_height_logits)
    current_height = current_max_layers * h * torch.sigmoid(effective_logits)
    new_height = current_height.clone()
    mask = disc_height >= layer_to_remove
    new_height[mask] = new_height[mask] - h

    # Invert the sigmoid mapping:
    # We need new_pixel_height_logits such that:
    #    sigmoid(new_pixel_height_logits) = new_height / (new_max_layers * h)
    eps = 1e-6
    ratio = torch.clamp(new_height / (new_max_layers * h), eps, 1.0 - eps)
    new_effective_logits = torch.log(ratio) - torch.log1p(-ratio)
    new_pixel_height_logits = new_effective_logits - (
        effective_logits - params["pixel_height_logits"]
    )
    new_params = {
        "global_logits": new_global_logits,
        "pixel_height_logits": new_pixel_height_logits,
        "height_offsets": params["height_offsets"],
    }
    return new_params, new_max_layers


def prune_redundant_layers(
    optimizer: FilamentOptimizer,
    perception_loss_module,  # kept for API compatibility
    pruning_min_layers: int = 0,
    pruning_max_layers: int = int(1e6),
    tolerance: float = 0.10,  # kept for API compatibility
    n_jobs: int | None = -1,  # how many workers joblib should spawn
    *,
    fast: bool = True,  # NEW: enable 10 percent incremental search
    chunking_percent=0.05,  # percentage of layers to process at once
    allowed_loss_increase_percent=0.000,  # percentage of loss increase allowed over best loss
    preview_callback=None,
):
    """Iteratively drop layers until the loss cannot be improved.

    When *fast* is True we evaluate candidate removals in 10 percent batches,
    stopping as soon as one batch yields an improvement.
    """
    current_max_layers = optimizer.best_params["global_logits"].shape[0]
    optimizer.max_layers = current_max_layers  # keep optimiser in sync

    # Baseline loss with current best parameters
    best_loss = get_initial_loss(current_max_layers, optimizer)

    print(
        f"PRUNING: Layer - initial loss={best_loss:.4f}, initial layer={current_max_layers:d}"
    )

    tbar = tqdm(
        desc=f"Layer pruning | Loss {best_loss:.4f}",
        total=max(current_max_layers - 1, 0),
        leave=False,
    )
    removed_layers = 0
    improvement = True

    # ----------------------------------------------------------
    # Inner helper: evaluate the effect of removing layer idx
    # ----------------------------------------------------------
    def score_layer(idx: int) -> tuple[float, dict, int]:
        cand_params, cand_max_layers = remove_layer_from_solution(
            optimizer,
            optimizer.best_params,
            idx,
            optimizer.vis_tau,
            optimizer.h,
            current_max_layers,
            optimizer.best_seed,
        )
        eff_logits = optimizer._apply_height_offset(
            cand_params["pixel_height_logits"], cand_params["height_offsets"]
        )
        with _gpu_lock, torch.no_grad():
            cand_comp = composite_image_disc(
                eff_logits,
                cand_params["global_logits"],
                optimizer.vis_tau,
                optimizer.vis_tau,
                optimizer.h,
                cand_max_layers,
                optimizer.material_colors,
                optimizer.material_TDs,
                optimizer.background,
                rng_seed=optimizer.best_seed,
            )
            cand_loss = compute_loss(cand_comp, optimizer.target, focus_map=optimizer.focus_map).item()
        return cand_loss, cand_params, cand_max_layers

    # ----------------------------------------------------------
    # Main pruning loop
    # ----------------------------------------------------------
    while current_max_layers > pruning_min_layers and (
        improvement or current_max_layers > pruning_max_layers
    ):
        tbar.update(1)
        improvement = False
        layer_indices = list(range(current_max_layers))

        if preview_callback is not None:
            try:
                preview_callback(
                    optimizer,
                    int((1 - current_max_layers / max(pruning_max_layers, 1)) * 100),
                )
            except Exception:
                pass

        if fast:
            random.shuffle(layer_indices)
            chunk_size = max(1, math.ceil(len(layer_indices) * chunking_percent))
            best_candidate = None
            best_cand_loss = float("inf")

            for chunk in _chunked(layer_indices, chunk_size):
                cand_results = Parallel(
                    n_jobs=n_jobs, backend="threading", prefer="threads"
                )(delayed(score_layer)(idx) for idx in chunk)

                cand_loss, cand_params, cand_max_layers = min(
                    cand_results, key=lambda x: x[0]
                )

                if cand_loss <= best_loss * (1 + allowed_loss_increase_percent):
                    # Accept immediately and restart outer loop
                    removed_layers += 1
                    best_loss = cand_loss
                    current_max_layers = cand_max_layers
                    optimizer.best_params = cand_params
                    optimizer.max_layers = current_max_layers
                    tbar.set_description(
                        f"Layer pruning | Loss {best_loss:.4f} | "
                        f"Removed {removed_layers} | Layers {current_max_layers}"
                    )
                    improvement = True
                    break

                # Track the best candidate in case we must enforce a hard limit
                if cand_loss < best_cand_loss:
                    best_cand_loss = cand_loss
                    best_candidate = (cand_params, cand_max_layers)

            # Forced removal to meet pruning_max_layers
            if (
                not improvement
                and current_max_layers > pruning_max_layers
                and best_candidate is not None
            ):
                cand_params, cand_max_layers = best_candidate
                removed_layers += 1
                best_loss = best_cand_loss
                current_max_layers = cand_max_layers
                optimizer.best_params = cand_params
                optimizer.max_layers = current_max_layers
                tbar.set_description(
                    f"Layer pruning | Loss {best_loss:.4f} | "
                    f"Removed {removed_layers} | Layers {current_max_layers}"
                )
                improvement = True

            if not improvement:
                break  # no chunk helped and layer count is within budget
        else:
            # Exhaustive search
            cand_results = Parallel(
                n_jobs=n_jobs, backend="threading", prefer="threads"
            )(delayed(score_layer)(idx) for idx in layer_indices)

            cand_loss, cand_params, cand_max_layers = min(
                cand_results, key=lambda x: x[0]
            )

            tbar.set_description(
                f"Layer pruning | Current Candidate Loss: {cand_loss:.4f} | Best Loss {best_loss:.4f} | "
                f"Removed {removed_layers} | Layers {current_max_layers}"
            )

            if (
                cand_loss <= best_loss * (1 + allowed_loss_increase_percent)
                or current_max_layers > pruning_max_layers
            ):
                removed_layers += 1
                best_loss = cand_loss
                current_max_layers = cand_max_layers
                optimizer.best_params = cand_params
                optimizer.max_layers = current_max_layers

                improvement = True
            else:
                break

    print(
        f"PRUNING: Layers - final loss={best_loss:.4f}, final layers={current_max_layers:d}"
    )

    tbar.close()
    return optimizer.best_params, best_loss, current_max_layers


def get_initial_loss(current_max_layers, optimizer):
    with _gpu_lock, torch.no_grad():
        eff_logits = optimizer._apply_height_offset(
            optimizer.best_params["pixel_height_logits"],
            optimizer.best_params["height_offsets"],
        )
        ref_comp = composite_image_disc(
            eff_logits,
            optimizer.best_params["global_logits"],
            optimizer.vis_tau,
            optimizer.vis_tau,
            optimizer.h,
            current_max_layers,
            optimizer.material_colors,
            optimizer.material_TDs,
            optimizer.background,
            rng_seed=optimizer.best_seed,
        )
        best_loss = compute_loss(ref_comp, optimizer.target, focus_map=optimizer.focus_map).item()
    return best_loss


def remove_outlier_pixels(
    height_logits: torch.Tensor, threshold: float
) -> torch.Tensor:
    """
    For every pixel in `height_logits`, if at least six out of its 8 neighbors have an
    absolute difference greater than `threshold` from the center pixel, replace the pixel
    with the average of only those neighbors exceeding the threshold.

    Args:
        height_logits (torch.Tensor): 2D tensor representing the depth map.
        threshold (float): The threshold value for differences.

    Returns:
        torch.Tensor: The cleaned depth map.
    """
    # Define the eight neighbor shifts (row_shift, col_shift)
    shifts = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]

    # Collect neighbors using torch.roll (note: this wraps around at the edges)
    neighbors = []
    for dx, dy in shifts:
        shifted = torch.roll(
            torch.roll(height_logits, shifts=dx, dims=0), shifts=dy, dims=1
        )
        neighbors.append(shifted)

    # Stack the neighbors to shape (8, H, W)
    neighbors = torch.stack(neighbors, dim=0)

    # Compute the absolute difference between each neighbor and the center pixel
    diff = torch.abs(neighbors - height_logits)

    # Create a boolean mask for neighbors exceeding the threshold
    valid = diff > threshold  # shape (8, H, W)
    count_valid = valid.sum(dim=0)  # number of neighbors exceeding threshold per pixel

    # Determine which pixels should be replaced (if at least six neighbors exceed threshold)
    mask = count_valid >= 6

    # Compute the sum of valid neighbor values for each pixel
    # We use torch.where to zero out values not exceeding the threshold.
    valid_neighbors = torch.where(
        valid,
        neighbors,
        torch.tensor(0.0, dtype=neighbors.dtype, device=neighbors.device),
    )
    sum_valid = valid_neighbors.sum(dim=0)

    # Compute the average for the valid neighbors.
    # We use count_valid as the divisor; note that for pixels not meeting the criteria, the value is unused.
    avg_valid = sum_valid / count_valid.clamp(min=1)

    # Create a copy to update the pixels in outlier regions
    cleaned_height_logits = height_logits.clone()
    cleaned_height_logits[mask] = avg_valid[mask]

    num_changed = mask.sum().item()

    return cleaned_height_logits


def prune_fireflies(optimizer, start_threshold=10, auto_set=True):
    """
    Iteratively reduces the threshold, computes the loss for each,
    and returns the pixel_height_logits with the best loss.

    Args:
        optimizer: Object that provides get_best_discretized_image() and has target attribute.
        compute_loss (function): Function to compute loss; expects parameters comp and target.
        start_threshold (float): Initial threshold value.
        auto_set (bool): Automatically set the best pixel_height_logits in the optimizer.

    Returns:
        best_custom_height_logits (torch.Tensor): The processed depth map with best loss.
        best_threshold (float): The threshold value that achieved the best loss.
        best_loss (float): The best loss achieved.
    """
    # Generate a series of threshold values between start_threshold and end_threshold.
    pixel_height_logits = optimizer.best_params["pixel_height_logits"]
    best_custom_height_logits = pixel_height_logits
    best_threshold = start_threshold
    th = start_threshold

    with torch.no_grad():
        out_im = optimizer.get_best_discretized_image(
            custom_height_logits=pixel_height_logits
        )
        best_loss = compute_loss(
            comp=out_im,
            target=optimizer.target,
            focus_map=optimizer.focus_map,
        )
    new_loss = best_loss
    while th > 0.1:
        # Apply your outlier removal with the current threshold.
        custom_height_logits = remove_outlier_pixels(pixel_height_logits, threshold=th)

        # Evaluate the resulting image by computing the loss.
        with torch.no_grad():
            out_im = optimizer.get_best_discretized_image(
                custom_height_logits=custom_height_logits
            )
            loss = compute_loss(
                comp=out_im,
                target=optimizer.target,
                focus_map=optimizer.focus_map,
            )

        # print(f"Threshold: {th:.3f}, Loss: {loss:.4f}")
        # Track the best performing threshold.
        if loss < best_loss * 1.05:
            new_loss = best_loss
            best_custom_height_logits = custom_height_logits
            best_threshold = th

        th *= 0.95

    print(f"New loss: {new_loss:.4f}, Best threshold: {best_threshold:.3f}")
    if auto_set:
        optimizer.best_params["pixel_height_logits"] = best_custom_height_logits

    return best_custom_height_logits


def smooth_coplanar_faces(
    height_logits: torch.Tensor, angle_threshold: float
) -> torch.Tensor:
    """
    Smooths regions in the depth map that are considered coplanar.
    For each pixel, this function computes an approximate surface normal via finite differences
    and compares it to the normals of the eight neighboring pixels. Neighbors with an angle
    difference (in degrees) less than `angle_threshold` are considered coplanar. The pixel is
    replaced with the average of itself and its coplanar neighbors.

    Args:
        height_logits (torch.Tensor): 2D tensor representing the depth map.
        angle_threshold (float): Maximum angle difference (in degrees) for neighbors to be
                                 considered coplanar.

    Returns:
        torch.Tensor: The smoothed depth map.
    """
    # Convert the angle threshold from degrees to radians.
    threshold_rad = math.radians(angle_threshold)

    # Compute gradients using central differences via torch.roll (wraps at the edges)
    grad_x = (
        torch.roll(height_logits, shifts=-1, dims=1)
        - torch.roll(height_logits, shifts=1, dims=1)
    ) / 2.0
    grad_y = (
        torch.roll(height_logits, shifts=-1, dims=0)
        - torch.roll(height_logits, shifts=1, dims=0)
    ) / 2.0

    # Approximate the surface normals.
    # For a height function h(x,y), one common estimate is n = [-dh/dx, -dh/dy, 1] (then normalized)
    ones = torch.ones_like(height_logits)
    normal_x = -grad_x
    normal_y = -grad_y
    normal_z = ones
    norm = torch.sqrt(normal_x**2 + normal_y**2 + normal_z**2)
    normal_x /= norm
    normal_y /= norm
    normal_z /= norm
    # Stack into a tensor of shape (3, H, W)
    normals = torch.stack([normal_x, normal_y, normal_z], dim=0)

    # Define the eight neighbor shifts (for height map, we work with the spatial dims 0 and 1)
    shifts = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]

    # Initialize accumulators for the heights and a count of contributing pixels.
    # We always include the center pixel.
    coplanar_sum = height_logits.clone()
    count = torch.ones_like(height_logits)

    # For each neighbor, compute the angle difference between the center and neighbor normals.
    for dx, dy in shifts:
        # Shift the normals to obtain the neighbor's normal at each pixel.
        # Note: the normals tensor shape is (3, H, W) so we shift dims 1 and 2.
        neighbor_normals = torch.roll(
            torch.roll(normals, shifts=dx, dims=1), shifts=dy, dims=2
        )
        # Dot product between the center normals and the neighbor normals.
        dot = (normals * neighbor_normals).sum(dim=0)
        # Clamp dot products for numerical stability.
        dot = dot.clamp(-1.0, 1.0)
        # Compute the angle difference (in radians)
        angle_diff = torch.acos(dot)
        # Determine which neighbors are nearly coplanar.
        mask = angle_diff < threshold_rad
        # Retrieve the corresponding neighbor heights from height_logits.
        neighbor_heights = torch.roll(
            torch.roll(height_logits, shifts=dx, dims=0), shifts=dy, dims=1
        )
        # Add neighbor height values where the coplanar condition is met.
        coplanar_sum += neighbor_heights * mask.float()
        count += mask.float()

    # Compute the average height for the coplanar region.
    smoothed_height_logits = coplanar_sum / count.clamp(min=1)

    return smoothed_height_logits


def optimise_swap_positions(
    optimizer: FilamentOptimizer,
    n_jobs: int | None = -1,
    *,
    allowed_loss_increase_percent: float = 0.0,
    preview_callback=None,
) -> torch.Tensor:
    """
    Exhaustively move each swap boundary to every admissible layer and keep the
    best configuration. Uses a progress bar to display changes.

    Returns
    -------
    torch.Tensor
        The discrete global assignment with optimised swap positions.
    """

    num_materials = optimizer.material_colors.shape[0]

    def disc_loss(dg_test: torch.Tensor) -> float:
        logits_for_disc = disc_to_logits(dg_test, num_materials, big_pos=1e5)
        with _gpu_lock, torch.no_grad():
            out = optimizer.get_best_discretized_image(
                custom_global_logits=logits_for_disc
            )
            return compute_loss(comp=out, target=optimizer.target, focus_map=optimizer.focus_map).item()

    best_dg, _ = optimizer.get_discretized_solution(best=True)
    best_loss = disc_loss(best_dg)
    print(f"PRUNING: Swap position - initial loss={best_loss:.4f}")

    outer_tbar = tqdm(desc="Optimising swap positions", total=100, leave=False)
    improved = True
    pass_idx = 0
    while improved:
        improved = False
        pass_idx += 1
        outer_tbar.update(1)

        if preview_callback is not None:
            try:
                preview_callback(optimizer, min(90 + pass_idx, 99))
            except Exception:
                pass

        bands = find_color_bands(best_dg)
        num_swaps = len(bands) - 1
        if num_swaps == 0:
            break

        for swap_idx in range(num_swaps):
            band_a = bands[swap_idx]
            band_b = bands[swap_idx + 1]

            lower_limit = bands[swap_idx - 1][1] + 1 if swap_idx > 0 else 0
            upper_limit = (
                bands[swap_idx + 2][0] - 1
                if swap_idx + 2 < len(bands)
                else best_dg.size(0) - 1
            )

            if lower_limit >= upper_limit:
                continue

            def candidate_loss(new_boundary: int):
                if new_boundary == band_a[1]:
                    return float("inf"), None, None
                dg_new = best_dg.clone()
                dg_new[lower_limit : new_boundary + 1] = band_a[2]
                dg_new[new_boundary + 1 : upper_limit + 1] = band_b[2]
                return disc_loss(dg_new), dg_new, new_boundary

            results = Parallel(n_jobs=n_jobs, backend="threading", prefer="threads")(
                delayed(candidate_loss)(b) for b in range(lower_limit, upper_limit + 1)
            )

            cand_loss, cand_dg, new_pos = min(results, key=lambda x: x[0])

            if cand_loss < best_loss * (1 + allowed_loss_increase_percent):
                old_pos = band_a[1]
                best_dg = cand_dg
                best_loss = cand_loss
                optimizer.best_params["global_logits"] = disc_to_logits(
                    best_dg, num_materials=num_materials, big_pos=1e5
                )
                outer_tbar.set_description(
                    f"Swap {swap_idx} moved {old_pos}->{new_pos} | Loss {best_loss:.4f}"
                )
                # print(
                #     f"Swap {swap_idx} moved {old_pos}->{new_pos} | Loss {best_loss:.4f}"
                # )
                improved = True
                break

    outer_tbar.close()
    print(f"PRUNING: Swap position - final loss={best_loss:.4f}")
    return best_dg


def _compute_loss_for_heightmap(
    optimizer: FilamentOptimizer,
    disc_global: torch.Tensor,
    *,
    custom_height_logits: torch.Tensor | None = None,
) -> float:
    """Compute discrete loss for a given discrete height map/global assignment.

    If ``custom_height_logits`` is provided, they are used directly (bypassing
    discretization) when compositing the image; otherwise the optimizer's current
    best height logits are used.
    """
    logits_for_disc = disc_to_logits(
        disc_global, optimizer.material_colors.shape[0], big_pos=1e5
    )
    with _gpu_lock, torch.no_grad():
        out_im = optimizer.get_best_discretized_image(
            custom_height_logits=custom_height_logits,
            custom_global_logits=logits_for_disc,
        )
        return compute_loss(comp=out_im, target=optimizer.target, focus_map=optimizer.focus_map).item()


def _median_without_outliers(
    window: np.ndarray, threshold: float, max_outliers: int = 2
) -> tuple[float, bool, int]:
    """Return replacement for center pixel if it is a spike; also report spike count.

    A spike is when the center pixel is >= ``threshold`` above the window median and
    the window has at most ``max_outliers`` such outliers. Returns (replacement_value,
    is_spike, spike_count).
    """
    med = np.median(window)
    outlier_mask = window - med >= threshold
    spike_count = int(outlier_mask.sum())
    center_is_outlier = bool(outlier_mask[1, 1])
    if not center_is_outlier or spike_count == 0 or spike_count > max_outliers:
        return float(window[1, 1]), False, 0
    non_outliers = window[~outlier_mask]
    if non_outliers.size == 0:
        return float(window[1, 1]), False, 0
    replacement = float(np.median(non_outliers))
    return replacement, True, spike_count


def remove_height_spikes(
    disc_height: torch.Tensor, threshold_layers: int = 3, max_outliers: int = 2
) -> tuple[torch.Tensor, int]:
    """Remove tall spikes in a discrete height map using a 3x3 neighborhood.

    A spike occurs only when the *center* pixel of a 3x3 window is at least
    ``threshold_layers`` higher than the window median and the window has at most
    two such outliers. Only those center pixels are replaced with the median of
    the non-outlier values. Operates in layer units (0..max_layers).

    Returns:
        cleaned_height: torch.Tensor matching ``disc_height`` shape/device.
        spike_count: int number of pixels corrected.
    """
    if disc_height.ndim != 2:
        raise ValueError("disc_height must be 2D [H,W]")
    if threshold_layers <= 0:
        return disc_height, 0
    if max_outliers <= 0:
        return disc_height, 0

    dh_np = disc_height.detach().cpu().numpy()
    H, W = dh_np.shape
    if H < 3 or W < 3:
        return disc_height, 0

    out = dh_np.copy()
    spikes = 0
    queue = deque((y, x) for y in range(1, H - 1) for x in range(1, W - 1))
    total_num = len(queue)
    max_iters = (H - 2) * (W - 2) * 10  # safety cap
    iters = 0
    tbar = tqdm(total=total_num, desc="Removing height spikes", leave=False)
    while queue and iters < max_iters:
        tbar.update(1)
        y, x = queue.popleft()
        iters += 1
        window = out[y - 1 : y + 2, x - 1 : x + 2]
        replacement, is_spike, _ = _median_without_outliers(
            window, threshold_layers, max_outliers
        )
        center_val = out[y, x]
        if (
            not is_spike
            or center_val < replacement + threshold_layers
            or replacement == center_val
        ):
            continue
        out[y, x] = replacement
        spikes += 1
        for ny in range(y - 1, y + 2):
            if ny <= 0 or ny >= H - 1:
                continue
            for nx in range(x - 1, x + 2):
                if nx <= 0 or nx >= W - 1:
                    continue
                queue.append((ny, nx))
                total_num += 1
                tbar.total = total_num
    tbar.close()

    if spikes == 0:
        return disc_height, 0

    cleaned = torch.tensor(out, dtype=disc_height.dtype, device=disc_height.device)
    return cleaned, spikes
