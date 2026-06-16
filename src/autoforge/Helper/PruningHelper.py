import math
import random

import torch
import torch.nn.functional as F
from tqdm import tqdm
from joblib import Parallel, delayed
import threading
import numpy as np

from autoforge.Helper.OptimizerHelper import (
    composite_image_disc,
    adaptive_round,
    bleed_layer_effect,
    deterministic_rand_like,
)
from autoforge.Loss.LossFunctions import compute_loss
from autoforge.Modules.Optimizer import FilamentOptimizer

# One global lock that serialises every call that needs GPU / VRAM
_gpu_lock = threading.Lock()


def _make_shared_eff_thick(optimizer: FilamentOptimizer) -> torch.Tensor:
    """
    Compute the [L, H, W] effective-thickness prefix shared by all pruning
    candidates.  This is the expensive part of the composite pipeline and is
    independent of which materials are assigned to each layer.
    """
    eff_logits = optimizer._apply_height_offset(
        optimizer.best_params["pixel_height_logits"],
        optimizer.best_params["height_offsets"],
    )
    max_layers = optimizer.max_layers
    h = optimizer.h
    device = eff_logits.device

    pixel_height = (float(max_layers) * h) * torch.sigmoid(eff_logits)  # [H,W]
    z_cont = pixel_height / h
    z_disc = adaptive_round(z_cont, optimizer.vis_tau, 1.0, 0.0, 0.1)
    z_disc = torch.clamp(z_disc, 0.0, float(max_layers))
    z_int = torch.round(z_disc).to(torch.int64)  # [H,W]

    layer_idx = torch.arange(max_layers, device=device).view(-1, 1, 1)  # [L,1,1]
    p_print = (layer_idx < z_int.unsqueeze(0)).to(eff_logits.dtype)      # [L,H,W]
    p_bleed = bleed_layer_effect(p_print, 0.1)                           # [L,H,W]
    del p_print
    eff = torch.clamp(p_bleed, 0.0, 1.0) * h                             # [L,H,W]
    del p_bleed
    return eff


def _opacity_from_ratio(ratio: torch.Tensor) -> torch.Tensor:
    """ratio [H,W] → opac [H,W]  (same formula as composite_image_disc) ."""
    o, A, kk, bb = -2.9864511e-02, 4.0532556e-01, 8.2597107e+01, 1.2547257e+00
    return (o + A * torch.log1p(kk * ratio) + bb * ratio).clamp(0, 1)


def _compose_candidate(
    eff_thick: torch.Tensor,
    material_colors: torch.Tensor,
    material_TDs: torch.Tensor,
    background: torch.Tensor,
) -> torch.Tensor:
    """
    Iterative top-to-bottom compositing for one candidate.
    No [L, H, W] intermediates created; just [H,W] temporaries.

    Parameters
    ----------
    eff_thick : [L, H, W]  effective thickness per layer (shared prefix)
    material_colors : [L, 3]  per-layer material color
    material_TDs : [L]  per-layer transmission distance
    background : [3]  background color

    Returns [H, W, 3]
    """
    device = eff_thick.device
    L, H, W = eff_thick.shape
    comp = torch.zeros(H, W, 3, device=device, dtype=torch.float32)
    remain = torch.ones(H, W, device=device, dtype=torch.float32)

    for l in range(L - 1, -1, -1):  # top → bottom
        ratio = eff_thick[l] / material_TDs[l]                            # [H,W]
        opac = _opacity_from_ratio(ratio)                                 # [H,W]

        comp += remain.unsqueeze(-1) * opac.unsqueeze(-1) * material_colors[l].view(1, 1, 3)
        remain *= (1.0 - opac)

    comp += remain.unsqueeze(-1) * background
    return comp * 255.0


def material_select_from_logits(
    global_logits: torch.Tensor,
    material_colors: torch.Tensor,
    material_TDs: torch.Tensor,
    rng_seed: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Material selection for a single candidate [L, M].

    Returns (layer_colors [L,3], layer_TDs [L]).
    """
    L, M = global_logits.shape
    cols = torch.empty(L, 3, device=global_logits.device, dtype=material_colors.dtype)
    tds = torch.empty(L, device=global_logits.device, dtype=material_TDs.dtype)

    for j in range(L):
        noise = deterministic_rand_like(global_logits[j], rng_seed + j)
        g = -torch.log(-torch.log(noise + 1e-20) + 1e-20)
        y = torch.softmax((global_logits[j] + g) / 0.01, dim=-1)
        idx = y.argmax()
        cols[j] = material_colors[idx]
        tds[j] = material_TDs[idx].clamp(1e-8, 1e8)

    return cols, tds


def _material_select_batched(
    global_logits_b: torch.Tensor,
    material_colors: torch.Tensor,
    material_TDs: torch.Tensor,
    rng_seed: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Batched material selection [B, L, M] → ([B, L, 3], [B, L])."""
    B, L, M = global_logits_b.shape
    cols = torch.empty(B, L, 3, device=global_logits_b.device, dtype=material_colors.dtype)
    tds = torch.empty(B, L, device=global_logits_b.device, dtype=material_TDs.dtype)

    for j in range(L):
        gl_j = global_logits_b[:, j, :]                                  # [B, M]
        noise = deterministic_rand_like(global_logits_b[0, j], rng_seed + j)
        g = -torch.log(-torch.log(noise + 1e-20) + 1e-20)
        g_b = g.unsqueeze(0).expand(B, -1)                                # [B, M]
        y = torch.softmax((gl_j + g_b) / 0.01, dim=-1)                    # [B, M]
        idx = y.argmax(dim=-1)                                            # [B]
        cols[:, j] = material_colors[idx]
        tds[:, j] = material_TDs[idx].clamp(1e-8, 1e8)

    return cols, tds


def _eval_candidates_batch(
    optimizer: FilamentOptimizer,
    dg_candidates: list[torch.Tensor],
    eff_thick: torch.Tensor | None = None,
) -> list[tuple[float, torch.Tensor]]:
    """
    Evaluate a batch of discrete-assignment candidates, returning
    ``[(loss, dg), …]`` in the same order as *dg_candidates*.

    The expensive [L, H, W] effective thickness is either passed in or
    computed once and reused.  The score functions ``get_image_loss`` /
    ``score_color`` use a bare inner loop rather than joblib threads,
    eliminating GPU lock contention.
    """
    n_cands = len(dg_candidates)
    if n_cands == 0:
        return []

    if eff_thick is None:
        eff_thick = _make_shared_eff_thick(optimizer)

    num_materials = optimizer.material_colors.shape[0]

    # Build batched global_logits [B, L, M]
    L = dg_candidates[0].shape[0]
    gl_batch = torch.stack([
        disc_to_logits_vectorized(d, num_materials, big_pos=1e5)
        for d in dg_candidates
    ], dim=0)  # [B, L, M]

    # Material selection — batched (shared prefix)
    cols, tds = _material_select_batched(
        gl_batch, optimizer.material_colors, optimizer.material_TDs,
        rng_seed=optimizer.best_seed,
    )

    # Composite + loss per candidate — iterative, no [B, L, H, W]
    results = []
    for b in range(n_cands):
        with _gpu_lock, torch.no_grad():
            comp = _compose_candidate(
                eff_thick, cols[b], tds[b], optimizer.background,
            )
            loss = compute_loss(
                comp=comp, target=optimizer.target, focus_map=optimizer.focus_map,
            ).item()
        results.append((loss, dg_candidates[b]))

    return results


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
    logits = dg.new_full(
        (max_layers, num_materials), fill_value=-big_pos, dtype=torch.float32
    )
    logits.scatter_(
        dim=1,
        index=dg.contiguous().to(torch.long).view(-1, 1),
        value=big_pos,
    )
    return logits


def disc_to_logits_vectorized(
    dg: torch.Tensor, num_materials: int, big_pos: float = 1e5
) -> torch.Tensor:
    return disc_to_logits(dg,num_materials,big_pos)

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
    pruning_batch_size: int = 0,
) -> torch.Tensor:
    num_materials = optimizer.material_colors.shape[0]
    disc_global, _ = optimizer.get_discretized_solution(best=True)

    # Precompute shared prefix for batched evaluation (if batching enabled)
    shared_eff = None
    if pruning_batch_size > 0:
        shared_eff = _make_shared_eff_thick(optimizer)

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
            mat_colors = optimizer.material_colors
            pair_distances = [
                float(torch.sum((mat_colors[a] - mat_colors[b]) ** 2))
                for (a, b) in merge_pairs
            ]
            sorted_idx = sorted(
                range(len(merge_pairs)), key=lambda i: pair_distances[i]
            )
            merge_pairs = [merge_pairs[i] for i in sorted_idx]

            chunk_size = max(1, math.ceil(len(merge_pairs) * chunking_percent))

            improved = False
            best_candidate = None
            best_candidate_loss = float("inf")

            for chunk in _chunked(merge_pairs, chunk_size):
                if pruning_batch_size > 0:
                    dg_list = [merge_color(best_dg, *pair) for pair in chunk]
                    cand_results = _eval_candidates_batch(
                        optimizer, dg_list, eff_thick=shared_eff,
                    )
                else:
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
            if pruning_batch_size > 0:
                dg_list = [merge_color(best_dg, *pair) for pair in merge_pairs]
                cand_results = _eval_candidates_batch(
                    optimizer, dg_list, eff_thick=shared_eff,
                )
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
    pruning_batch_size: int = 0,
) -> torch.Tensor:
    """Reduce the number of color boundaries until it is <= max_swaps_allowed.
    If necessary, merges are forced even when they worsen the loss.
    """

    num_materials = optimizer.material_colors.shape[0]
    disc_global, _ = optimizer.get_discretized_solution(best=True)

    shared_eff = None
    if pruning_batch_size > 0:
        shared_eff = _make_shared_eff_thick(optimizer)

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
            mat_colors = optimizer.material_colors
            band_distances = [
                float(
                    torch.sum(
                        (mat_colors[spec[0][2]] - mat_colors[spec[1][2]]) ** 2
                    )
                )
                for spec in merge_specs
            ]
            sorted_idx = sorted(
                range(len(merge_specs)), key=lambda i: band_distances[i]
            )
            merge_specs = [merge_specs[i] for i in sorted_idx]

            chunk_size = max(1, math.ceil(len(merge_specs) * chunking_percent))

            improved = False
            best_candidate = None
            best_candidate_loss = float("inf")

            for chunk in _chunked(merge_specs, chunk_size):
                if pruning_batch_size > 0:
                    dg_list = [merge_bands(best_dg, *spec) for spec in chunk]
                    cand_results = _eval_candidates_batch(
                        optimizer, dg_list, eff_thick=shared_eff,
                    )
                else:
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
            if pruning_batch_size > 0:
                dg_list = [merge_bands(best_dg, *spec) for spec in merge_specs]
                cand_results = _eval_candidates_batch(
                    optimizer, dg_list, eff_thick=shared_eff,
                )
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
            # Sort layers by pixel coverage (fewest pixels first), so the
            # least impactful removals are tried first and the early-break
            # logic finds a good candidate sooner.
            _, disc_height = optimizer.get_discretized_solution(best=True)
            layer_counts = torch.bincount(
                disc_height.ravel().to(torch.int64), minlength=current_max_layers
            )
            layer_indices = sorted(
                layer_indices, key=lambda i: layer_counts[i].item()
            )

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


def remove_height_spikes(
    disc_height: torch.Tensor, threshold_layers: int = 3, max_outliers: int = 2,
    num_passes: int = 4,
) -> tuple[torch.Tensor, int]:
    """Remove tall spikes in a discrete height map using a 3x3 neighborhood.

    Vectorized GPU implementation — no BFS / Python loop per pixel.  Multiple
    passes approximate the old BFS propagation behaviour (fixing a spike may
    reveal new spikes in its neighbourhood on the next pass).

    A spike occurs only when the *center* pixel of a 3x3 window is at least
    ``threshold_layers`` higher than the window median and the window has at most
    ``max_outliers`` such outliers.  Those center pixels are replaced with the
    median of the non-outlier values.

    Args:
        disc_height: [H, W] tensor (any device).
        threshold_layers: pixels whose value exceeds window median by this
                          amount are considered outliers.
        max_outliers:  spike only if window has at most this many outliers.
        num_passes:    number of refinement passes (1 = one-shot, 4 ≈ old BFS).

    Returns:
        cleaned_height, spike_count
    """
    if disc_height.ndim != 2:
        raise ValueError("disc_height must be 2D [H,W]")
    if threshold_layers <= 0 or max_outliers <= 0 or num_passes < 1:
        return disc_height, 0

    H, W = disc_height.shape
    if H < 3 or W < 3:
        return disc_height, 0

    device = disc_height.device
    dtype = disc_height.dtype
    threshold = float(threshold_layers)

    # Pre-compute the 3x3 unfold once (it's the same every pass)
    img = disc_height.float().view(1, 1, H, W)
    img_pad = F.pad(img, (1, 1, 1, 1), mode="replicate")
    windows = F.unfold(img_pad, kernel_size=3, padding=0, stride=1)  # [1, 9, H*W]
    del img, img_pad

    # Static: window values are always the first 9 rows of the unfold result.
    # But after fixes we need to re-extract.  We'll recompute unfold each pass
    # on the current data; it's still sub-millisecond for 1500x1500.
    total_spikes = 0
    cur = disc_height.contiguous()

    for _ in range(num_passes):
        # Re-extract windows from current data
        img_cur = cur.float().view(1, 1, H, W)
        img_pad = F.pad(img_cur, (1, 1, 1, 1), mode="replicate")
        wins = F.unfold(img_pad, kernel_size=3, padding=0, stride=1)  # [1, 9, H*W]
        wins = wins.squeeze(0)  # [9, H*W]
        del img_cur, img_pad

        x_sorted, _ = wins.sort(dim=0)  # [9, H*W]
        median_all = x_sorted[4]         # [H*W]

        outlier_mask = (wins - median_all.unsqueeze(0)) >= threshold  # [9, H*W]
        is_center_outlier = outlier_mask[4]  # [H*W]
        outlier_count = outlier_mask.sum(dim=0)  # [H*W]
        del outlier_mask

        fix_mask = (
            is_center_outlier & (outlier_count <= max_outliers) & (outlier_count > 0)
        )
        n_this = int(fix_mask.sum().item())
        if n_this == 0:
            break
        total_spikes += n_this
        print("Removed",n_this,"spikes...")

        # Replacement: median of non-outlier values
        wins_masked = wins.clone()
        wins_masked[wins - median_all.unsqueeze(0) >= threshold] = float("inf")

        x_masked_sorted, _ = wins_masked.sort(dim=0)  # [9, H*W]
        n_valid = 9 - outlier_count  # [H*W]
        med_idx = (n_valid // 2).clamp(0, 8)

        replacement = x_masked_sorted[
            med_idx, torch.arange(H * W, device=device)
        ]
        replacement = torch.where(n_valid > 0, replacement, wins[4])

        flat = cur.view(-1)  # [H*W]
        flat[fix_mask] = replacement[fix_mask].to(dtype)

    if total_spikes == 0:
        return disc_height, 0

    return cur, total_spikes
