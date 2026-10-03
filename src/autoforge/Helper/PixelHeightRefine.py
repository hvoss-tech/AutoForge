"""Per-pixel height refinement of a finished discrete solution.

Training only moves the per-cluster height offsets; the per-pixel height
logits stay frozen at their k-means initialisation, so a pixel can only ever
sit at the height its cluster puts it. With the layer materials fixed, the
colour a pixel shows is (almost) a function of its own height alone - the
only coupling to its neighbours is the 3x3 edge bleed. That makes the
per-pixel height an ideal target for exact coordinate descent on the real
discrete loss:

* Pixels are split into 9 classes by (y mod 3, x mod 3). Changing one pixel
  only changes the composite inside its own 3x3 window, and windows of two
  pixels of the same class never overlap, so every pixel of a class can be
  moved at once and each move judged independently.
* For every candidate height, all pixels of the class are set to it, the
  image is composited through the real discrete path, and each pixel's
  3x3-window error is read off. Each pixel then takes its best candidate
  (its current height is always among them), so no sweep can raise the loss.
"""

import threading

import torch
import torch.nn.functional as F

from autoforge.Helper.ImageHelper import srgb_to_lab
from autoforge.Helper.OptimizerHelper import (
    _coverage,
    composite_graph_scope,
    composite_image_disc,
    srgb_lightness,
)


def _heights_to_eff_logits(z: torch.Tensor, max_layers: int) -> torch.Tensor:
    """Effective height logits that discretize back to exactly ``z``."""
    normalized = (z.to(torch.float32) / float(max_layers)).clamp(1e-6, 1 - 1e-6)
    return torch.log(normalized) - torch.log1p(-normalized)


def _heights_compositor(optimizer, dg, disc_logits, L):
    """``f(z) -> [H,W,3]`` discrete composite of integer heights ``z`` under
    the fixed stack ``dg``: the fused kernel on CUDA (see FusedComposite),
    else the real composite."""
    from autoforge.Helper import FusedComposite as fc

    if fc.fused_available(optimizer.material_colors):
        params = fc.stack_params(dg, optimizer.material_colors, optimizer.material_TDs, optimizer.background, optimizer.h)
        bg = fc.background_tensor(optimizer.background)
        return lambda z: fc.composite_heights(z, params, bg, optimizer.h)

    def f(z):
        return composite_image_disc(
            _heights_to_eff_logits(z, L), disc_logits, optimizer.vis_tau, optimizer.vis_tau,
            optimizer.h, L, optimizer.material_colors, optimizer.material_TDs, optimizer.background,
            rng_seed=optimizer.best_seed, compute_dtype=optimizer.composite_compute_dtype,
        )

    return f


def _heights_error_fn(optimizer, dg, disc_logits, L, target_lab, weights, smooth):
    """``f(z) -> [H,W]`` per-pixel weighted Lab squared error of heights
    ``z`` under the fixed stack plus ``smooth`` times the height variation;
    one fused kernel on CUDA."""
    from autoforge.Helper import FusedComposite as fc

    if fc.fused_available(optimizer.material_colors):
        params = fc.stack_params(dg, optimizer.material_colors, optimizer.material_TDs, optimizer.background, optimizer.h)
        bg = fc.background_tensor(optimizer.background)
        tl32 = target_lab.to(torch.float32).contiguous()
        w32 = weights.to(torch.float32).contiguous()
        return lambda z: fc.heights_error(z, params, bg, optimizer.h, tl32, w32, smooth)
    composite_z = _heights_compositor(optimizer, dg, disc_logits, L)

    def f(zmap):
        e = (srgb_to_lab(composite_z(zmap)) - target_lab).pow(2).sum(-1) * weights
        if smooth > 0:
            e = e + smooth * _height_variation(zmap)
        return e

    return f


def _pixel_weights(optimizer, shape):
    """Per-pixel loss weights matching compute_loss (focus map / alpha)."""
    w = torch.ones(shape, device=optimizer.device, dtype=torch.float32)
    if optimizer.focus_map is not None:
        fm = optimizer.focus_map
        if fm.dim() == 3:
            fm = fm.squeeze(-1)
        w = w * (0.1 + 0.9 * fm.clamp(min=0.0))
    if optimizer.alpha is not None:
        a = optimizer.alpha
        if a.dim() == 3:
            a = a.squeeze(-1)
        if a.shape != w.shape:
            a = F.interpolate(a[None, None], size=w.shape, mode="nearest")[0, 0]
        w = w * (a >= 128).float()
    return w


def _movable(optimizer, shape) -> torch.Tensor:
    """[H,W] bool: pixels a refine may move. Transparent pixels (alpha <
    128) stay unprinted, as in the optimizer's own height assignment: their
    colour error weighs nothing, but the height-jump cost and the edge bleed
    into their valid neighbours would otherwise raise them."""
    if optimizer.alpha is None:
        return torch.ones(shape, dtype=torch.bool, device=optimizer.device)
    a = optimizer.alpha
    if a.dim() == 3:
        a = a.squeeze(-1)
    if tuple(a.shape) != tuple(shape):
        a = F.interpolate(a[None, None].float(), size=shape, mode="nearest")[0, 0]
    return (a >= 128).to(optimizer.device)


def _height_variation(z: torch.Tensor) -> torch.Tensor:
    """[H,W] summed |height difference| (layers) to the 8 neighbours,
    diagonals weighted 1/sqrt(2); image borders count as equal height.

    Summed over a pixel's 3x3 window this is twice its own edges plus terms
    no move of that pixel can change, so adding it to the colour error makes
    the coordinate descent trade colour against height jumps exactly."""
    H, W = z.shape
    zf = z.to(torch.float32)
    pad = F.pad(zf[None, None], (1, 1, 1, 1), mode="replicate")[0, 0]
    out = torch.zeros_like(zf)
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dy == 0 and dx == 0:
                continue
            w = 1.0 if dy == 0 or dx == 0 else 0.7071
            out = out + w * (zf - pad[1 + dy : 1 + dy + H, 1 + dx : 1 + dx + W]).abs()
    return out


def _median3(z: torch.Tensor) -> torch.Tensor:
    """3x3 median of an integer height map (replicate border)."""
    from autoforge.Helper import FusedComposite as fc

    if fc.fused_available(z):
        return fc.median3(z)
    H, W = z.shape
    pad = F.pad(z.float()[None, None], (1, 1, 1, 1), mode="replicate")
    return F.unfold(pad, 3)[0].median(dim=0)[0].view(H, W).to(z.dtype)


def spike_mask(z: torch.Tensor, threshold: float, max_outliers: int = 2) -> torch.Tensor:
    """Pixels the first pass of ``remove_height_spikes`` would flag."""
    from autoforge.Helper import FusedComposite as fc

    if fc.fused_available(z):
        return fc.spike_mask(z, threshold, max_outliers)
    H, W = z.shape
    pad = F.pad(z.float().view(1, 1, H, W), (1, 1, 1, 1), mode="replicate")
    wins = F.unfold(pad, kernel_size=3)[0]  # [9, H*W]
    median = wins.sort(dim=0)[0][4]
    outlier = (wins - median) >= threshold
    count = outlier.sum(dim=0)
    return (outlier[4] & (count <= max_outliers) & (count > 0)).view(H, W)


def _drop_new_spikes(z_old, z_new, threshold, max_rounds: int = 8):
    """Undo moves in ``z_new`` that created spikes absent from ``z_old``.

    A new spike can come from the moved pixel itself or from a neighbour
    whose window median it shifted, so every changed pixel within one pixel
    of a new spike is reverted, repeated until no new spike is left.
    """
    before = spike_mask(z_old, threshold)
    for _ in range(max_rounds):
        new = spike_mask(z_new, threshold) & ~before
        if not bool(new.any()):
            return z_new
        near = F.max_pool2d(new.float()[None, None], 3, stride=1, padding=1)[0, 0] > 0
        z_new = torch.where(near & (z_new != z_old), z_old, z_new)
    return z_old


def _keep_heights_if_better(optimizer, new_logits, dg, pre_loss):
    """Install ``new_logits`` as the solution's height logits and keep them
    only if the real discrete loss beats ``pre_loss``; otherwise (or on an
    error while measuring) the previous logits are restored.
    Returns (kept, post_loss)."""
    from autoforge.Helper.PruningHelper import _compute_loss_for_heightmap

    old_best = optimizer.best_params["pixel_height_logits"]
    old_live = optimizer.pixel_height_logits
    optimizer.best_params["pixel_height_logits"] = new_logits
    optimizer.pixel_height_logits = new_logits
    kept = False
    try:
        post_loss = _compute_loss_for_heightmap(optimizer, dg)
        kept = post_loss < pre_loss
    finally:
        if not kept:
            optimizer.best_params["pixel_height_logits"] = old_best
            optimizer.pixel_height_logits = old_live
    return kept, post_loss


@composite_graph_scope()
def refine_pixel_heights(
    optimizer,
    sweeps: int = 2,
    radius: int = -1,
    spike_aware: bool = False,
    block: int = 1,
    progress=None,
) -> bool:
    """Coordinate descent over per-pixel heights (see module docstring).

    ``progress`` (optional) is called with the fraction 0-1 of the sweeps done.

    ``block`` > 1 moves ``block x block`` squares of pixels to one common
    height instead of single pixels (classes then repeat every block+2
    pixels so influence regions still never overlap). A 2x2 block can be
    raised without becoming a spike, which single-pixel moves on a
    spike-free map never can.

    ``radius`` limits candidates to ``current +- radius`` layers; -1 tries
    every height. Writes the result into ``best_params['pixel_height_logits']``
    and keeps it only if the real discrete loss improved. Returns True if kept.

    ``spike_aware`` rejects moves that create a spike ``remove_height_spikes``
    would flag (with the run's spike settings), so a height map that has
    already been cleaned of spikes stays clean.
    """
    from autoforge.Helper.PruningHelper import (
        _compute_loss_for_heightmap,
        disc_to_logits,
    )

    dg, z = optimizer.get_discretized_solution(best=True)
    if z is None:
        return False
    pre_loss = _compute_loss_for_heightmap(optimizer, dg)
    L = int(optimizer.max_layers)
    num_materials = optimizer.material_colors.shape[0]
    disc_logits = disc_to_logits(dg, num_materials, big_pos=1e5)
    z = z.to(torch.int64).clone()
    H, W = z.shape
    target_lab = srgb_to_lab(optimizer.target)
    weights = _pixel_weights(optimizer, (H, W))
    stride = block + 2
    kernel = torch.ones(1, 1, stride, stride, device=z.device)
    smooth = float(getattr(optimizer.args, "pixel_height_smoothness", 1.0))
    # Pull toward the height map pruning handed over (quadratic, so it is
    # the large departures that cost), stored on the first refine of a prune.
    anchor_w = float(getattr(optimizer.args, "pixel_height_anchor", 0.0))
    z_anchor = getattr(optimizer, "_refine_anchor_z", None)
    if z_anchor is None or z_anchor.shape != z.shape:
        z_anchor = z.clone()
        optimizer._refine_anchor_z = z_anchor
    z_anchor_f = z_anchor.to(torch.float32)
    spike_thr = float(getattr(optimizer.args, "spike_threshold_layers", 1))

    pixel_err = _heights_error_fn(optimizer, dg, disc_logits, L, target_lab, weights, smooth)

    def window_err(zmap):
        """Error summed over the influence region of the block anchored at
        each pixel (rows/cols -1 .. block)."""
        e = pixel_err(zmap)
        if anchor_w > 0:
            e = e + anchor_w * (zmap.to(torch.float32) - z_anchor_f).pow(2)
        e = F.pad(e[None, None], (1, block, 1, block))
        return F.conv2d(e, kernel)[0, 0]

    yy = torch.arange(H, device=z.device).view(-1, 1)
    xx = torch.arange(W, device=z.device).view(1, -1)
    movable = _movable(optimizer, (H, W))
    classes = []
    for a in range(stride):
        for b in range(stride):
            oy, ox = (yy - a) % stride, (xx - b) % stride
            ay, ax = yy - oy, xx - ox
            anchor = (oy == 0) & (ox == 0)
            member = (oy < block) & (ox < block) & (ay >= 0) & (ax >= 0) & movable
            classes.append((anchor, member, ay.clamp(min=0), ax.clamp(min=0)))
    if radius < 0:
        cands = range(L + 1)
    else:
        cands = [c for c in range(-radius, radius + 1) if c != 0]
        # Plus every pixel's free-height optimum: the height whose flat
        # stack colour is nearest its target colour. Edge bleed is the only
        # thing it ignores, so together with the local window it finds most
        # of what trying every height would, at a fraction of the cost.
        kk = torch.arange(L + 1, device=z.device).view(1, 1, -1)
        layer = torch.arange(L, device=z.device).view(-1, 1, 1)
        pal = srgb_to_lab(_stack_palette(optimizer, dg.to(torch.long), (layer < kk).float() * optimizer.h))
        pz = torch.cdist(target_lab.reshape(-1, 3), pal).argmin(dim=1).view(H, W)

    total_steps = max(1, sweeps * len(classes))
    done_steps = 0
    with torch.no_grad():
        for _ in range(sweeps):
            changed = 0
            for anchor, member, ay, ax in classes:
                if progress is not None:
                    progress(done_steps / total_steps)
                done_steps += 1
                best_err = window_err(z)
                best_z = z.clone()
                cand_maps = (
                    [torch.full_like(z, c) for c in cands]
                    if radius < 0
                    # pz and the 3x3 median (what the neighbours agree on)
                    # are per pixel;
                    # a block takes its anchor's pz / neighbourhood median.
                    else [(z + c).clamp(0, L) for c in cands]
                    + [pz[ay, ax], _median3(z)[ay, ax]]
                )
                for cand_val in cand_maps:
                    trial = torch.where(member, cand_val, z)
                    err = window_err(trial)
                    better = anchor & (err < best_err)
                    best_err = torch.where(better, err, best_err)
                    best_z = torch.where(member & better[ay, ax], trial, best_z)
                if spike_aware:
                    best_z = _drop_new_spikes(z, best_z, spike_thr)
                changed += int((best_z != z).sum())
                z = best_z
            if changed == 0:
                break

    new_logits = optimizer._remove_height_offset(
        pixel_logits=_heights_to_eff_logits(z, L),
        height_offsets=optimizer.best_params["height_offsets"],
    )
    kept, post_loss = _keep_heights_if_better(optimizer, new_logits, dg, pre_loss)
    spikes = int(spike_mask(z, spike_thr).sum())
    dev = float((z - z_anchor).abs().float().mean())
    print(
        f"Pixel height refine: loss {pre_loss:.4f} -> {post_loss:.4f} | "
        f"{'kept' if kept else 'reverted'} | spikes in map {spikes} | mean |dz| from start {dev:.2f}"
    )
    return kept


@composite_graph_scope()
def refine_layer_materials(
    optimizer,
    max_colors: int = 10**9,
    max_swaps: int = 10**9,
    sweeps: int = 1,
    window: int = 0,
) -> bool:
    """Coordinate descent over the material of each layer, heights fixed.
    ``window`` > 0 only tries the materials within that many layers of it.

    Every layer in turn tries every material; a candidate is only allowed if
    the solution stays within ``max_colors`` distinct materials and
    ``max_swaps`` swaps. Scored with the shared-thickness batched path and
    verified with the real metric at the end (reverted if not better).
    """
    from autoforge.Helper.PruningHelper import (
        _compute_loss_for_heightmap,
        _eval_candidates_batch,
        _shared_scoring_handle,
        disc_to_logits,
        find_color_bands,
    )

    dg, _ = optimizer.get_discretized_solution(best=True)
    if dg is None:
        return False
    pre_loss = _compute_loss_for_heightmap(optimizer, dg)
    num_materials = optimizer.material_colors.shape[0]
    L = int(dg.shape[0])
    with torch.no_grad():
        eff = _shared_scoring_handle(optimizer)
        best_dg = dg.clone()
        best, _ = _eval_candidates_batch(optimizer, [best_dg], eff_thick=eff)
        for _ in range(sweeps):
            improved = False
            for layer in range(L - 1, -1, -1):
                cands = []
                if window > 0:
                    mats = set(best_dg[max(0, layer - window) : layer + window + 1].tolist())
                else:
                    mats = range(num_materials)
                for m in mats:
                    if m == int(best_dg[layer]):
                        continue
                    c = best_dg.clone()
                    c[layer] = m
                    if _layer_colors(c, getattr(optimizer, "base_material", None)) > max_colors:
                        continue
                    if _stack_swaps(c, getattr(optimizer, "base_material", None)) > max_swaps:
                        continue
                    cands.append(c)
                if not cands:
                    continue
                loss, cand = _eval_candidates_batch(optimizer, cands, eff_thick=eff)
                if loss < best - 1e-6:
                    best, best_dg = loss, cand
                    improved = True
            if not improved:
                break

    old_logits = optimizer.best_params["global_logits"]
    optimizer.best_params["global_logits"] = disc_to_logits(best_dg, num_materials, big_pos=1e5)
    post_loss = _compute_loss_for_heightmap(optimizer, best_dg)
    kept = post_loss < pre_loss
    if not kept:
        optimizer.best_params["global_logits"] = old_logits
    print(
        f"Layer material refine: loss {pre_loss:.4f} -> {post_loss:.4f} | "
        f"{'kept' if kept else 'reverted'}"
    )
    return kept


def _stack_palette(optimizer, dg: torch.Tensor, eff: torch.Tensor) -> torch.Tensor:
    """[L+1,3] colour (0..255) of the stack ``dg`` printed up to 0..L layers
    (no edge bleed - every pixel of a flat area sees exactly this)."""
    from autoforge.Helper.PruningHelper import _compose_candidate

    cols = optimizer.material_colors.index_select(0, dg)
    tds = optimizer.material_TDs.index_select(0, dg).clamp(1e-8, 1e8)
    comp = _compose_candidate(eff, cols, tds, optimizer.background, optimizer.h)
    return comp[0]


def palette_loss_fn(optimizer):
    """Returns ``f(dg) -> loss tensor``: the loss the stack ``dg`` would reach
    if every pixel could independently take its best height (nearest palette
    colour in Lab), ignoring edge bleed. A cheap proxy for searching stacks
    when heights are going to be re-solved per pixel afterwards anyway."""
    L = int(optimizer.max_layers)
    dev = optimizer.material_colors.device
    k = torch.arange(L + 1, device=dev).view(1, 1, -1)
    layer = torch.arange(L, device=dev).view(-1, 1, 1)
    eff = (layer < k).to(torch.float32) * optimizer.h  # [L,1,L+1]
    target = optimizer.target
    tl = srgb_to_lab(target).reshape(-1, 3)
    w = _pixel_weights(optimizer, target.shape[:2]).reshape(-1)
    # Sub-sample large images: the proxy only has to rank stacks.
    if tl.shape[0] > 40000:
        idx = torch.randperm(tl.shape[0], device=dev, generator=None)[:40000]
        tl, w = tl[idx], w[idx]
    wsum = w.sum().clamp(min=1e-8)

    def f(dg):
        pal = srgb_to_lab(_stack_palette(optimizer, dg, eff))  # [L+1,3]
        d = torch.cdist(tl, pal).pow(2)  # [N,L+1]
        return (d.min(dim=1)[0] * w).sum() / wsum / 3.0

    return f


@composite_graph_scope()
def refine_stack_palette(
    optimizer,
    max_colors: int = 10**9,
    max_swaps: int = 10**9,
    sweeps: int = 4,
    window: int = 0,
) -> torch.Tensor:
    """Coordinate descent over layer materials under ``palette_loss_fn``.
    ``window`` > 0 only tries the materials within that many layers of each
    layer (as refine_layer_materials). Returns the new stack (does not touch
    the optimizer's solution)."""
    from autoforge.Helper.PruningHelper import find_color_bands

    dg, _ = optimizer.get_discretized_solution(best=True)
    f = palette_loss_fn(optimizer)
    num_materials = optimizer.material_colors.shape[0]
    L = int(dg.shape[0])
    best_dg = dg.clone().to(torch.long)
    with torch.no_grad():
        best = float(f(best_dg))
        start = best
        done = 0
        for _ in range(sweeps):
            done += 1
            improved = False
            for layer in range(L - 1, -1, -1):
                cands = []
                if window > 0:
                    mats = set(best_dg[max(0, layer - window) : layer + window + 1].tolist())
                else:
                    mats = range(num_materials)
                for m in mats:
                    if m == int(best_dg[layer]):
                        continue
                    c = best_dg.clone()
                    c[layer] = m
                    if max_colors < 10**9 and _layer_colors(c, getattr(optimizer, "base_material", None)) > max_colors:
                        continue
                    if max_swaps < 10**9 and _stack_swaps(c, getattr(optimizer, "base_material", None)) > max_swaps:
                        continue
                    cands.append(c)
                if not cands:
                    continue
                losses = torch.stack([f(c) for c in cands])
                i = int(torch.argmin(losses))
                if float(losses[i]) < best - 1e-6:
                    best, best_dg = float(losses[i]), cands[i]
                    improved = True
            if not improved:
                break
    print(f"Stack palette search: proxy loss {start:.4f} -> {best:.4f} ({done} sweeps)")
    return best_dg


def apply_stack(optimizer, dg: torch.Tensor, refine_sweeps: int = 2, radius: int = -1, progress=None) -> bool:
    """Swap in stack ``dg``, re-solve the heights per pixel, and keep the
    result only if the real loss beats the current solution."""
    before = optimizer.solution_loss()
    snap = optimizer.solution_snapshot()
    try:
        after = _apply_stack_unchecked(optimizer, dg, refine_sweeps, radius, progress)
    except BaseException:
        # Never leave the new stack half applied (stack swapped in, heights
        # not yet re-solved) behind an error.
        optimizer.restore_solution_snapshot(snap)
        raise
    shown = f"{_fmt_loss(before)} -> {_fmt_loss(after)}"
    if after is None or before is None or after >= before:
        optimizer.restore_solution_snapshot(snap)
        print(f"Stack swap: {shown} | reverted")
        return False
    print(f"Stack swap: {shown} | kept")
    return True


def _fmt_loss(loss) -> str:
    return "n/a" if loss is None else f"{loss:.4f}"


def _apply_stack_unchecked(optimizer, dg, refine_sweeps, radius, progress):
    """Swap in stack ``dg`` and re-solve the heights; the new loss."""
    from autoforge.Helper.PruningHelper import disc_to_logits

    optimizer.best_params["global_logits"] = disc_to_logits(
        dg, optimizer.material_colors.shape[0], big_pos=1e5
    )
    # Start from this stack's own assigned heights (the smoothed per-pixel
    # optimum), not the previous stack's: the refine only moves heights
    # locally and, from another stack's map, regularly got stuck above the
    # current loss and rejected every searched stack.
    if hasattr(optimizer, "assigned_heights"):
        with torch.no_grad():
            bp = optimizer.best_params
            eff = optimizer._apply_height_offset(bp["pixel_height_logits"], bp["height_offsets"])
            z = optimizer.assigned_heights(dg.to(torch.long), eff)
            new_logits = optimizer._remove_height_offset(
                pixel_logits=_heights_to_eff_logits(z, int(optimizer.max_layers)),
                height_offsets=bp["height_offsets"],
            )
        bp["pixel_height_logits"] = new_logits
        optimizer.pixel_height_logits = new_logits
    refine_pixel_heights(optimizer, sweeps=refine_sweeps, radius=radius, progress=progress)
    return optimizer.solution_loss()


@torch.jit.script
def _batched_stack_after(
    opac: torch.Tensor,  # [B,L]
    colors_ext: torch.Tensor,  # [B,L+1,3]
    above: torch.Tensor,  # [n,n] bool
    covered: torch.Tensor,  # [n,n] bool
) -> torch.Tensor:
    B = int(opac.shape[0])
    n = int(opac.shape[1]) + 1
    o_ext = torch.cat([torch.ones_like(opac[:, :1]), opac], 1)  # [B,n]
    trans = torch.where(above, (1.0 - o_ext).view(B, 1, n), torch.ones_like(o_ext[:1, :1]).view(1, 1, 1))
    through = torch.cumprod(trans, dim=2)
    weight = torch.where(covered, through * o_ext.view(B, n, 1), torch.zeros_like(o_ext[:1, :1]).view(1, 1, 1))
    return weight.transpose(1, 2) @ colors_ext  # [B,n,3]


@torch.jit.script
def _batched_opacity(
    dark: torch.Tensor,
    k: torch.Tensor,
    w_dark: torch.Tensor,
    run_thick: torch.Tensor,
    reach: torch.Tensor,
    continues: torch.Tensor,
) -> torch.Tensor:
    w = torch.where(dark, w_dark, torch.ones_like(w_dark))
    cov = _coverage(run_thick, reach, torch.sqrt(k * reach), w)
    cov_prev = torch.where(
        continues, torch.cat([torch.zeros_like(cov[:, :1]), cov[:, :-1]], 1), torch.zeros_like(cov)
    )
    return torch.clamp((cov - cov_prev) / torch.clamp(1.0 - cov_prev, min=1e-6), 0.0, 1.0)


@torch.jit.script
def batched_stack_palette(
    colors: torch.Tensor,  # [B,L,3] in 0..1, bottom to top
    tds: torch.Tensor,  # [B,L]
    background: torch.Tensor,  # [3]
    h: float,
    refine_steps: int = 8,
) -> torch.Tensor:
    """[B,L+1,3] (0..1) colour of B stacks printed up to 0..L full layers.

    The same model as ``layer_coverage_params`` + the compositing walk, but
    with a batch dimension and without an image: a flat area printed to
    height k shows exactly ``palette[k]``.
    """
    B = int(colors.shape[0])
    L = int(colors.shape[1])
    dev = colors.device
    colors = colors.to(torch.float32)
    tds = tds.to(torch.float32).clamp(1e-8, 1e8)
    n = L + 1
    idx = torch.arange(L, device=dev).view(1, L)
    differs = (colors[:, 1:] != colors[:, :-1]).any(-1) | (tds[:, 1:] != tds[:, :-1])
    change = torch.cat([torch.ones(B, 1, dtype=torch.bool, device=dev), differs], 1)
    run_start = torch.cummax(torch.where(change, idx, torch.zeros_like(idx)), dim=1)[0]
    run_thick = (idx - run_start + 1).to(torch.float32) * h
    continues = run_start < idx

    reach = torch.clamp(0.1 * tds, min=1e-9)
    w_dark = torch.clamp((1.0 - tds) / 0.4, 0.0, 1.0)
    layer_light = srgb_lightness(colors)
    darker_than_mid = layer_light < 0.5
    colors_ext = torch.cat([background.to(torch.float32).view(1, 1, 3).expand(B, 1, 3), colors], 1)
    i = torch.arange(n, device=dev)
    above = i.view(1, n) > i.view(n, 1)
    covered = above | (i.view(1, n) == i.view(n, 1))

    dark = torch.zeros_like(continues)
    k = torch.ones_like(run_thick)
    for _ in range(refine_steps):
        opac = _batched_opacity(dark, k, w_dark, run_thick, reach, continues)
        light_below = srgb_lightness(_batched_stack_after(opac, colors_ext, above, covered)[:, :-1])
        dark = (light_below > layer_light) & darker_than_mid
        denom = 1.0 - (light_below - layer_light)
        k = torch.where(denom > 0.0, light_below / torch.clamp(denom, min=1e-12), torch.ones_like(k))
        k = torch.clamp(k, min=1.0)
    return _batched_stack_after(
        _batched_opacity(dark, k, w_dark, run_thick, reach, continues), colors_ext, above, covered
    )


def _decision_dtype(device: torch.device) -> torch.dtype:
    """dtype of the on-device accept/reject bookkeeping in the stack searches:
    float64 (as the Python-float comparison it replaced), except on MPS, which
    has no float64 - there fp32, exact for the fp32 losses it compares."""
    from autoforge.Helper import DeviceUtils

    return torch.float64 if DeviceUtils.has_float64(device) else torch.float32


class PaletteProxy:
    """Batched ``palette_loss_fn``: scores many stacks at once.

    The target pixels are merged into weighted Lab grid bins (bin mean as
    the representative colour), growing the bin size until at most
    ``max_points`` remain - a proxy only has to rank stacks, and this keeps
    the [stacks, points, heights] distance tensor small.
    """

    def __init__(self, optimizer, max_points: int = 2048, chunk: int = 64):
        self.opt = optimizer
        self.chunk = chunk
        tl = srgb_to_lab(optimizer.target).reshape(-1, 3)
        w = _pixel_weights(optimizer, optimizer.target.shape[:2]).reshape(-1)
        keep = w > 0
        tl, w = tl[keep], w[keep]
        size = 1.0
        while True:
            q = torch.round(tl / size).to(torch.int64)
            _, inv, _ = torch.unique(q, dim=0, return_inverse=True, return_counts=True)
            nb = int(inv.max()) + 1
            if nb <= max_points:
                break
            size *= 1.25
        # Bin sums on the CPU (float64 bincount): index_add_ on the GPU sums
        # with float atomics, so the centroids - and every proxy decision
        # after them - changed from run to run.
        import numpy as np

        # All float64 work stays on the host: MPS has no float64 at all.
        inv_np = inv.cpu().numpy()
        w_np = w.cpu().double().numpy()
        tlw = tl.cpu().double().numpy() * w_np[:, None]
        wsum64 = np.bincount(inv_np, weights=w_np, minlength=nb)
        cent64 = np.stack([np.bincount(inv_np, weights=tlw[:, c], minlength=nb) for c in range(3)], 1)
        tl64 = cent64 / np.maximum(wsum64, 1e-12)[:, None]
        w64 = wsum64 / max(wsum64.sum(), 1e-8)
        self.tl = torch.from_numpy(tl64.astype(np.float32)).to(tl.device)
        self.w = torch.from_numpy(w64.astype(np.float32)).to(tl.device)
        self.t2 = (self.tl * self.tl).sum(-1)
        self._tl32 = self.tl.to(torch.float32).contiguous()
        self._w32 = self.w.to(torch.float32).contiguous()
        self._graphs = {}
        self._pool = torch.cuda.graph_pool_handle() if self.tl.is_cuda else None

    def palette_lab(self, dg: torch.Tensor) -> torch.Tensor:
        o = self.opt
        from autoforge.Helper import FusedComposite as fc

        if fc.fused_available(dg):
            pal = fc.stack_palettes(o.material_colors[dg], o.material_TDs[dg], o.background, o.h)
        else:
            pal = batched_stack_palette(
                o.material_colors[dg], o.material_TDs[dg], o.background, o.h
            )
        return srgb_to_lab(pal * 255.0)

    def __call__(self, dg: torch.Tensor) -> torch.Tensor:
        """dg [B,L] long -> [B] proxy loss.

        Launch-bound (a few hundred tiny kernels whatever B is), so on CUDA
        each batch shape is captured into a graph after a few eager calls
        and replayed from then on - the replay runs the same kernels.
        """
        if not dg.is_cuda or torch.cuda.is_current_stream_capturing():
            return self._eval(dg)
        key = tuple(dg.shape)
        entry = self._graphs.setdefault(key, {"calls": 0, "graph": None, "threads": set()})
        if entry["graph"] is None:
            entry["calls"] += 1
            # Only capture in a thread that has run this eagerly (per-thread
            # library handles must exist before capture; see _replay_captured).
            tid = threading.get_ident()
            if entry["calls"] <= 3 or tid not in entry["threads"]:
                entry["threads"].add(tid)
                return self._eval(dg)
            static_in = dg.clone()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, pool=self._pool, capture_error_mode="thread_local"):
                static_out = self._eval(static_in)
            entry.update(graph=graph, inp=static_in, out=static_out)
        entry["inp"].copy_(dg)
        entry["graph"].replay()
        return entry["out"].clone()

    def _eval(self, dg: torch.Tensor) -> torch.Tensor:
        # The palette is a chain of small launch-bound ops: one call for the
        # whole batch. Only the [b, points, heights] distances are chunked.
        pal = self.palette_lab(dg)  # [B,n,3]
        from autoforge.Helper import FusedComposite as fc

        if fc.fused_available(pal):
            return fc.proxy_losses(pal, self._tl32, self._w32)
        p2 = (pal * pal).sum(-1)  # [B,n]
        out = []
        for lo in range(0, dg.shape[0], self.chunk):
            p = pal[lo : lo + self.chunk]
            d2 = torch.baddbmm(
                p2[lo : lo + self.chunk].unsqueeze(1) + self.t2.view(1, -1, 1),
                self.tl.unsqueeze(0).expand(p.shape[0], -1, -1),
                p.transpose(1, 2),
                alpha=-2.0,
            )  # [b,N,n]
            out.append((d2.min(dim=2)[0].clamp(min=0) * self.w).sum(-1) / 3.0)
        return torch.cat(out)


def _layer_colors(dg: torch.Tensor, base) -> int:
    """Filaments of stack ``dg`` besides the base filament."""
    mats = set(torch.unique(dg).tolist())
    mats.discard(base)
    return len(mats)


def _stack_swaps(dg: torch.Tensor, base) -> int:
    """Filament changes from the base up (the base is band 0)."""
    first = int(dg[0]) != (base if base is not None else -1)
    return int((dg[1:] != dg[:-1]).sum()) + int(first)


def _within_limits(c: torch.Tensor, max_colors: int, max_swaps: int, base=None, num_materials: int = None) -> torch.Tensor:
    """[B] bool: stacks c [B,L] within the colour / swap limits (colours
    besides the ``base`` filament, which layers may reuse at no cost).
    ``num_materials`` (an upper bound on the material indices) avoids a host
    sync for the colour count."""
    ok = torch.ones(c.shape[0], dtype=torch.bool, device=c.device)
    if max_swaps < 10**9:
        # The base is band 0: a first layer in another filament is a swap
        # (always, for a base colour that is no filament).
        first = c[:, 0] != (base if base is not None else -1)
        ok &= (c[:, 1:] != c[:, :-1]).sum(1) + first.long() <= max_swaps
    if max_colors < 10**9:
        width = int(num_materials) if num_materials is not None else int(c.max()) + 1
        onehot = torch.zeros(c.shape[0], width, dtype=torch.bool, device=c.device)
        onehot.scatter_(1, c, True)
        if base is not None and base < onehot.shape[1]:
            onehot[:, base] = False
        ok &= onehot.sum(1) <= max_colors
    return ok


class _LayerStep:
    """One coordinate-descent step of ``search_stack`` (``fn(layer)``, the
    layer a 0-dim device tensor) replayed from a CUDA graph: the step is ~25
    tiny launch-bound kernels and runs thousands of times, while a replay
    is one launch plus the layer index. The first call runs eagerly (Triton
    compiles and library handles must not happen inside a capture), the
    second captures; a captured graph only records, so capturing never
    changes the state - the step is then replayed for real. Off CUDA, or
    while another capture is running, every call is eager."""

    def __init__(self, fn, device, pool=None):
        self.fn = fn
        self.layer = torch.zeros((), dtype=torch.long, device=device)
        self.graph = None
        self.calls = 0
        self.pool = pool
        self.use_graph = device.type == "cuda" and not torch.cuda.is_current_stream_capturing()

    def __call__(self, layer: int) -> None:
        self.layer.fill_(layer)
        if not self.use_graph:
            self.fn(self.layer)
            return
        if self.graph is None:
            self.calls += 1
            if self.calls == 1:
                self.fn(self.layer)
                return
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g, pool=self.pool, capture_error_mode="thread_local"):
                self.fn(self.layer)
            self.graph = g
        self.graph.replay()


def search_stack(
    optimizer,
    max_colors: int = 10**9,
    max_swaps: int = 10**9,
    rounds: int = 60,
    batch: int = 16,
    patience: int = 15,
    seed: int = 0,
    init_dg: torch.Tensor = None,
    proxy: "PaletteProxy" = None,
    verbose: bool = True,
    progress=None,
    history: list = None,
) -> torch.Tensor:
    """Greedy stack search under the palette proxy with a large neighbourhood.

    Starts with per-layer coordinate descent, then repeatedly draws a batch
    of random moves - set one layer, fill a segment with one material,
    insert a layer (shifting the ones above up) or delete one (shifting
    down) - and takes the best if it improves.
    """
    if init_dg is None:
        init_dg, _ = optimizer.get_discretized_solution(best=True)
    base = getattr(optimizer, "base_material", None)
    dg = init_dg
    if proxy is None:
        proxy = PaletteProxy(optimizer)
    M = optimizer.material_colors.shape[0]
    cur = dg.clone().to(torch.long)
    L = int(cur.shape[0])
    dev = cur.device
    acc = _decision_dtype(dev)
    g = torch.Generator(device="cpu").manual_seed(seed)
    with torch.no_grad():
        # The accept/reject decisions stay on the GPU (float64, as the
        # Python-float comparison they replace), with one host sync per
        # sweep instead of one per layer.
        best_t = proxy(cur[None])[0].to(acc)
        start = float(best_t)
        # The steps below update these buffers in place, so they can be
        # replayed from graphs (see _LayerStep); same ops as a plain loop.
        pool = torch.cuda.graph_pool_handle() if dev.type == "cuda" else None
        ar_m = torch.arange(M, device=dev)
        improved = torch.zeros((), dtype=torch.bool, device=dev)

        def cd_step(layer_t):
            """All materials for one layer, the best taken if it improves."""
            c = cur.repeat(M, 1)
            c.scatter_(1, layer_t.view(1, 1).expand(M, 1), ar_m.view(M, 1))
            ok = _within_limits(c, max_colors, max_swaps, base, M)
            losses = torch.where(ok, proxy._eval(c), torch.full((M,), float("inf"), device=dev))
            # index_select, not losses[i]: a 0-dim index tensor is read on
            # the host (a sync per step, and no graph capture).
            i = torch.argmin(losses).view(1)
            li = losses.index_select(0, i)[0].to(acc)
            take = li < best_t - 1e-6
            best_t.copy_(torch.where(take, li, best_t))
            cur.copy_(torch.where(take, c.index_select(0, i)[0], cur))
            improved.logical_or_(take)

        step = _LayerStep(cd_step, dev, pool)
        # Coordinate descent: all materials for one layer per batch.
        # Progress: the descent is the first fifth, the rounds the rest (an
        # early stop jumps to the end).
        for sweep_i in range(12):
            if progress is not None:
                progress(0.2 * sweep_i / 12)
            improved.zero_()
            for layer in range(L - 1, -1, -1):
                step(layer)
            if not bool(improved):
                break
        best = float(best_t)
        cd = best
        cur = cur.clone()

        ds = {}  # descend buffers and step, per batch size

        def descend(c, lo, hi, sweeps=3):
            """Parallel coordinate descent of stacks c [B,L] over layers lo..hi."""
            B = c.shape[0]
            if B not in ds:
                st = {
                    "c": torch.empty_like(c),
                    "cl": torch.empty(B, dtype=torch.float32, device=dev),
                    "moved": torch.zeros((), dtype=torch.bool, device=dev),
                }
                ar_b = ar_m.repeat(B).view(-1, 1)
                rows = torch.arange(B, device=dev)

                def d_step(layer_t, st=st, ar_b=ar_b, rows=rows):
                    v = st["c"].repeat_interleave(M, 0)
                    v.scatter_(1, layer_t.view(1, 1).expand(B * M, 1), ar_b)
                    ok = _within_limits(v, max_colors, max_swaps, base, M)
                    lv = torch.where(ok, proxy._eval(v), torch.full((v.shape[0],), float("inf"), device=dev))
                    lv, j = lv.view(B, M).min(dim=1)
                    better = lv < st["cl"] - 1e-6
                    st["moved"].logical_or_(better.any())
                    st["c"].copy_(torch.where(better[:, None], v.view(B, M, L)[rows, j], st["c"]))
                    st["cl"].copy_(torch.where(better, lv, st["cl"]))

                st["step"] = _LayerStep(d_step, dev, pool)
                ds[B] = st
            st = ds[B]
            # A random move can break the colour/swap limits by itself (a
            # segment fill adds a colour, an insert adds swaps); such a start
            # scores inf, so only stacks within the limits can ever win.
            st["c"].copy_(c)
            st["cl"].copy_(torch.where(
                _within_limits(c, max_colors, max_swaps, base, M),
                proxy(c),
                torch.full((B,), float("inf"), device=dev),
            ))
            for _ in range(sweeps):
                st["moved"].zero_()
                for layer in range(hi - 1, lo - 1, -1):
                    st["step"](layer)
                if not bool(st["moved"]):
                    break
            return st["c"].clone(), st["cl"].clone()

        stall = 0
        for _r in range(rounds):
            if progress is not None:
                progress(0.2 + 0.8 * _r / max(rounds, 1))
            cands = []
            a = int(torch.randint(0, L, (1,), generator=g))
            lo, hi = max(0, a - 6), min(L, a + 7)
            for _b in range(batch):
                c = cur.clone()
                kind = int(torch.randint(0, 3, (1,), generator=g))
                m = int(torch.randint(0, M, (1,), generator=g))
                b0 = int(torch.randint(lo, hi, (1,), generator=g))
                if kind == 0:
                    ln = int(torch.randint(1, 7, (1,), generator=g))
                    c[b0 : b0 + ln] = m
                elif kind == 1:  # insert m at b0, shift up
                    c[b0 + 1 :] = cur[b0:-1].clone()
                    c[b0] = m
                else:  # delete b0, shift down, repeat top
                    c[b0:-1] = cur[b0 + 1 :].clone()
                cands.append(c)
            c, cl = descend(torch.stack(cands), lo, hi)
            i = int(torch.argmin(cl))
            if float(cl[i]) < best - 1e-6:
                best, cur = float(cl[i]), c[i].clone()
                if history is not None:
                    history.append(cur)
                stall = 0
            else:
                stall += 1
                if stall >= patience:
                    break
    if verbose:
        print(f"Stack search: proxy loss {start:.4f} -> {cd:.4f} (descent) -> {best:.4f}")
    if not bool(_within_limits(cur[None], max_colors, max_swaps, base)[0]):
        # Never hand back a stack beyond the limits (the start itself may
        # be one, when the limits are tighter than the solution).
        return init_dg
    return cur



def palette_search_stack(
    optimizer,
    dg: torch.Tensor,
    max_colors: int = 10**9,
    max_swaps: int = 10**9,
    rounds: int = 4,
    sweeps: int = 12,
) -> torch.Tensor:
    """Global palette moves on stack ``dg`` under the palette proxy: replace
    one material by another everywhere (a move the layer-wise search cannot
    make under a tight colour limit, the half-way stack has one colour too
    many), each followed by a layer-wise descent. Returns the new stack."""
    base = getattr(optimizer, "base_material", None)
    proxy = PaletteProxy(optimizer)
    M = optimizer.material_colors.shape[0]
    cur = dg.clone().to(torch.long)
    L = int(cur.shape[0])
    dev = cur.device
    acc = _decision_dtype(dev)
    inf = float("inf")
    ar = torch.arange(M, device=dev)
    with torch.no_grad():
        best_t = proxy(cur[None])[0].to(acc)

        def descend(cur, best_t):
            for _ in range(sweeps):
                improved = torch.zeros((), dtype=torch.bool, device=dev)
                for layer in range(L - 1, -1, -1):
                    c = cur.repeat(M, 1)
                    c[:, layer] = ar
                    ok = _within_limits(c, max_colors, max_swaps, base, M)
                    losses = torch.where(ok, proxy(c), torch.full((M,), inf, device=dev))
                    i = torch.argmin(losses)
                    li = losses[i].to(acc)
                    take = li < best_t - 1e-6
                    best_t = torch.where(take, li, best_t)
                    cur = torch.where(take, c[i], cur)
                    improved |= take
                if not bool(improved):
                    break
            return cur, best_t

        start = float(best_t)
        for _ in range(rounds):
            cands = [
                torch.where(cur == a, torch.full_like(cur, m), cur)
                for a in torch.unique(cur).tolist()
                for m in range(M)
                if m != a
            ]
            c = torch.stack(cands)
            ok = _within_limits(c, max_colors, max_swaps, base, M)
            losses = torch.where(ok, proxy(c), torch.full((c.shape[0],), inf, device=dev))
            i = torch.argmin(losses)
            if not float(losses[i]) < float(best_t) - 1e-6:
                break
            cur, best_t = descend(c[i], losses[i].to(acc))
    print(f"Palette search: proxy loss {start:.4f} -> {float(best_t):.4f}")
    return cur


def _plateau_classes(labels: "np.ndarray", n: int):
    """Greedy colouring of the plateaus (labels 1..n) so that plateaus of one
    colour are >= 3 pixels apart (their 3x3 influence regions never overlap).
    Returns an int array [n+1] of colour ids.

    The adjacency is gathered with NumPy (each distinct ordered pair of
    labels found within two pixels once, as the set of pairs it replaces),
    so the degrees - and with them the colouring order - are unchanged."""
    import numpy as np

    H, W = labels.shape
    K = n + 1
    codes = []
    for dy in range(-2, 3):
        for dx in range(-2, 3):
            if dy < 0 or (dy == 0 and dx <= 0):
                continue
            a = labels[: H - dy, max(0, -dx) : W - max(0, dx)]
            b = labels[dy:, max(0, dx) : W - max(0, -dx)]
            m = a != b
            codes.append(a[m].astype(np.int64) * K + b[m])
    codes = np.unique(np.concatenate(codes)) if codes else np.zeros(0, dtype=np.int64)
    A, B = codes // K, codes % K
    deg = np.bincount(A, minlength=K) + np.bincount(B, minlength=K)
    # Neighbours of every label (both directions), as flat Python lists.
    src = np.concatenate([A, B])
    dst = np.concatenate([B, A])
    order = np.argsort(src, kind="stable")
    flat = dst[order].tolist()
    ptr = np.concatenate([[0], np.cumsum(np.bincount(src, minlength=K))]).tolist()
    color = [-1] * K
    # Most neighbours first (ties by label, as the stable sort it replaces).
    for v in (np.argsort(-deg[1:], kind="stable") + 1).tolist():
        used = {color[u] for u in flat[ptr[v]:ptr[v + 1]]}
        c = 0
        while c in used:
            c += 1
        color[v] = c
    return np.asarray(color, dtype=np.int64)


def _label_equal_regions(values: torch.Tensor):
    """4-connected regions of equal value, numbered 1..n by (value, raster
    index of their first pixel) - exactly what labelling ``values == v`` for
    every value v in ascending order with scipy.ndimage.label and offsetting
    the labels gives - computed on the device: every pixel takes the lowest
    index of its region by propagation with pointer jumping. Returns
    (labels int64 [H,W] numpy, n)."""
    H, W = values.shape
    N = H * W
    dev = values.device
    v = values.reshape(H, W)
    idx = torch.arange(N, device=dev, dtype=torch.int64).view(H, W)
    lab = idx.clone()
    eq_d = v[1:, :] == v[:-1, :]
    eq_r = v[:, 1:] == v[:, :-1]
    big = torch.iinfo(torch.int64).max
    while True:
        new = lab.clone()
        new[1:, :] = torch.minimum(new[1:, :], torch.where(eq_d, lab[:-1, :], big))
        new[:-1, :] = torch.minimum(new[:-1, :], torch.where(eq_d, lab[1:, :], big))
        new[:, 1:] = torch.minimum(new[:, 1:], torch.where(eq_r, lab[:, :-1], big))
        new[:, :-1] = torch.minimum(new[:, :-1], torch.where(eq_r, lab[:, 1:], big))
        flat = new.view(-1)
        for _ in range(4):
            flat = flat[flat]
        new = flat.view(H, W)
        if torch.equal(new, lab):
            break
        lab = new
    roots = torch.nonzero(lab.view(-1) == idx.view(-1)).view(-1)  # one per region, raster order
    rv = v.reshape(-1)[roots]
    rank = torch.argsort(rv, stable=True)  # by value, raster order within a value
    ids = torch.empty(N, dtype=torch.int64, device=dev)
    ids[roots[rank]] = torch.arange(1, roots.numel() + 1, device=dev)
    return ids[lab.view(-1)].view(H, W).cpu().numpy(), int(roots.numel())


@composite_graph_scope()
def refine_plateaus(optimizer, min_size: int = 2, shifts=(-3, -2, -1, 1, 2, 3), spike_aware: bool = False, mode: str = "height", cell: float = 6.0) -> bool:
    """Move whole connected equal-height plateaus at once (region moves).

    Single-pixel coordinate descent cannot shift a plateau: every pixel
    alone pays the height-jump penalty to its neighbours. A plateau moved as
    one only changes its boundary, so those moves are judged here. Each
    plateau tries a shift of its height (``shifts``) and the height whose
    flat stack colour is nearest its pixels; plateaus of a colour class are
    >= 3 pixels apart, so every one is judged on its own footprint through
    the real composite. Kept only if the real loss improved.
    """
    import numpy as np

    from autoforge.Helper.PruningHelper import _compute_loss_for_heightmap, disc_to_logits

    dg, z = optimizer.get_discretized_solution(best=True)
    if z is None:
        return False
    pre_loss = _compute_loss_for_heightmap(optimizer, dg)
    L = int(optimizer.max_layers)
    disc_logits = disc_to_logits(dg, optimizer.material_colors.shape[0], big_pos=1e5)
    z = z.to(torch.int64).clone()
    H, W = z.shape
    dev = z.device
    target_lab = srgb_to_lab(optimizer.target)
    weights = _pixel_weights(optimizer, (H, W))
    smooth = float(getattr(optimizer.args, "pixel_height_smoothness", 1.0))
    spike_thr = float(getattr(optimizer.args, "spike_threshold_layers", 1))
    kk = torch.arange(L + 1, device=dev).view(1, 1, -1)
    layer = torch.arange(L, device=dev).view(-1, 1, 1)
    pal = srgb_to_lab(_stack_palette(optimizer, dg.to(torch.long), (layer < kk).float() * optimizer.h))
    dist = torch.cdist(target_lab.reshape(-1, 3), pal).pow(2) * weights.reshape(-1, 1)  # [HW,L+1]

    err_map = _heights_error_fn(optimizer, dg, disc_logits, L, target_lab, weights, smooth)

    # Plateaus: connected components of equal height (4-connectivity).
    zc = z
    if mode == "target":
        # Regions of similar target colour (same Lab grid cell, connected).
        zc = torch.round(target_lab / cell).to(torch.int64)
        zc = (zc[..., 0] * 1009 + zc[..., 1]) * 1009 + zc[..., 2]
    labels, n = _label_equal_regions(zc)
    movable = _movable(optimizer, (H, W))
    sizes = np.bincount(labels.ravel(), minlength=n + 1)
    big = sizes >= min_size
    big[0] = False
    if not big.any():
        return False
    color = _plateau_classes(np.where(big[labels], labels, 0), n)
    color = np.where(big, color, -1)
    lab_t = torch.from_numpy(labels).to(dev)
    color_t = torch.from_numpy(color).to(dev)
    # Region optimum height per plateau.
    cost = torch.zeros(n + 1, L + 1, device=dev).index_add_(0, lab_t.reshape(-1), dist)
    region_opt = cost.argmin(dim=1)  # [n+1]
    cnt = torch.zeros(n + 1, device=dev).index_add_(0, lab_t.reshape(-1), torch.ones(H * W, device=dev))
    region_mean = (torch.zeros(n + 1, device=dev).index_add_(0, lab_t.reshape(-1), z.reshape(-1).float()) / cnt.clamp(min=1)).round().long()

    with torch.no_grad():
        for c in range(int(color.max()) + 1):
            in_class = (color_t[lab_t] == c) & movable  # [H,W] pixels of plateaus in this class
            if not bool(in_class.any()):
                continue
            fp_lab = F.max_pool2d(torch.where(in_class, lab_t, torch.zeros_like(lab_t)).float()[None, None], 3, 1, 1)[0, 0].long()
            base = torch.zeros(n + 1, device=dev).index_add_(0, fp_lab.reshape(-1), err_map(z).reshape(-1))
            best_err = base.clone()
            bvm = torch.full_like(z, -1)
            cands = [(z + s).clamp(0, L) for s in shifts] + [region_opt[lab_t]]
            if mode == "target":
                cands.append(region_mean[lab_t])
            for cv in cands:
                trial = torch.where(in_class, cv, z)
                err = torch.zeros(n + 1, device=dev).index_add_(0, fp_lab.reshape(-1), err_map(trial).reshape(-1))
                better = err < best_err - 1e-9
                best_err = torch.where(better, err, best_err)
                bvm = torch.where(better[lab_t] & in_class, cv, bvm)
            z_new = torch.where(bvm >= 0, bvm, z)
            if spike_aware:
                z_new = _drop_new_spikes(z, z_new, spike_thr)
            z = z_new

    new_logits = optimizer._remove_height_offset(
        pixel_logits=_heights_to_eff_logits(z, L), height_offsets=optimizer.best_params["height_offsets"]
    )
    kept, post_loss = _keep_heights_if_better(optimizer, new_logits, dg, pre_loss)
    print(f"Plateau refine ({mode}): {n} plateaus, loss {pre_loss:.4f} -> {post_loss:.4f} | {'kept' if kept else 'reverted'}")
    return kept
