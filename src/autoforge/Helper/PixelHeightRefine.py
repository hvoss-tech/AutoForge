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
    H, W = z.shape
    pad = F.pad(z.float()[None, None], (1, 1, 1, 1), mode="replicate")
    return F.unfold(pad, 3)[0].median(dim=0)[0].view(H, W).to(z.dtype)


def spike_mask(z: torch.Tensor, threshold: float, max_outliers: int = 2) -> torch.Tensor:
    """Pixels the first pass of ``remove_height_spikes`` would flag."""
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


@composite_graph_scope()
def refine_pixel_heights(
    optimizer,
    sweeps: int = 2,
    radius: int = -1,
    spike_aware: bool = False,
    block: int = 1,
) -> bool:
    """Coordinate descent over per-pixel heights (see module docstring).

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
    smooth = float(getattr(optimizer.args, "pixel_height_smoothness", 2.0))
    # Pull toward the height map pruning handed over (quadratic, so it is
    # the large departures that cost), stored on the first refine of a prune.
    anchor_w = float(getattr(optimizer.args, "pixel_height_anchor", 0.0))
    z_anchor = getattr(optimizer, "_refine_anchor_z", None)
    if z_anchor is None or z_anchor.shape != z.shape:
        z_anchor = z.clone()
        optimizer._refine_anchor_z = z_anchor
    z_anchor_f = z_anchor.to(torch.float32)
    spike_thr = float(getattr(optimizer.args, "spike_threshold_layers", 1))

    def window_err(zmap):
        """Error summed over the influence region of the block anchored at
        each pixel (rows/cols -1 .. block)."""
        comp = composite_image_disc(
            _heights_to_eff_logits(zmap, L),
            disc_logits,
            optimizer.vis_tau,
            optimizer.vis_tau,
            optimizer.h,
            L,
            optimizer.material_colors,
            optimizer.material_TDs,
            optimizer.background,
            rng_seed=optimizer.best_seed,
            compute_dtype=optimizer.composite_compute_dtype,
        )
        e = (srgb_to_lab(comp) - target_lab).pow(2).sum(-1) * weights
        if smooth > 0:
            e = e + smooth * _height_variation(zmap)
        if anchor_w > 0:
            e = e + anchor_w * (zmap.to(torch.float32) - z_anchor_f).pow(2)
        e = F.pad(e[None, None], (1, block, 1, block))
        return F.conv2d(e, kernel)[0, 0]

    yy = torch.arange(H, device=z.device).view(-1, 1)
    xx = torch.arange(W, device=z.device).view(1, -1)
    classes = []
    for a in range(stride):
        for b in range(stride):
            oy, ox = (yy - a) % stride, (xx - b) % stride
            ay, ax = yy - oy, xx - ox
            anchor = (oy == 0) & (ox == 0)
            member = (oy < block) & (ox < block) & (ay >= 0) & (ax >= 0)
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

    with torch.no_grad():
        for _ in range(sweeps):
            changed = 0
            for anchor, member, ay, ax in classes:
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
    old_best = optimizer.best_params["pixel_height_logits"]
    old_live = optimizer.pixel_height_logits
    optimizer.best_params["pixel_height_logits"] = new_logits
    optimizer.pixel_height_logits = new_logits
    post_loss = _compute_loss_for_heightmap(optimizer, dg)
    kept = post_loss < pre_loss
    if not kept:
        optimizer.best_params["pixel_height_logits"] = old_best
        optimizer.pixel_height_logits = old_live
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
) -> bool:
    """Coordinate descent over the material of each layer, heights fixed.

    Every layer in turn tries every material; a candidate is only allowed if
    the solution stays within ``max_colors`` distinct materials and
    ``max_swaps`` swaps. Scored with the shared-thickness batched path and
    verified with the real metric at the end (reverted if not better).
    """
    from autoforge.Helper.PruningHelper import (
        _compute_loss_for_heightmap,
        _eval_candidates_batch,
        _make_shared_eff_thick,
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
        eff = _make_shared_eff_thick(optimizer)
        best_dg = dg.clone()
        best, _ = _eval_candidates_batch(optimizer, [best_dg], eff_thick=eff)
        for _ in range(sweeps):
            improved = False
            for layer in range(L - 1, -1, -1):
                cands = []
                for m in range(num_materials):
                    if m == int(best_dg[layer]):
                        continue
                    c = best_dg.clone()
                    c[layer] = m
                    if int(torch.unique(c).numel()) > max_colors:
                        continue
                    if len(find_color_bands(c)) - 1 > max_swaps:
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
) -> torch.Tensor:
    """Coordinate descent over layer materials under ``palette_loss_fn``.
    Returns the new stack (does not touch the optimizer's solution)."""
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
                for m in range(num_materials):
                    if m == int(best_dg[layer]):
                        continue
                    c = best_dg.clone()
                    c[layer] = m
                    if max_colors < 10**9 and int(torch.unique(c).numel()) > max_colors:
                        continue
                    if max_swaps < 10**9 and len(find_color_bands(c)) - 1 > max_swaps:
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


def apply_stack(optimizer, dg: torch.Tensor, refine_sweeps: int = 2, radius: int = -1) -> bool:
    """Swap in stack ``dg``, re-solve the heights per pixel, and keep the
    result only if the real loss beats the current solution."""
    from autoforge.Helper.PruningHelper import disc_to_logits

    before = optimizer.solution_loss()
    snap = optimizer.solution_snapshot()
    optimizer.best_params["global_logits"] = disc_to_logits(
        dg, optimizer.material_colors.shape[0], big_pos=1e5
    )
    refine_pixel_heights(optimizer, sweeps=refine_sweeps, radius=radius)
    after = optimizer.solution_loss()
    if after is None or before is None or after >= before:
        optimizer.restore_solution_snapshot(snap)
        print(f"Stack swap: {before:.4f} -> {after:.4f} | reverted")
        return False
    print(f"Stack swap: {before:.4f} -> {after:.4f} | kept")
    return True


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
        wsum = torch.zeros(nb, device=tl.device).index_add_(0, inv, w)
        cent = torch.zeros(nb, 3, device=tl.device).index_add_(0, inv, tl * w[:, None])
        self.tl = cent / wsum.clamp(min=1e-12)[:, None]
        self.w = wsum / wsum.sum().clamp(min=1e-8)
        self.t2 = (self.tl * self.tl).sum(-1)
        self._graphs = {}
        self._pool = torch.cuda.graph_pool_handle() if self.tl.is_cuda else None

    def palette_lab(self, dg: torch.Tensor) -> torch.Tensor:
        o = self.opt
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


def _within_limits(c: torch.Tensor, max_colors: int, max_swaps: int) -> torch.Tensor:
    """[B] bool: stacks c [B,L] within the colour / swap limits."""
    ok = torch.ones(c.shape[0], dtype=torch.bool, device=c.device)
    if max_swaps < 10**9:
        ok &= (c[:, 1:] != c[:, :-1]).sum(1) <= max_swaps
    if max_colors < 10**9:
        onehot = torch.zeros(c.shape[0], int(c.max()) + 1, dtype=torch.bool, device=c.device)
        onehot.scatter_(1, c, True)
        ok &= onehot.sum(1) <= max_colors
    return ok


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
) -> torch.Tensor:
    """Greedy stack search under the palette proxy with a large neighbourhood.

    Starts with per-layer coordinate descent, then repeatedly draws a batch
    of random moves - set one layer, fill a segment with one material,
    insert a layer (shifting the ones above up) or delete one (shifting
    down) - and takes the best if it improves.
    """
    if init_dg is None:
        init_dg, _ = optimizer.get_discretized_solution(best=True)
    dg = init_dg
    if proxy is None:
        proxy = PaletteProxy(optimizer)
    M = optimizer.material_colors.shape[0]
    cur = dg.clone().to(torch.long)
    L = int(cur.shape[0])
    dev = cur.device
    g = torch.Generator(device="cpu").manual_seed(seed)
    with torch.no_grad():
        # The accept/reject decisions stay on the GPU (float64, as the
        # Python-float comparison they replace), with one host sync per
        # sweep instead of one per layer.
        best_t = proxy(cur[None])[0].double()
        start = float(best_t)
        # Coordinate descent: all materials for one layer per batch.
        for _ in range(12):
            improved = torch.zeros((), dtype=torch.bool, device=dev)
            for layer in range(L - 1, -1, -1):
                c = cur.repeat(M, 1)
                c[:, layer] = torch.arange(M, device=dev)
                ok = _within_limits(c, max_colors, max_swaps)
                losses = torch.where(ok, proxy(c), torch.full((M,), float("inf"), device=dev))
                i = torch.argmin(losses)
                li = losses[i].double()
                take = li < best_t - 1e-6
                best_t = torch.where(take, li, best_t)
                cur = torch.where(take, c[i], cur)
                improved |= take
            if not bool(improved):
                break
        best = float(best_t)
        cd = best

        def descend(c, lo, hi, sweeps=3):
            """Parallel coordinate descent of stacks c [B,L] over layers lo..hi."""
            B = c.shape[0]
            cl = proxy(c)
            ar = torch.arange(M, device=dev)
            rows = torch.arange(B, device=dev)
            for _ in range(sweeps):
                moved = torch.zeros((), dtype=torch.bool, device=dev)
                for layer in range(hi - 1, lo - 1, -1):
                    v = c.repeat_interleave(M, 0)
                    v[:, layer] = ar.repeat(B)
                    ok = _within_limits(v, max_colors, max_swaps)
                    lv = torch.where(ok, proxy(v), torch.full((v.shape[0],), float("inf"), device=dev))
                    lv, j = lv.view(B, M).min(dim=1)
                    better = lv < cl - 1e-6
                    moved |= better.any()
                    c = torch.where(better[:, None], v.view(B, M, L)[rows, j], c)
                    cl = torch.where(better, lv, cl)
                if not bool(moved):
                    break
            return c, cl

        stall = 0
        for _r in range(rounds):
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
                stall = 0
            else:
                stall += 1
                if stall >= patience:
                    break
    if verbose:
        print(f"Stack search: proxy loss {start:.4f} -> {cd:.4f} (descent) -> {best:.4f}")
    return cur

