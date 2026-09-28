"""Per-pixel height assignment for a fixed layer stack.

With the layer materials fixed, a pixel printed to height ``k`` shows (up to
the 3x3 edge bleed) the colour of the stack after ``k`` full layers. The best
height of every pixel is therefore an assignment problem over the ``L + 1``
stack colours - the "E-step" the gradient training cannot do itself: it only
moves heights through per-cluster offsets and a smooth coarse field, so a
pixel stays in the colour band its k-means cluster put it in.

The assignment trades the Lab colour error against height jumps to the 8
neighbours (the same cost as the pixel height refine: ``smooth`` per layer of
height difference, diagonals weighted 1/sqrt(2)), minimised by iterated
conditional modes over the four (y mod 2, x mod 2) pixel classes - pixels of
one class are not neighbours, so each class moves at once and no update can
raise the cost.
"""

import torch
import torch.nn.functional as F

from autoforge.Helper.ImageHelper import srgb_to_lab
from autoforge.Helper.OptimizerHelper import composite_image_disc


def heights_to_logits(z: torch.Tensor, max_layers: int) -> torch.Tensor:
    """Height logits that discretize back to exactly ``z`` (layers)."""
    t = (z.to(torch.float32) / float(max_layers)).clamp(1e-6, 1 - 1e-6)
    return torch.log(t) - torch.log1p(-t)


@torch.no_grad()
def stack_colors(
    dg: torch.Tensor,
    max_layers: int,
    h: float,
    material_colors: torch.Tensor,
    material_TDs: torch.Tensor,
    background: torch.Tensor,
) -> torch.Tensor:
    """[L+1, 3] colour (0-255) of a uniform region printed 0..L layers high
    with the discrete stack ``dg``, through the real discrete composite."""
    L = int(max_layers)
    from autoforge.Helper import FusedComposite as fc

    if fc.fused_available(dg):
        # A uniform region's colour is exactly the flat-stack palette.
        d = dg.long()
        return fc.stack_palettes(material_colors[d][None], material_TDs[d][None], background, h)[0] * 255.0
    ks = torch.arange(L + 1, device=dg.device, dtype=torch.float32)
    # One 3x3 block per height; the centre pixel sees only its own height.
    img = heights_to_logits(ks, L).view(-1, 1, 1).expand(L + 1, 3, 3).reshape((L + 1) * 3, 3)
    n_mat = material_colors.shape[0]
    logits = torch.full((L, n_mat), -1e5, device=dg.device, dtype=torch.float32)
    logits.scatter_(1, dg.long().view(-1, 1), 1e5)
    comp = composite_image_disc(
        img.contiguous(), logits, 0.01, 0.01, h, L, material_colors, material_TDs, background, rng_seed=0
    )
    return comp.view(L + 1, 3, 3, 3)[:, 1, 1, :]


def _neighbour_cost(z: torch.Tensor, k: torch.Tensor, smooth: float) -> torch.Tensor:
    """[H, W, K] summed weighted |k - z_neighbour| over the 8 neighbours."""
    H, W = z.shape
    pad = F.pad(z.to(torch.float32)[None, None], (1, 1, 1, 1), mode="replicate")[0, 0]
    out = torch.zeros(H, W, k.shape[0], device=z.device, dtype=torch.float32)
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dy == 0 and dx == 0:
                continue
            w = 1.0 if dy == 0 or dx == 0 else 0.7071
            nb = pad[1 + dy : 1 + dy + H, 1 + dx : 1 + dx + W]
            out += w * (k.view(1, 1, -1) - nb.unsqueeze(-1)).abs()
    return out * smooth


@torch.no_grad()
def assign_heights(
    colors: torch.Tensor,
    target: torch.Tensor,
    z0,
    smooth: float = 2.0,
    sweeps: int = 6,
    weights: torch.Tensor = None,
    target_lab: torch.Tensor = None,
) -> torch.Tensor:
    """ICM height assignment. ``colors`` [K,3] (0-255) are the stack colours
    per height, ``target`` [H,W,3] (0-255), ``z0`` [H,W] the starting
    heights (None: the colour-only optimum). Returns int64 [H,W] heights in
    0..K-1.

    Both sides of every edge are charged (``2 * smooth``), as when the
    per-pixel costs are summed over each pixel's 3x3 window in the refine."""
    H, W = target.shape[:2]
    K = colors.shape[0]
    c_lab = srgb_to_lab(colors.view(1, K, 3).float()).view(K, 3)
    if target_lab is None:
        target_lab = srgb_to_lab(target.float())
    err = torch.cdist(target_lab.view(-1, 3), c_lab).pow(2).view(H, W, K)  # Lab SSE
    if weights is not None:
        err = err * weights.unsqueeze(-1)
    if z0 is None or smooth <= 0:
        z = err.argmin(-1)
        if smooth <= 0:
            return z
    else:
        z = z0.to(torch.int64).clamp(0, K - 1).clone()
    s = 2.0 * smooth
    ks = torch.arange(K, device=z.device, dtype=torch.float32)
    # Neighbour offsets and weights (diagonals 1/sqrt(2)).
    offs = [(dy, dx) for dy in (-1, 0, 1) for dx in (-1, 0, 1) if dy or dx]
    nw = torch.tensor([1.0 if dy == 0 or dx == 0 else 0.7071 for dy, dx in offs], device=z.device)
    w_sum = float(nw.sum())
    # The four classes' error slices, taken once.
    classes = [(a, b, err[a::2, b::2].contiguous()) for a in range(2) for b in range(2)]
    from autoforge.Helper import FusedComposite as fc

    if fc.fused_available(z) and z.is_contiguous():
        # One kernel per class update (same cost, evaluated per height).
        for _ in range(sweeps):
            before = z.clone()
            for a, b, err_c in classes:
                fc.icm_class_update(err_c, z, a, b, s)
            if torch.equal(before, z):
                break
        return z
    for _ in range(sweeps):
        before = z.clone()
        for a, b, err_c in classes:
            h, w = err_c.shape[:2]
            pad = F.pad(z[None, None].float(), (1, 1, 1, 1), mode="replicate")[0, 0]
            # [h, w, 8] neighbour heights of the class's pixels.
            nb = torch.stack(
                [pad[1 + a + dy : 1 + a + dy + 2 * h : 2, 1 + b + dx : 1 + b + dx + 2 * w : 2] for dy, dx in offs],
                dim=-1,
            )
            # sum_q w_q |k - z_q| = k (2 A(k) - W) - 2 B(k) + Z, with A/B the
            # cumulative neighbour weight / weighted height up to k.
            idx = nb.long()
            A = torch.zeros(h, w, K, device=z.device).scatter_add_(2, idx, nw.expand(h, w, -1))
            B = torch.zeros(h, w, K, device=z.device).scatter_add_(2, idx, nb * nw)
            A = A.cumsum(2)
            B = B.cumsum(2)
            Z = (nb * nw).sum(-1, keepdim=True)
            cost = err_c + s * (ks * (2.0 * A - w_sum) - 2.0 * B + Z)
            z[a::2, b::2] = cost.argmin(-1)
        if torch.equal(before, z):
            break
    return z
