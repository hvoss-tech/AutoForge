"""Colour / swap limits held by the optimizer itself (--constrained_opt).

``project_assignment`` maps per-layer material scores [L, M] to the best
discrete layer stack that uses at most ``max_colors`` distinct materials
besides the base filament (``free``: layers may reuse it at no cost) and
at most ``max_swaps`` material changes between consecutive layers, where
"best" means the highest summed per-layer score. With log-probabilities as
scores that is the most likely feasible stack; with Gumbel-perturbed
log-probabilities it is a random feasible sample (perturb-and-MAP), which is
what the discrete snapshots during training use.
"""
from itertools import combinations
from typing import Optional

import numpy as np
import torch

# Enumerate every palette up to this many subsets, greedy search beyond.
MAX_SUBSETS = 5000


def _mat(base: Optional[int]) -> Optional[int]:
    """The base's filament index (None when there is no base filament)."""
    return base if base is not None and base >= 0 else None


def _viterbi_swaps(score: torch.Tensor, max_swaps: int, base: Optional[int] = None) -> torch.Tensor:
    """argmax over stacks with <= max_swaps changes of sum_l score[b, l, dg[l]].

    With ``base`` (the base's filament, -1 for a base color that is no
    filament) the change from the base to the first layer counts too: the
    base is the stack's band 0.

    score: [B, L, M] (use a large negative score for forbidden materials).
    Returns [B, L] int64 on score's device.

    Runs in NumPy on the CPU: the tables are tiny ([B, max_swaps+1, M] per
    layer), and as a loop of small GPU kernels it took ~18ms per call with
    the GPU idle in between - the bulk of the constrained snapshot cost.
    """
    dev = score.device
    # A NaN score makes every comparison false, which backtracked into a
    # swap count of -1.
    sc = np.nan_to_num(
        score.detach().to("cpu", torch.float64).numpy(), nan=-1e9, posinf=1e9, neginf=-1e9
    )
    B, L, M = sc.shape
    S = max_swaps + 1
    neg = -1e300
    dp = np.full((B, S, M), neg)
    if base is None:
        dp[:, 0] = sc[:, 0]
    else:
        # Staying in the base filament is free, anything else is a swap.
        if 0 <= base < M:
            dp[:, 0, base] = sc[:, 0, base]
        if S > 1:
            other = np.arange(M) != base
            dp[:, 1, other] = sc[:, 0, other]
    back_stay = np.empty((max(L - 1, 0), B, S, M), dtype=bool)
    back_from = np.empty((max(L - 1, 0), B, S, M), dtype=np.int64)
    ar = np.arange(M)
    for l in range(1, L):
        # Best previous material other than m, one swap earlier.
        i1 = dp.argmax(axis=2)  # [B,S]
        v1 = np.take_along_axis(dp, i1[..., None], axis=2)[..., 0]
        masked = dp.copy()
        np.put_along_axis(masked, i1[..., None], neg, axis=2)
        i2 = masked.argmax(axis=2)
        v2 = np.take_along_axis(masked, i2[..., None], axis=2)[..., 0]
        is_best = i1[..., None] == ar
        other_v = np.where(is_best, v2[..., None], v1[..., None])  # [B,S,M]
        other_i = np.where(is_best, i2[..., None], i1[..., None])
        switch_v = np.full_like(dp, neg)
        switch_v[:, 1:] = other_v[:, :-1]
        switch_i = np.zeros_like(other_i)
        switch_i[:, 1:] = other_i[:, :-1]
        stay = dp >= switch_v
        stay[:, 0] = True  # no swap left to come from
        dp = np.where(stay, dp, switch_v) + sc[:, l][:, None, :]
        back_stay[l - 1] = stay
        back_from[l - 1] = switch_i
    flat = dp.reshape(B, -1).argmax(axis=1)
    s_idx, m_idx = flat // M, flat % M
    out = np.empty((B, L), dtype=np.int64)
    out[:, L - 1] = m_idx
    bi = np.arange(B)
    for l in range(L - 1, 0, -1):
        st = back_stay[l - 1][bi, s_idx, m_idx]
        prev = np.where(st, m_idx, back_from[l - 1][bi, s_idx, m_idx])
        s_idx = np.where(st, s_idx, s_idx - 1)
        m_idx = prev
        out[:, l - 1] = m_idx
    return torch.from_numpy(out).to(dev)


def _palette_candidates(M: int, max_colors: int, free: Optional[int], device) -> Optional[torch.Tensor]:
    """All palettes as [N, M] bool: ``max_colors`` materials besides
    ``free`` (the base filament, always in the palette at no cost), or None
    when there are too many to enumerate."""
    others = [m for m in range(M) if m != free]
    k = min(max_colors, len(others))
    n_sub = 1
    for i in range(k):
        n_sub = n_sub * (len(others) - i) // (i + 1)
    if n_sub > MAX_SUBSETS:
        return None
    subs = torch.tensor(list(combinations(others, k)), device=device, dtype=torch.long).view(-1, k)
    masks = torch.zeros((subs.shape[0], M), dtype=torch.bool, device=device)
    masks.scatter_(1, subs, True)
    if free is not None:
        masks[:, free] = True
    return masks


def top_palettes(score: torch.Tensor, max_colors: int, k: int, free: Optional[int] = None) -> torch.Tensor:
    """The k best palettes for scores [L, M] by sum_l max_{m in palette}
    score[l, m], as [k', M] bool (k' <= k; exhaustive search only)."""
    L, M = score.shape
    free = _mat(free)
    masks = _palette_candidates(M, max_colors, free, score.device)
    if masks is None:
        return _palette_masks(score.unsqueeze(0), max_colors, free)
    neg = torch.tensor(torch.finfo(score.dtype).min / 4, device=score.device)
    vals = torch.where(masks.unsqueeze(1), score.unsqueeze(0), neg).amax(-1).sum(-1)
    return masks[vals.topk(min(k, vals.shape[0])).indices]


def _palette_masks(score: torch.Tensor, max_colors: int, free: Optional[int] = None) -> torch.Tensor:
    """Best palette per batch item, as [B, M] bool: at most ``max_colors``
    materials besides ``free`` (the base filament, which is always in it),
    judged by sum_l max_{m in palette} score[b, l, m]."""
    B, L, M = score.shape
    dev = score.device
    neg = torch.finfo(score.dtype).min / 4
    masks = _palette_candidates(M, max_colors, free, dev)
    if masks is not None:
        # [B, N, L]: best in-palette score per layer
        vals = torch.where(
            masks.view(1, -1, 1, M), score.unsqueeze(1), torch.tensor(neg, device=dev)
        ).amax(dim=-1)
        best = vals.sum(-1).argmax(dim=1)  # [B]
        return masks[best]
    # Greedy: drop the material whose removal costs least, until small enough.
    mask = torch.ones((B, M), dtype=torch.bool, device=dev)
    n_other = M - (1 if free is not None else 0)
    for _ in range(max(0, n_other - max_colors)):
        costs = []
        for m in range(M):
            trial = mask.clone()
            trial[:, m] = False
            v = torch.where(trial.unsqueeze(1), score, torch.tensor(neg, device=dev))
            c = v.amax(-1).sum(-1)
            keep = mask[:, m] & (m != free)
            c = torch.where(keep, c, torch.full_like(c, neg))
            costs.append(c)
        drop = torch.stack(costs, 1).argmax(1)
        mask[torch.arange(B, device=dev), drop] = False
    return mask


def project_assignment(
    score: torch.Tensor,
    max_colors: Optional[int],
    max_swaps: Optional[int],
    free: Optional[int] = None,
) -> torch.Tensor:
    """Best feasible stack for scores [L, M] or [B, L, M] -> [L] / [B, L].

    ``free`` is the base: its filament index (a layer in it adds no colour
    to the print, and a first layer in it is no swap), -1 for a base colour
    that is no filament (a colour of its own, and the first layer is always
    a swap), or None to leave the base out entirely. ``max_colors`` limits
    the materials besides the base filament."""
    squeeze = score.dim() == 2
    if squeeze:
        score = score.unsqueeze(0)
    score = score.float()
    M = score.shape[-1]
    mat = _mat(free)
    if max_colors is not None and max_colors < M - (1 if mat is not None else 0):
        mask = _palette_masks(score, max_colors, mat)
        score = torch.where(
            mask.unsqueeze(1), score, torch.tensor(-1e9, device=score.device)
        )
    if max_swaps is not None and max_swaps < score.shape[1] - (1 if free is None else 0):
        out = _viterbi_swaps(score, max_swaps, free)
    else:
        out = score.argmax(-1)
    return out[0] if squeeze else out


def feasible(
    dg: torch.Tensor, max_colors: Optional[int], max_swaps: Optional[int], free: Optional[int] = None
) -> bool:
    c, s = count_colors_swaps(dg, free)
    return (max_colors is None or c <= max_colors) and (max_swaps is None or s <= max_swaps)


def neighbour_stacks(
    dg: torch.Tensor,
    n_mat: int,
    max_colors: Optional[int],
    max_swaps: Optional[int],
    free: Optional[int] = None,
) -> list:
    """Feasible single-move neighbours of the stack ``dg`` [L]: recolour a
    band, shift a band boundary by 1-2 layers, recolour one layer, replace one
    palette material everywhere by any other material."""
    L = dg.shape[0]
    mat = _mat(free)
    cur = dg.tolist()
    seen = {tuple(cur)}
    out = []

    def add(cand):
        t = tuple(cand)
        if t in seen:
            return
        seen.add(t)
        colors = len(set(cand) - {mat})
        swaps = sum(1 for a, b in zip(cand[:-1], cand[1:]) if a != b)
        if free is not None and cand[0] != free:
            swaps += 1  # from the base to the first layer
        if (max_colors is None or colors <= max_colors) and (
            max_swaps is None or swaps <= max_swaps
        ):
            out.append(cand)

    bands = []
    start = 0
    for i in range(1, L + 1):
        if i == L or cur[i] != cur[start]:
            bands.append((start, i - 1, cur[start]))
            start = i
    for a, b, c in bands:
        for m in range(n_mat):
            if m != c:
                add(cur[:a] + [m] * (b - a + 1) + cur[b + 1:])
    for (a0, b0, c0), (a1, b1, c1) in zip(bands[:-1], bands[1:]):
        for d in (1, 2):
            if b0 - d + 1 >= a0:  # upper band grows downward
                add(cur[: b0 - d + 1] + [c1] * d + cur[b0 + 1:])
            if a1 + d - 1 <= b1:  # lower band grows upward
                add(cur[:a1] + [c0] * d + cur[a1 + d:])
    # The base filament is always available to a layer, at no colour cost.
    palette = sorted(set(cur) | ({mat} if mat is not None else set()))
    for l in range(L):
        for m in palette if max_colors is not None else range(n_mat):
            if m != cur[l]:
                add(cur[:l] + [m] + cur[l + 1:])
    for p in sorted(set(cur)):
        for m in range(n_mat):
            if m != p:  # replace (m outside the palette) or merge (m inside, the base included)
                add([m if v == p else v for v in cur])
    return [torch.tensor(c, dtype=dg.dtype, device=dg.device) for c in out]


def count_colors_swaps(dg: torch.Tensor, free: Optional[int] = None) -> tuple[int, int]:
    """(materials in the stack besides the base filament, swaps) - ``free``
    as in ``project_assignment``. With a base the print's colours are the
    first number plus one, and the swaps include the change from the base to
    the first layer."""
    mats = set(torch.unique(dg).tolist())
    mats.discard(_mat(free))
    swaps = int((dg[1:] != dg[:-1]).sum().item())
    if free is not None and int(dg[0]) != free:
        swaps += 1
    return len(mats), swaps


def base_material_index(background: torch.Tensor, material_colors: torch.Tensor) -> Optional[int]:
    """Index of the filament the base is printed in, or None when the base
    color is not one of the filaments (a custom color: it then takes a
    color slot of its own). Matched by color, which covers the automatically
    picked base as well as a filament chosen by hand."""
    d = (material_colors.float() - background.float().view(1, -1)).abs().amax(dim=1)
    i = int(torch.argmin(d))
    return i if float(d[i]) < 0.5 / 255 else None
