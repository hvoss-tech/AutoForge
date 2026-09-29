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
import os
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
    return torch.from_numpy(
        _viterbi_swaps_np(score.detach().to("cpu", torch.float64).numpy(), max_swaps, base)
    ).to(dev)


def _viterbi_swaps_np(sc: np.ndarray, max_swaps: int, base: Optional[int] = None) -> np.ndarray:
    """``_viterbi_swaps`` on a NumPy [B, L, M] score, returns [B, L] int64."""
    # A NaN score makes every comparison false, which backtracked into a
    # swap count of -1.
    sc = np.nan_to_num(sc.astype(np.float64, copy=False), nan=-1e9, posinf=1e9, neginf=-1e9)
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
    return out


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
    # The best in-palette score of a layer without material m is the runner-up
    # where m is the layer's best and the best elsewhere, so the top two per
    # layer score every candidate drop at once (a loop over the materials
    # was thousands of tiny kernels per call).
    mask = torch.ones((B, M), dtype=torch.bool, device=dev)
    n_other = M - (1 if free is not None else 0)
    rows = torch.arange(B, device=dev)
    ar = torch.arange(M, device=dev)
    droppable = ar != free if free is not None else torch.ones(M, dtype=torch.bool, device=dev)
    neg_t = torch.tensor(neg, device=dev)
    for _ in range(max(0, n_other - max_colors)):
        v = torch.where(mask.unsqueeze(1), score, neg_t)  # [B,L,M]
        top_v, top_i = v.topk(2, dim=-1)  # [B,L,2]
        without = torch.where(
            top_i[..., :1] == ar, top_v[..., 1:], top_v[..., :1]
        )  # [B,L,M]: best in-palette score with m dropped
        c = without.sum(1)  # [B,M]
        c = torch.where(mask & droppable, c, torch.full_like(c, neg))
        drop = c.argmax(1)
        mask[rows, drop] = False
    return mask


def _palette_masks_greedy_np(sc: np.ndarray, max_colors: int, free: Optional[int] = None) -> np.ndarray:
    """``_palette_masks``' greedy branch on a NumPy [B, L, M] score (float64):
    [B, M] bool. Same rule - drop the material whose removal costs least -
    with the cost of every drop from the top two scores per layer."""
    B, L, M = sc.shape
    neg = np.finfo(np.float32).min / 4
    mask = np.ones((B, M), dtype=bool)
    droppable = np.ones(M, dtype=bool)
    if free is not None and 0 <= free < M:
        droppable[free] = False
    n_other = M - (1 if free is not None else 0)
    rows = np.arange(B)
    bl = np.repeat(rows, L)
    for _ in range(max(0, n_other - max_colors)):
        v = np.where(mask[:, None, :], sc, neg)  # [B,L,M]
        top2 = np.partition(v, M - 2, axis=-1)[..., M - 2:]  # [B,L,2]: runner-up, best
        v1, v2 = top2[..., 1], top2[..., 0]
        i1 = v.argmax(-1)  # [B,L]
        # Best in-palette score summed over the layers with m dropped: the
        # full sum, minus what the layers whose best is m lose to their
        # runner-up.
        cost = np.zeros((B, M))
        np.add.at(cost, (bl, i1.reshape(-1)), (v1 - v2).reshape(-1))
        c = v1.sum(1)[:, None] - cost
        c = np.where(mask & droppable, c, neg)
        mask[rows, c.argmax(1)] = False
    return mask


try:
    import triton
    import triton.language as tl

    from autoforge.Helper import FusedComposite as _fc

    _HAS_TRITON = _fc._HAS_TRITON
except Exception:  # pragma: no cover - triton missing
    _HAS_TRITON = False

if _HAS_TRITON:

    @triton.jit
    def _project_kernel(
        sc_ptr, out_ptr, bs_ptr, bf_ptr, scr_ptr, L, M, n_drop, S, pal_free, vit_base,
        PALETTE: tl.constexpr, SWAPS: tl.constexpr, HAS_BASE: tl.constexpr,
        BL: tl.constexpr, BM: tl.constexpr, BS: tl.constexpr,
    ):
        """project_assignment for one stack (program b) of [B, L, M] scores:
        the greedy palette, then the swap-limited Viterbi DP (or the
        per-layer argmax) - the NumPy path's rules, in float32 (float64 runs
        at 1/64 speed on consumer GPUs; near-ties may resolve differently)."""
        b = tl.program_id(0)
        rl = tl.arange(0, BL)
        rm = tl.arange(0, BM)
        lv = rl < L
        mv = rm < M
        NEG = -8.507058665963222e37  # float32 min / 4, as the NumPy path
        base = sc_ptr + b * L * M
        sc = tl.load(base + rl[:, None] * M + rm[None, :], mask=lv[:, None] & mv[None, :], other=0.0)
        allowed = mv
        if PALETTE:
            droppable = mv & (rm != pal_free)
            for _ in range(n_drop):
                v = tl.where(allowed[None, :], sc, NEG)
                v1 = tl.max(v, axis=1)
                i1 = tl.argmax(v, axis=1)
                v2 = tl.max(tl.where(rm[None, :] == i1[:, None], NEG, v), axis=1)
                gain = tl.where(lv, v1 - v2, 0.0)
                cost = tl.sum(tl.where((rm[None, :] == i1[:, None]) & lv[:, None], gain[:, None], 0.0), axis=0)
                c = tl.sum(tl.where(lv, v1, 0.0), axis=0) - cost
                c = tl.where(allowed & droppable, c, NEG)
                drop = tl.argmax(c, axis=0)
                allowed = allowed & (rm != drop)
            sc = tl.where(allowed[None, :], sc, -1e9)
        if not SWAPS:
            best = tl.argmax(tl.where(mv[None, :], sc, float("-inf")), axis=1)
            tl.store(out_ptr + b * L + rl, best.to(tl.int64), mask=lv)
        else:
            rs = tl.arange(0, BS)
            sv = rs < S
            ninf = float("-inf")
            row0 = tl.sum(tl.where(rl[:, None] == 0, sc, 0.0), axis=0)  # [BM]
            dp = tl.full([BS, BM], ninf, tl.float32)
            if HAS_BASE:
                in_range = (vit_base >= 0) & (vit_base < M)
                dp = tl.where((rs[:, None] == 0) & (rm[None, :] == vit_base) & in_range, row0[None, :], dp)
                dp = tl.where((rs[:, None] == 1) & (rm[None, :] != vit_base) & mv[None, :] & (S > 1), row0[None, :], dp)
            else:
                dp = tl.where((rs[:, None] == 0) & mv[None, :], row0[None, :], dp)
            dp = tl.where(sv[:, None] & mv[None, :], dp, ninf)
            bsb = bs_ptr + b * L * BS * BM
            bfb = bf_ptr + b * L * BS * BM
            for l in range(1, L):
                srow = tl.load(base + l * M + rm, mask=mv, other=0.0)
                if PALETTE:
                    srow = tl.where(allowed, srow, -1e9)
                i1 = tl.argmax(dp, axis=1)  # [BS]
                v1 = tl.max(dp, axis=1)
                masked = tl.where(rm[None, :] == i1[:, None], ninf, dp)
                i2 = tl.argmax(masked, axis=1)
                v2 = tl.max(masked, axis=1)
                is_best = i1[:, None] == rm[None, :]
                other_v = tl.where(is_best, v2[:, None], v1[:, None])  # [BS, BM]
                other_i = tl.where(is_best, i2[:, None], i1[:, None])
                # one swap more: row s takes row s - 1 (through a scratch row)
                so = scr_ptr + b * 2 * BS * BM + rs[:, None] * BM + rm[None, :]
                tl.store(so, other_v)
                tl.store(so + BS * BM, other_i.to(tl.float32))
                tl.debug_barrier()
                up = (rs[:, None] >= 1) & (rm[None, :] >= 0)
                switch_v = tl.load(so - BM, mask=up, other=ninf)
                switch_i = tl.load(so + BS * BM - BM, mask=up, other=0.0).to(tl.int32)
                tl.debug_barrier()
                stay = (dp >= switch_v) | (rs[:, None] == 0)
                dp = tl.where(stay, dp, switch_v) + srow[None, :]
                dp = tl.where(sv[:, None] & mv[None, :], dp, ninf)
                o = (l - 1) * BS * BM + rs[:, None] * BM + rm[None, :]
                tl.store(bsb + o, stay.to(tl.int8))
                tl.store(bfb + o, switch_i.to(tl.int32))
            # The back-pointers were written by all threads: sync before
            # one of them walks them.
            tl.debug_barrier()
            # argmax over (s, m), the first in row-major order: the first row
            # holding the maximum, then its first column holding it.
            s_idx = tl.argmax(tl.max(dp, axis=1), axis=0)
            m_idx = tl.argmax(tl.max(tl.where(rs[:, None] == s_idx, dp, ninf), axis=0), axis=0)
            tl.store(out_ptr + b * L + L - 1, m_idx.to(tl.int64))
            for j in range(0, L - 1):
                l = L - 1 - j
                o = (l - 1) * BS * BM + s_idx * BM + m_idx
                st = tl.load(bsb + o)
                prev = tl.where(st != 0, m_idx, tl.load(bfb + o))
                s_idx = tl.where(st != 0, s_idx, s_idx - 1)
                m_idx = prev
                tl.store(out_ptr + b * L + l - 1, m_idx.to(tl.int64))


def _project_gpu(score: torch.Tensor, max_colors, max_swaps, free, need_palette, need_swaps):
    """project_assignment's greedy path in one kernel launch, [B, L] int64."""
    B, L, M = score.shape
    mat = _mat(free)
    sc = torch.nan_to_num(score.detach().to(torch.float32), nan=-1e9, posinf=1e9, neginf=-1e9).contiguous()
    n_other = M - (1 if mat is not None else 0)
    S = (max_swaps + 1) if need_swaps else 1
    BL = max(16, triton.next_power_of_2(L))
    BM = max(16, triton.next_power_of_2(M))
    BS = max(2, triton.next_power_of_2(S))
    out = torch.empty(B, L, dtype=torch.int64, device=score.device)
    if need_swaps:
        bs = torch.empty(B, L, BS, BM, dtype=torch.int8, device=score.device)
        bf = torch.empty(B, L, BS, BM, dtype=torch.int32, device=score.device)
        scr = torch.empty(B, 2, BS, BM, dtype=torch.float32, device=score.device)
    else:
        bs = bf = scr = out
    _project_kernel[(B,)](
        sc, out, bs, bf, scr, L, M, max(0, n_other - max_colors) if need_palette else 0, S,
        mat if mat is not None else -1, free if free is not None else -1,
        PALETTE=need_palette, SWAPS=need_swaps, HAS_BASE=free is not None,
        BL=BL, BM=BM, BS=BS, num_warps=4,
    )
    return out


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
    need_palette = max_colors is not None and max_colors < M - (1 if mat is not None else 0)
    need_swaps = max_swaps is not None and max_swaps < score.shape[1] - (1 if free is None else 0)
    if (
        need_palette and _palette_candidates(M, max_colors, mat, "cpu") is None
        and _HAS_TRITON and score.is_cuda and (not need_swaps or max_swaps + 1 <= 64)
        and os.environ.get("AF_PROJECT_GPU", "1") == "1"
    ):
        # Greedy palette and swap DP in one kernel (one program per stack).
        out = _project_gpu(score, max_colors, max_swaps, free, need_palette, need_swaps)
        return out[0] if squeeze else out
    if need_palette and _palette_candidates(M, max_colors, mat, "cpu") is None:
        # Greedy palette and swap DP both on the host, from one copy: as
        # small GPU kernels they were ~1000 launches and a sync per call.
        dev = score.device
        sc = score.detach().to("cpu", torch.float64).numpy()
        mask = _palette_masks_greedy_np(sc, max_colors, mat)
        sc = np.where(mask[:, None, :], sc, -1e9)
        out_np = _viterbi_swaps_np(sc, max_swaps, free) if need_swaps else sc.argmax(-1)
        out = torch.from_numpy(out_np).to(dev)
        return out[0] if squeeze else out
    if need_palette:
        mask = _palette_masks(score, max_colors, mat)
        score = torch.where(
            mask.unsqueeze(1), score, torch.tensor(-1e9, device=score.device)
        )
    if need_swaps:
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
    palette material everywhere by any other material. In that order, each
    stack once (its first occurrence), the stack itself left out.

    Built as one array (a candidate list per move was ~10 ms of Python and a
    host-to-device copy per candidate); returns views of one device tensor."""
    cur = np.asarray(dg.tolist(), dtype=np.int64)
    L = cur.shape[0]
    mat = _mat(free)
    M = int(n_mat)
    groups = []
    starts = np.flatnonzero(np.r_[True, cur[1:] != cur[:-1]])
    ends = np.r_[starts[1:], L]  # exclusive
    ms = np.arange(M)
    # Recolour a band.
    for a, b in zip(starts, ends):
        c = int(cur[a])
        g = np.repeat(cur[None, :], M, 0)
        g[:, a:b] = ms[:, None]
        groups.append(g[ms != c])
    # Shift a band boundary by 1-2 layers.
    for k in range(len(starts) - 1):
        a0, b0 = int(starts[k]), int(ends[k]) - 1
        a1, b1 = int(starts[k + 1]), int(ends[k + 1]) - 1
        c0, c1 = int(cur[a0]), int(cur[a1])
        for d in (1, 2):
            if b0 - d + 1 >= a0:  # upper band grows downward
                g = cur.copy()
                g[b0 - d + 1 : b0 + 1] = c1
                groups.append(g[None])
            if a1 + d - 1 <= b1:  # lower band grows upward
                g = cur.copy()
                g[a1 : a1 + d] = c0
                groups.append(g[None])
    # Recolour one layer (within the palette, the base filament included,
    # under a colour limit).
    palette = np.array(sorted(set(cur.tolist()) | ({mat} if mat is not None else set())), dtype=np.int64)
    choices = palette if max_colors is not None else ms
    g = np.repeat(np.repeat(cur[None, :], len(choices), 0)[None], L, 0)  # [L, K, L]
    g[np.arange(L), :, np.arange(L)] = choices[None, :]
    keep = choices[None, :] != cur[:, None]  # [L, K]
    groups.append(g[keep])
    # Replace one material everywhere (a new one, or merge into another).
    for p in sorted(set(cur.tolist())):
        g = np.repeat(cur[None, :], M, 0)
        g[:, cur == p] = ms[:, None]
        groups.append(g[ms != p])
    cand = np.concatenate([cur[None]] + groups, 0) if groups else cur[None]
    # First occurrence of each stack, in order; row 0 is the stack itself.
    first = {}
    for i, row in enumerate(cand):
        first.setdefault(row.tobytes(), i)
    idx = np.fromiter(first.values(), dtype=np.int64, count=len(first))
    cand = cand[idx[idx != 0]]
    swaps = (cand[:, 1:] != cand[:, :-1]).sum(1)
    if free is not None:
        swaps = swaps + (cand[:, 0] != free)  # from the base to the first layer
    ok = np.ones(cand.shape[0], dtype=bool)
    if max_swaps is not None:
        ok &= swaps <= max_swaps
    if max_colors is not None:
        present = np.zeros((cand.shape[0], max(M, int(cand.max()) + 1)), dtype=bool)
        present[np.arange(cand.shape[0])[:, None], cand] = True
        if mat is not None:
            present[:, mat] = False
        ok &= present.sum(1) <= max_colors
    cand = cand[ok]
    if cand.shape[0] == 0:
        return []
    return list(torch.from_numpy(cand).to(device=dg.device, dtype=dg.dtype).unbind(0))


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
