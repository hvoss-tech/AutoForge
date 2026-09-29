"""Fused discrete composite for a fixed layer stack and integer heights.

``composite_image_disc`` materialises several [L,H,W] tensors per call (print
mask, bleed, run thickness, coverage, opacity, transmittance). With the layer
stack fixed and the heights already whole layers, every pixel's colour only
needs its own height and its 8 neighbours' (the 3x3 bleed), so one kernel
walks the stack bottom to top per pixel in registers: same model, no [L,H,W]
memory. Computed in fp32 throughout (the torch path rounds its [L,H,W]
intermediates to the autocast dtype), so results agree to rounding, not bit
for bit - callers verify with the real metric before keeping anything.
"""

import os

import torch

from autoforge.Helper.OptimizerHelper import material_run_starts, run_starts

try:
    import triton
    import triton.language as tl

    _HAS_TRITON = True
except Exception:  # pragma: no cover - triton missing
    _HAS_TRITON = False

# AUTOFORGE_TRITON=off takes the plain PyTorch path everywhere, e.g. on a ROCm
# GPU whose Triton backend can't compile these kernels. Only the kernel
# definitions stay skipped; `triton` itself may still be imported above.
if os.environ.get("AUTOFORGE_TRITON", "").strip().lower() in {"off", "0", "false", "no"}:
    _HAS_TRITON = False


if _HAS_TRITON:

    @triton.jit
    def _lab_f(t):
        return tl.where(
            t > 0.008856,
            tl.exp(tl.log(tl.maximum(t, 1e-6)) * (1.0 / 3.0)),
            (903.3 * t + 16.0) / 116.0,
        )

    @triton.jit
    def _srgb_lin(c):
        c = tl.minimum(tl.maximum(c, 0.0), 1.0)
        return tl.where(
            c <= 0.04045,
            c / 12.92,
            tl.exp(tl.log(tl.maximum((c + 0.055) / 1.055, 1e-6)) * 2.4),
        )

    @triton.jit
    def _disc_composite_kernel(
        z_ptr, out_ptr, params_ptr, bg_ptr, H, W, L, h, inv_h,
        tlab_ptr, wt_ptr, smooth, zstride,
        ERR: tl.constexpr, BLOCK: tl.constexpr, LOSS: tl.constexpr = False,
    ):
        pid = tl.program_id(0)
        # Batch of stacks (grid axis 1): stack b reads its own [L,8] table;
        # with LOSS each program writes its block's error sum to
        # out[b, pid] instead of the per-pixel map.
        pb = tl.program_id(1)
        params_ptr = params_ptr + pb * L * 8
        z_ptr = z_ptr + pb * zstride  # per-stack height maps (0: shared)
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        n = H * W
        valid = offs < n
        y = offs // W
        x = offs % W
        z = tl.load(z_ptr + offs, mask=valid, other=0)
        # Neighbour heights; outside the image counts as never printed (the
        # bleed convolution zero-pads).
        up = y > 0
        dn = y < H - 1
        lf = x > 0
        rt = x < W - 1
        z_u = tl.load(z_ptr + offs - W, mask=valid & up, other=0)
        z_d = tl.load(z_ptr + offs + W, mask=valid & dn, other=0)
        z_l = tl.load(z_ptr + offs - 1, mask=valid & lf, other=0)
        z_r = tl.load(z_ptr + offs + 1, mask=valid & rt, other=0)
        z_ul = tl.load(z_ptr + offs - W - 1, mask=valid & up & lf, other=0)
        z_ur = tl.load(z_ptr + offs - W + 1, mask=valid & up & rt, other=0)
        z_dl = tl.load(z_ptr + offs + W - 1, mask=valid & dn & lf, other=0)
        z_dr = tl.load(z_ptr + offs + W + 1, mask=valid & dn & rt, other=0)

        cr = tl.zeros([BLOCK], dtype=tl.float32) + tl.load(bg_ptr + 0)
        cg = tl.zeros([BLOCK], dtype=tl.float32) + tl.load(bg_ptr + 1)
        cb = tl.zeros([BLOCK], dtype=tl.float32) + tl.load(bg_ptr + 2)
        run_thick = tl.zeros([BLOCK], dtype=tl.float32)
        cov_prev = tl.zeros([BLOCK], dtype=tl.float32)
        # Above the highest of every pixel's and its neighbours' heights no
        # layer is printed or bled into: the colour cannot change there, so
        # the walk stops at the block's maximum (exactly the same result).
        zmax = tl.maximum(tl.maximum(tl.maximum(z, z_u), tl.maximum(z_d, z_l)),
                          tl.maximum(tl.maximum(z_r, z_ul), tl.maximum(z_ur, tl.maximum(z_dl, z_dr))))
        ltop = tl.minimum(tl.max(tl.where(valid, zmax, 0), axis=0), L)
        for l in range(0, ltop):
            # params row: r, g, b, rsqrt(reach), rsqrt(slow_reach), w, continues
            p = params_ptr + l * 8
            lr = tl.load(p + 0)
            lg = tl.load(p + 1)
            lb = tl.load(p + 2)
            ir = tl.load(p + 3)
            isl = tl.load(p + 4)
            w = tl.load(p + 5)
            cont = tl.load(p + 6)
            nb = (
                (l < z_u).to(tl.float32) + (l < z_d).to(tl.float32)
                + (l < z_l).to(tl.float32) + (l < z_r).to(tl.float32)
                + (l < z_ul).to(tl.float32) + (l < z_ur).to(tl.float32)
                + (l < z_dl).to(tl.float32) + (l < z_dr).to(tl.float32)
            )
            m = (l < z).to(tl.float32) + 0.1 * (0.125 * nb)
            eff = tl.minimum(tl.maximum(m, 0.0), 1.0) * h
            run_thick = eff + cont * run_thick
            root = tl.sqrt(tl.maximum(run_thick, 1e-12))
            plain = tl.minimum(root * ir, 1.0)
            slow = tl.minimum(root * isl, 1.0)
            cov = slow + w * (plain - slow)
            cp = cont * cov_prev
            op = (cov - cp) / tl.maximum(1.0 - cp, 1e-6)
            op = tl.minimum(tl.maximum(op, 0.0), 1.0) * (eff * inv_h)
            cr = cr + op * (lr - cr)
            cg = cg + op * (lg - cg)
            cb = cb + op * (lb - cb)
            cov_prev = cov
        if ERR:
            # Weighted Lab squared error (srgb_to_lab) plus the smoothness
            # term of the refine: smooth * sum_q w_q |z - z_q| over the 8
            # neighbours, the border counting as equal height.
            rl = _srgb_lin(cr)
            gl = _srgb_lin(cg)
            bl = _srgb_lin(cb)
            fx = _lab_f((0.4124564 * rl + 0.3575761 * gl + 0.1804375 * bl) / 0.95047)
            fy = _lab_f(0.2126729 * rl + 0.7151522 * gl + 0.0721750 * bl)
            fz = _lab_f((0.0193339 * rl + 0.1191920 * gl + 0.9503041 * bl) / 1.08883)
            dL = 116.0 * fy - 16.0 - tl.load(tlab_ptr + offs * 3 + 0, mask=valid, other=0.0)
            da = 500.0 * (fx - fy) - tl.load(tlab_ptr + offs * 3 + 1, mask=valid, other=0.0)
            db = 200.0 * (fy - fz) - tl.load(tlab_ptr + offs * 3 + 2, mask=valid, other=0.0)
            e = (dL * dL + da * da + db * db) * tl.load(wt_ptr + offs, mask=valid, other=0.0)
            zf = z.to(tl.float32)
            var = (
                tl.where(up, tl.abs(zf - z_u.to(tl.float32)), 0.0)
                + tl.where(dn, tl.abs(zf - z_d.to(tl.float32)), 0.0)
                + tl.where(lf, tl.abs(zf - z_l.to(tl.float32)), 0.0)
                + tl.where(rt, tl.abs(zf - z_r.to(tl.float32)), 0.0)
                + 0.7071 * (
                    tl.where(up & lf, tl.abs(zf - z_ul.to(tl.float32)), 0.0)
                    + tl.where(up & rt, tl.abs(zf - z_ur.to(tl.float32)), 0.0)
                    + tl.where(dn & lf, tl.abs(zf - z_dl.to(tl.float32)), 0.0)
                    + tl.where(dn & rt, tl.abs(zf - z_dr.to(tl.float32)), 0.0)
                )
            )
            if LOSS:
                tl.store(out_ptr + pb * tl.num_programs(0) + pid, tl.sum(tl.where(valid, e, 0.0), axis=0))
            else:
                tl.store(out_ptr + offs, e + smooth * var, mask=valid)
        else:
            o = out_ptr + offs * 3
            tl.store(o + 0, cr * 255.0, mask=valid)
            tl.store(o + 1, cg * 255.0, mask=valid)
            tl.store(o + 2, cb * 255.0, mask=valid)


def stack_params(dg: torch.Tensor, material_colors, material_TDs, background, h: float) -> torch.Tensor:
    """[L,8] fp32 per-layer table for the kernel (see its params row)."""
    dg = dg.to(torch.long)
    cols = material_colors.index_select(0, dg)
    tds = material_TDs.index_select(0, dg).clamp(1e-8, 1e8)
    return layer_params(cols, tds, background, h)


def layer_params(cols: torch.Tensor, tds: torch.Tensor, background, h: float) -> torch.Tensor:
    """``stack_params`` from the per-layer colours [L,3] and TDs [L]."""
    cols = cols.to(torch.float32)
    tds = tds.to(torch.float32)
    run_start = material_run_starts(cols, tds)
    reach, slow_reach, w = coverage_params(cols, tds, run_start, background, h)
    L = cols.shape[0]
    idx = torch.arange(L, device=cols.device)
    p = torch.zeros(L, 8, device=cols.device, dtype=torch.float32)
    p[:, :3] = cols
    p[:, 3] = torch.rsqrt(reach.float())
    p[:, 4] = torch.rsqrt(slow_reach.float())
    p[:, 5] = w.float()
    p[:, 6] = (run_start < idx).float()
    return p


def background_tensor(background: torch.Tensor) -> torch.Tensor:
    """The background colour as the kernels read it: contiguous fp32 [3]."""
    return background.detach().to(torch.float32).contiguous()


def fused_available(t: torch.Tensor) -> bool:
    return _HAS_TRITON and t.is_cuda


def composite_heights(z: torch.Tensor, params: torch.Tensor, bg: torch.Tensor, h: float) -> torch.Tensor:
    """[H,W,3] (0..255) discrete composite of integer heights ``z`` [H,W]
    under the stack described by ``params`` (``stack_params``); ``bg`` is the
    background colour (``background_tensor``)."""
    H, W = z.shape
    zi = z.to(torch.int32).contiguous()
    out = torch.empty(H, W, 3, device=z.device, dtype=torch.float32)
    BLOCK = 256
    grid = (triton.cdiv(H * W, BLOCK),)
    _disc_composite_kernel[grid](
        zi, out, params.contiguous(), bg, H, W, int(params.shape[0]),
        float(h), 1.0 / float(h), out, out, 0.0, 0, ERR=False, BLOCK=BLOCK,
    )
    return out


def heights_error(z, params, bg: torch.Tensor, h: float, target_lab, weights, smooth: float) -> torch.Tensor:
    """[H,W] per-pixel refine error of heights ``z``: weighted Lab squared
    error of the composite against ``target_lab`` plus ``smooth`` times the
    8-neighbour height variation (see PixelHeightRefine._height_variation)."""
    H, W = z.shape
    zi = z.to(torch.int32).contiguous()
    out = torch.empty(H, W, device=z.device, dtype=torch.float32)
    BLOCK = 256
    grid = (triton.cdiv(H * W, BLOCK),)
    _disc_composite_kernel[grid](
        zi, out, params.contiguous(), bg, H, W, int(params.shape[0]),
        float(h), 1.0 / float(h), target_lab.contiguous(), weights.contiguous(), float(smooth), 0,
        ERR=True, BLOCK=BLOCK,
    )
    return out


# --------------------------------------------------------------------------
# Continuous (training) composite: forward and backward kernels.
#
# Same model as ``composite_image_cont`` from the print mask on: soft print
# mask sigmoid((z - l - 0.5) * scale), 3x3 neighbour bleed (zero padded),
# run thickness, coverage, per-layer opacity and the front-to-back blend -
# walked bottom to top per pixel in registers. The backward recomputes the
# walk instead of keeping a tape: a bottom-up pass stores the run thickness
# and g . (colour below) per layer, a top-down pass produces the gradients,
# and a last kernel gathers the print-mask gradient over each pixel's 3x3
# window into dL/dz. Per-layer parameter gradients are reduced per block and
# summed over blocks afterwards.
# --------------------------------------------------------------------------

if _HAS_TRITON:

    @triton.jit
    def _sig(x):
        return 1.0 / (1.0 + tl.exp(-x))

    @triton.jit
    def _nb_sum(z_u, z_d, z_l, z_r, z_ul, z_ur, z_dl, z_dr, v_u, v_d, v_l, v_r, v_ul, v_ur, v_dl, v_dr, t, scale):
        return (
            v_u * _sig((z_u - t) * scale) + v_d * _sig((z_d - t) * scale)
            + v_l * _sig((z_l - t) * scale) + v_r * _sig((z_r - t) * scale)
            + v_ul * _sig((z_ul - t) * scale) + v_ur * _sig((z_ur - t) * scale)
            + v_dl * _sig((z_dl - t) * scale) + v_dr * _sig((z_dr - t) * scale)
        )

    @triton.jit
    def _load_nbrs(z_ptr, offs, valid, y, x, H, W):
        up = y > 0
        dn = y < H - 1
        lf = x > 0
        rt = x < W - 1
        z_u = tl.load(z_ptr + offs - W, mask=valid & up, other=0.0)
        z_d = tl.load(z_ptr + offs + W, mask=valid & dn, other=0.0)
        z_l = tl.load(z_ptr + offs - 1, mask=valid & lf, other=0.0)
        z_r = tl.load(z_ptr + offs + 1, mask=valid & rt, other=0.0)
        z_ul = tl.load(z_ptr + offs - W - 1, mask=valid & up & lf, other=0.0)
        z_ur = tl.load(z_ptr + offs - W + 1, mask=valid & up & rt, other=0.0)
        z_dl = tl.load(z_ptr + offs + W - 1, mask=valid & dn & lf, other=0.0)
        z_dr = tl.load(z_ptr + offs + W + 1, mask=valid & dn & rt, other=0.0)
        return (
            z_u, z_d, z_l, z_r, z_ul, z_ur, z_dl, z_dr,
            up.to(tl.float32), dn.to(tl.float32), lf.to(tl.float32), rt.to(tl.float32),
            (up & lf).to(tl.float32), (up & rt).to(tl.float32),
            (dn & lf).to(tl.float32), (dn & rt).to(tl.float32),
        )

    @triton.jit
    def _cov(rt_, ir, isl, w):
        root = tl.sqrt(tl.maximum(rt_, 1e-12))
        plain = tl.minimum(root * ir, 1.0)
        slow = tl.minimum(root * isl, 1.0)
        return slow + w * (plain - slow)

    @triton.jit
    def _cont_fwd_kernel(
        z_ptr, out_ptr, prm_ptr, bg_ptr, rt_ptr, s_ptr, g_ptr,
        H, W, L, h, inv_h, scale,
        STORE: tl.constexpr, BLOCK: tl.constexpr,
    ):
        """Forward walk. With STORE it is the backward's first pass instead:
        writes run thickness and g . (colour below) per layer, no output."""
        pid = tl.program_id(0)
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        N = H * W
        valid = offs < N
        y = offs // W
        x = offs % W
        z = tl.load(z_ptr + offs, mask=valid, other=0.0)
        (z_u, z_d, z_l, z_r, z_ul, z_ur, z_dl, z_dr,
         v_u, v_d, v_l, v_r, v_ul, v_ur, v_dl, v_dr) = _load_nbrs(z_ptr, offs, valid, y, x, H, W)
        cr = tl.zeros([BLOCK], dtype=tl.float32) + tl.load(bg_ptr + 0)
        cg = tl.zeros([BLOCK], dtype=tl.float32) + tl.load(bg_ptr + 1)
        cb = tl.zeros([BLOCK], dtype=tl.float32) + tl.load(bg_ptr + 2)
        if STORE:
            gr = tl.load(g_ptr + offs * 3 + 0, mask=valid, other=0.0)
            gg = tl.load(g_ptr + offs * 3 + 1, mask=valid, other=0.0)
            gb = tl.load(g_ptr + offs * 3 + 2, mask=valid, other=0.0)
        run_thick = tl.zeros([BLOCK], dtype=tl.float32)
        cov_prev = tl.zeros([BLOCK], dtype=tl.float32)
        for l in range(0, L):
            p = prm_ptr + l * 8
            lr = tl.load(p + 0)
            lg = tl.load(p + 1)
            lb = tl.load(p + 2)
            ir = tl.load(p + 3)
            isl = tl.load(p + 4)
            w = tl.load(p + 5)
            cont = tl.load(p + 6)
            t = l + 0.5
            m = _sig((z - t) * scale) + 0.0125 * _nb_sum(
                z_u, z_d, z_l, z_r, z_ul, z_ur, z_dl, z_dr,
                v_u, v_d, v_l, v_r, v_ul, v_ur, v_dl, v_dr, t, scale)
            eff = tl.minimum(tl.maximum(m, 0.0), 1.0) * h
            run_thick = eff + cont * run_thick
            cov = _cov(run_thick, ir, isl, w)
            cp = cont * cov_prev
            op = tl.minimum(tl.maximum((cov - cp) / tl.maximum(1.0 - cp, 1e-6), 0.0), 1.0) * (eff * inv_h)
            if STORE:
                tl.store(rt_ptr + l * N + offs, run_thick, mask=valid)
                tl.store(s_ptr + l * N + offs, gr * cr + gg * cg + gb * cb, mask=valid)
            cr = cr + op * (lr - cr)
            cg = cg + op * (lg - cg)
            cb = cb + op * (lb - cb)
            cov_prev = cov
        if not STORE:
            o = out_ptr + offs * 3
            tl.store(o + 0, cr * 255.0, mask=valid)
            tl.store(o + 1, cg * 255.0, mask=valid)
            tl.store(o + 2, cb * 255.0, mask=valid)

    @triton.jit
    def _cont_bwd_kernel(
        z_ptr, g_ptr, prm_ptr, rt_ptr, s_ptr, red_ptr,
        H, W, L, h, inv_h, scale,
        BLOCK: tl.constexpr,
    ):
        """Top-down pass: parameter gradients (per-block partial sums into red
        [blocks,L+1,8], summed afterwards - deterministic, unlike atomics) and
        the print-mask gradient per layer, written over s."""
        pid = tl.program_id(0)
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        N = H * W
        valid = offs < N
        vf = valid.to(tl.float32)
        y = offs // W
        x = offs % W
        z = tl.load(z_ptr + offs, mask=valid, other=0.0)
        (z_u, z_d, z_l, z_r, z_ul, z_ur, z_dl, z_dr,
         v_u, v_d, v_l, v_r, v_ul, v_ur, v_dl, v_dr) = _load_nbrs(z_ptr, offs, valid, y, x, H, W)
        gr = tl.load(g_ptr + offs * 3 + 0, mask=valid, other=0.0)
        gg = tl.load(g_ptr + offs * 3 + 1, mask=valid, other=0.0)
        gb = tl.load(g_ptr + offs * 3 + 2, mask=valid, other=0.0)
        T = vf
        g_rt_carry = tl.zeros([BLOCK], dtype=tl.float32)
        g_cov_carry = tl.zeros([BLOCK], dtype=tl.float32)
        for li in range(0, L):
            l = L - 1 - li
            p = prm_ptr + l * 8
            lr = tl.load(p + 0)
            lg = tl.load(p + 1)
            lb = tl.load(p + 2)
            ir = tl.load(p + 3)
            isl = tl.load(p + 4)
            w = tl.load(p + 5)
            cont = tl.load(p + 6)
            lp = tl.maximum(l - 1, 0)
            pp = prm_ptr + lp * 8
            ir_p = tl.load(pp + 3)
            isl_p = tl.load(pp + 4)
            w_p = tl.load(pp + 5)
            rt_l = tl.load(rt_ptr + l * N + offs, mask=valid, other=0.0)
            rt_p = tl.load(rt_ptr + lp * N + offs, mask=valid & (l > 0), other=0.0)
            s = tl.load(s_ptr + l * N + offs, mask=valid, other=0.0)
            t = l + 0.5
            m = _sig((z - t) * scale) + 0.0125 * _nb_sum(
                z_u, z_d, z_l, z_r, z_ul, z_ur, z_dl, z_dr,
                v_u, v_d, v_l, v_r, v_ul, v_ur, v_dl, v_dr, t, scale)
            eff = tl.minimum(tl.maximum(m, 0.0), 1.0) * h
            root = tl.sqrt(tl.maximum(rt_l, 1e-12))
            plain_raw = root * ir
            slow_raw = root * isl
            plain = tl.minimum(plain_raw, 1.0)
            slow = tl.minimum(slow_raw, 1.0)
            cov = slow + w * (plain - slow)
            cov_p = _cov(rt_p, ir_p, isl_p, w_p)
            cp = cont * cov_p
            den_raw = 1.0 - cp
            den = tl.maximum(den_raw, 1e-6)
            q = (cov - cp) / den
            ratio = tl.minimum(tl.maximum(q, 0.0), 1.0)
            op = ratio * (eff * inv_h)

            gop = T * (gr * lr + gg * lg + gb * lb - s)
            tw = T * op
            tl.store(red_ptr + pid * (L + 1) * 8 + l * 8 + 0, tl.sum(gr * tw, axis=0))
            tl.store(red_ptr + pid * (L + 1) * 8 + l * 8 + 1, tl.sum(gg * tw, axis=0))
            tl.store(red_ptr + pid * (L + 1) * 8 + l * 8 + 2, tl.sum(gb * tw, axis=0))
            T = T * (1.0 - op)

            g_ratio = gop * eff * inv_h
            g_eff = gop * ratio * inv_h
            gq = tl.where((q >= 0.0) & (q <= 1.0), g_ratio, 0.0)
            g_cov = gq / den + g_cov_carry
            g_cp = gq * (-1.0 / den + tl.where(den_raw >= 1e-6, (cov - cp) / (den * den), 0.0))
            g_cov_carry = cont * g_cp
            tl.store(red_ptr + pid * (L + 1) * 8 + l * 8 + 5, tl.sum(g_cov * (plain - slow), axis=0))
            gp = tl.where(plain_raw <= 1.0, g_cov * w, 0.0)
            gs = tl.where(slow_raw <= 1.0, g_cov * (1.0 - w), 0.0)
            tl.store(red_ptr + pid * (L + 1) * 8 + l * 8 + 3, tl.sum(gp * root, axis=0))
            tl.store(red_ptr + pid * (L + 1) * 8 + l * 8 + 4, tl.sum(gs * root, axis=0))
            g_root = gp * ir + gs * isl
            g_rt = tl.where(rt_l >= 1e-12, g_root * 0.5 / root, 0.0) + g_rt_carry
            g_eff = g_eff + g_rt
            g_rt_carry = cont * g_rt
            g_m = tl.where((m >= 0.0) & (m <= 1.0), g_eff * h, 0.0)
            tl.store(s_ptr + l * N + offs, g_m, mask=valid)
        tl.store(red_ptr + pid * (L + 1) * 8 + L * 8 + 0, tl.sum(gr * T, axis=0))
        tl.store(red_ptr + pid * (L + 1) * 8 + L * 8 + 1, tl.sum(gg * T, axis=0))
        tl.store(red_ptr + pid * (L + 1) * 8 + L * 8 + 2, tl.sum(gb * T, axis=0))

    @triton.jit
    def _cont_dz_kernel(z_ptr, gm_ptr, dz_ptr, H, W, L, scale, BLOCK: tl.constexpr):
        """dL/dz: each pixel's print mask feeds its own mask (weight 1) and
        its 8 neighbours' (0.0125 each)."""
        pid = tl.program_id(0)
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        N = H * W
        valid = offs < N
        y = offs // W
        x = offs % W
        up = y > 0
        dn = y < H - 1
        lf = x > 0
        rt = x < W - 1
        z = tl.load(z_ptr + offs, mask=valid, other=0.0)
        acc = tl.zeros([BLOCK], dtype=tl.float32)
        for l in range(0, L):
            b = gm_ptr + l * N + offs
            nb = (
                tl.load(b - W, mask=valid & up, other=0.0) + tl.load(b + W, mask=valid & dn, other=0.0)
                + tl.load(b - 1, mask=valid & lf, other=0.0) + tl.load(b + 1, mask=valid & rt, other=0.0)
                + tl.load(b - W - 1, mask=valid & up & lf, other=0.0)
                + tl.load(b - W + 1, mask=valid & up & rt, other=0.0)
                + tl.load(b + W - 1, mask=valid & dn & lf, other=0.0)
                + tl.load(b + W + 1, mask=valid & dn & rt, other=0.0)
            )
            gsum = tl.load(b, mask=valid, other=0.0) + 0.0125 * nb
            ps = _sig((z - (l + 0.5)) * scale)
            acc += gsum * (scale * ps * (1.0 - ps))
        tl.store(dz_ptr + offs, acc, mask=valid)


_BLOCK = 128
_BLOCK_BWD = 64


class _ContComposite(torch.autograd.Function):
    @staticmethod
    def forward(ctx, z, colors, ir, isl, w, cont, background, h, scale):
        H, W = z.shape
        L = colors.shape[0]
        z = z.contiguous().float()
        prm = torch.stack(
            [colors[:, 0], colors[:, 1], colors[:, 2], ir, isl, w, cont, torch.zeros_like(w)], dim=1
        ).float().contiguous()
        bg = background.detach().float().contiguous()
        out = torch.empty(H, W, 3, device=z.device, dtype=torch.float32)
        grid = (triton.cdiv(H * W, _BLOCK),)
        _cont_fwd_kernel[grid](
            z, out, prm, bg, out, out, out, H, W, L, float(h), 1.0 / float(h), float(scale),
            STORE=False, BLOCK=_BLOCK,
        )
        ctx.save_for_backward(z, prm, bg)
        ctx.h, ctx.scale = float(h), float(scale)
        return out

    @staticmethod
    def backward(ctx, grad_out):
        z, prm, bg = ctx.saved_tensors
        return _cont_backward(z, prm, bg, ctx.h, ctx.scale, (grad_out.float() * 255.0).contiguous()) + (None, None)


def _cont_backward(z, prm, bg, h, scale, g):
    """Gradients of the continuous composite for g = dL/d(colour, 0..1)
    per pixel [H,W,3]: (dz, dcolors, dir, disl, dw, None, dbackground)."""
    dz, red = _cont_backward_raw(z, prm, bg, h, scale, g)
    L = prm.shape[0]
    return dz, red[:L, 0:3], red[:L, 3], red[:L, 4], red[:L, 5], None, red[L, 0:3]


def _cont_backward_raw(z, prm, bg, h, scale, g):
    """(dz, red): red [L+1, 8] holds the per-layer table gradient in the
    table's layout (columns 6-7 zero) and the background's in row L."""
    if True:
        H, W = z.shape
        L = prm.shape[0]
        N = H * W
        rt = torch.empty(L, N, device=z.device, dtype=torch.float32)
        s = torch.empty(L, N, device=z.device, dtype=torch.float32)
        grid = (triton.cdiv(N, _BLOCK),)
        # The top-down pass is bound by its per-layer block reductions: small
        # single-warp blocks keep them to warp shuffles (2x faster than 128/4).
        grid_b = (triton.cdiv(N, _BLOCK_BWD),)
        red = torch.zeros(grid_b[0], L + 1, 8, device=z.device, dtype=torch.float32)
        _cont_fwd_kernel[grid](
            z, g, prm, bg, rt, s, g, H, W, L, h, 1.0 / h, scale, STORE=True, BLOCK=_BLOCK,
        )
        _cont_bwd_kernel[grid_b](
            z, g, prm, rt, s, red, H, W, L, h, 1.0 / h, scale, BLOCK=_BLOCK_BWD, num_warps=1,
        )
        del rt
        red = red.sum(0)
        dz = torch.empty_like(z)
        _cont_dz_kernel[grid](z, s, dz, H, W, L, scale, BLOCK=_BLOCK)
        return dz, red


if _HAS_TRITON:

    @triton.jit
    def _lab_loss_kernel(comp_ptr, tlab_ptr, w_ptr, g_ptr, part_ptr, N, inv_norm, BLOCK: tl.constexpr):
        """Weighted Lab squared error of a 0..255 composite (srgb_to_lab,
        fp32) - per-block partial sums, times inv_norm - and its gradient
        with respect to the composite, per pixel."""
        pid = tl.program_id(0)
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        valid = offs < N
        r0 = tl.load(comp_ptr + offs * 3 + 0, mask=valid, other=0.0) / 255.0
        g0 = tl.load(comp_ptr + offs * 3 + 1, mask=valid, other=0.0) / 255.0
        b0 = tl.load(comp_ptr + offs * 3 + 2, mask=valid, other=0.0) / 255.0
        # clamp to [0, 1]: no gradient outside (torch.clamp passes it at the bounds)
        ir = ((r0 >= 0.0) & (r0 <= 1.0)).to(tl.float32)
        ig = ((g0 >= 0.0) & (g0 <= 1.0)).to(tl.float32)
        ib = ((b0 >= 0.0) & (b0 <= 1.0)).to(tl.float32)
        r = tl.minimum(tl.maximum(r0, 0.0), 1.0)
        g = tl.minimum(tl.maximum(g0, 0.0), 1.0)
        b = tl.minimum(tl.maximum(b0, 0.0), 1.0)
        # inverse gamma
        ur = tl.maximum((r + 0.055) / 1.055, 1e-6)
        ug = tl.maximum((g + 0.055) / 1.055, 1e-6)
        ub = tl.maximum((b + 0.055) / 1.055, 1e-6)
        pr = tl.exp(tl.log(ur) * 2.4)
        pg = tl.exp(tl.log(ug) * 2.4)
        pb = tl.exp(tl.log(ub) * 2.4)
        lr = tl.where(r <= 0.04045, r / 12.92, pr)
        lg = tl.where(g <= 0.04045, g / 12.92, pg)
        lb = tl.where(b <= 0.04045, b / 12.92, pb)
        dlr = tl.where(r <= 0.04045, 1.0 / 12.92, 2.4 * pr / ur / 1.055)
        dlg = tl.where(g <= 0.04045, 1.0 / 12.92, 2.4 * pg / ug / 1.055)
        dlb = tl.where(b <= 0.04045, 1.0 / 12.92, 2.4 * pb / ub / 1.055)
        # XYZ over the white point
        tx = (0.4124564 * lr + 0.3575761 * lg + 0.1804375 * lb) / 0.95047
        ty = 0.2126729 * lr + 0.7151522 * lg + 0.0721750 * lb
        tz = (0.0193339 * lr + 0.1191920 * lg + 0.9503041 * lb) / 1.08883
        cx = tl.exp(tl.log(tl.maximum(tx, 1e-6)) * (1.0 / 3.0))
        cy = tl.exp(tl.log(tl.maximum(ty, 1e-6)) * (1.0 / 3.0))
        cz = tl.exp(tl.log(tl.maximum(tz, 1e-6)) * (1.0 / 3.0))
        fx = tl.where(tx > 0.008856, cx, (903.3 * tx + 16.0) / 116.0)
        fy = tl.where(ty > 0.008856, cy, (903.3 * ty + 16.0) / 116.0)
        fz = tl.where(tz > 0.008856, cz, (903.3 * tz + 16.0) / 116.0)
        dfx = tl.where(tx > 0.008856, cx / (3.0 * tl.maximum(tx, 1e-6)), 903.3 / 116.0)
        dfy = tl.where(ty > 0.008856, cy / (3.0 * tl.maximum(ty, 1e-6)), 903.3 / 116.0)
        dfz = tl.where(tz > 0.008856, cz / (3.0 * tl.maximum(tz, 1e-6)), 903.3 / 116.0)
        eL = 116.0 * fy - 16.0 - tl.load(tlab_ptr + offs * 3 + 0, mask=valid, other=0.0)
        ea = 500.0 * (fx - fy) - tl.load(tlab_ptr + offs * 3 + 1, mask=valid, other=0.0)
        eb = 200.0 * (fy - fz) - tl.load(tlab_ptr + offs * 3 + 2, mask=valid, other=0.0)
        wt = tl.load(w_ptr + offs, mask=valid, other=0.0)
        e = (eL * eL + ea * ea + eb * eb) * wt
        tl.store(part_ptr + pid, tl.sum(tl.where(valid, e, 0.0), axis=0) * inv_norm)
        # d loss / d (L, a, b) -> f -> t -> linear rgb -> 0..255 composite
        k = 2.0 * wt * inv_norm
        gx = (500.0 * ea * k) * dfx / 0.95047
        gy = (116.0 * eL - 500.0 * ea + 200.0 * eb) * k * dfy
        gz = (-200.0 * eb * k) * dfz / 1.08883
        gr = (0.4124564 * gx + 0.2126729 * gy + 0.0193339 * gz) * dlr * ir / 255.0
        gg = (0.3575761 * gx + 0.7151522 * gy + 0.1191920 * gz) * dlg * ig / 255.0
        gb = (0.1804375 * gx + 0.0721750 * gy + 0.9503041 * gz) * dlb * ib / 255.0
        tl.store(g_ptr + offs * 3 + 0, gr, mask=valid)
        tl.store(g_ptr + offs * 3 + 1, gg, mask=valid)
        tl.store(g_ptr + offs * 3 + 2, gb, mask=valid)


if _HAS_TRITON:

    @triton.jit
    def _hh_eff(pl_ptr, lab_ptr, off_ptr, du_ptr, o, m, HAS_DU: tl.constexpr):
        """Effective height logit: base + offset of the pixel's cluster
        (none for label 0) + smooth field."""
        e = tl.load(pl_ptr + o, mask=m, other=0.0)
        lab = tl.load(lab_ptr + o, mask=m, other=0)
        e += tl.where(lab != 0, tl.load(off_ptr + lab, mask=m, other=0.0), 0.0)
        if HAS_DU:
            e += tl.load(du_ptr + o, mask=m, other=0.0)
        return e

    @triton.jit
    def _hh_fwd_kernel(pl_ptr, lab_ptr, off_ptr, du_ptr, z_ptr, part_ptr, H, W, Lh, inv_h,
                       HAS_DU: tl.constexpr, BLOCK: tl.constexpr):
        """z = adaptive_round(L h sigmoid(eff) / h) at tau = 1 (soft round),
        and the block sums of the squared horizontal / vertical differences
        of eff (the smoothness penalty)."""
        pid = tl.program_id(0)
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        n = H * W
        valid = offs < n
        y = offs // W
        x = offs % W
        e = _hh_eff(pl_ptr, lab_ptr, off_ptr, du_ptr, offs, valid, HAS_DU)
        xc = (Lh * tl.sigmoid(e)) * inv_h
        fl = tl.floor(xc)
        z = fl + tl.sigmoid((xc - fl - 0.5) / 0.1)
        tl.store(z_ptr + offs, z, mask=valid)
        mr = valid & (x < W - 1)
        md = valid & (y < H - 1)
        er = _hh_eff(pl_ptr, lab_ptr, off_ptr, du_ptr, offs + 1, mr, HAS_DU)
        ed = _hh_eff(pl_ptr, lab_ptr, off_ptr, du_ptr, offs + W, md, HAS_DU)
        dh = tl.where(mr, (er - e) * (er - e), 0.0)
        dv = tl.where(md, (ed - e) * (ed - e), 0.0)
        nb = tl.num_programs(0)
        tl.store(part_ptr + pid, tl.sum(dh, axis=0))
        tl.store(part_ptr + nb + pid, tl.sum(dv, axis=0))

    @triton.jit
    def _hh_bwd_kernel(pl_ptr, lab_ptr, off_ptr, du_ptr, dz_ptr, gpen_ptr, deff_ptr, H, W, Lh, inv_h, ch, cv,
                       HAS_DU: tl.constexpr, BLOCK: tl.constexpr):
        pid = tl.program_id(0)
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        n = H * W
        valid = offs < n
        y = offs // W
        x = offs % W
        e = _hh_eff(pl_ptr, lab_ptr, off_ptr, du_ptr, offs, valid, HAS_DU)
        sg = tl.sigmoid(e)
        xc = (Lh * sg) * inv_h
        fl = tl.floor(xc)
        sr = tl.sigmoid((xc - fl - 0.5) / 0.1)
        dz = tl.load(dz_ptr + offs, mask=valid, other=0.0)
        d = dz * (sr * (1.0 - sr) / 0.1) * (Lh * inv_h) * (sg * (1.0 - sg))
        gp = tl.load(gpen_ptr)
        ml = valid & (x > 0)
        mr = valid & (x < W - 1)
        mu = valid & (y > 0)
        md = valid & (y < H - 1)
        el = _hh_eff(pl_ptr, lab_ptr, off_ptr, du_ptr, offs - 1, ml, HAS_DU)
        er = _hh_eff(pl_ptr, lab_ptr, off_ptr, du_ptr, offs + 1, mr, HAS_DU)
        eu = _hh_eff(pl_ptr, lab_ptr, off_ptr, du_ptr, offs - W, mu, HAS_DU)
        ed = _hh_eff(pl_ptr, lab_ptr, off_ptr, du_ptr, offs + W, md, HAS_DU)
        dp = (
            tl.where(ml, 2.0 * ch * (e - el), 0.0) - tl.where(mr, 2.0 * ch * (er - e), 0.0)
            + tl.where(mu, 2.0 * cv * (e - eu), 0.0) - tl.where(md, 2.0 * cv * (ed - e), 0.0)
        )
        tl.store(deff_ptr + offs, d + gp * dp, mask=valid)

    @triton.jit
    def _segsum_kernel(x_ptr, perm_ptr, ends_ptr, out_ptr, BLOCK: tl.constexpr):
        """out[c] = sum of x over the pixels labelled c (perm sorted by label,
        ends its cumulative counts), in a fixed order; out[0] = 0 (label 0
        has no offset)."""
        c = tl.program_id(0)
        end = tl.load(ends_ptr + c)
        start = tl.where(c > 0, tl.load(ends_ptr + c - 1, mask=c > 0, other=0), 0)
        acc = tl.zeros([BLOCK], dtype=tl.float32)
        for lo in range(start, end, BLOCK):
            o = lo + tl.arange(0, BLOCK)
            m = o < end
            idx = tl.load(perm_ptr + o, mask=m, other=0)
            acc += tl.load(x_ptr + idx, mask=m, other=0.0)
        tot = tl.sum(acc, axis=0)
        tl.store(out_ptr + c, tl.where(c > 0, tot, 0.0))


class _HeightHead(torch.autograd.Function):
    """(z, penalty) of the training step's height map: effective logits =
    base + per-cluster offset + smooth field, z their soft-rounded layer
    index (adaptive_round at tau = 1), penalty the smoothness term of
    compute_loss (coef * mean squared neighbour differences)."""

    @staticmethod
    def forward(ctx, offsets, du, pl, labels, perm, ends, L, h, coef):
        H, W = pl.shape
        N = H * W
        has_du = du is not None
        du_ = du.contiguous().float() if has_du else pl
        BLOCK = 256
        nblk = triton.cdiv(N, BLOCK)
        z = torch.empty(H, W, device=pl.device, dtype=torch.float32)
        part = torch.empty(2, nblk, device=pl.device, dtype=torch.float32)
        off = offsets.contiguous().float()
        _hh_fwd_kernel[(nblk,)](pl, labels, off, du_, z, part, H, W, float(L) * float(h), 1.0 / float(h),
                                HAS_DU=has_du, BLOCK=BLOCK)
        sums = torch.empty(2, device=pl.device, dtype=torch.float32)
        _rowsum_kernel[(2,)](part, sums, nblk, BLOCK=1024)
        ch = float(coef) / max(H * (W - 1), 1)
        cv = float(coef) / max((H - 1) * W, 1)
        pen = sums[0] * ch + sums[1] * cv
        ctx.save_for_backward(off, du_, pl, labels, perm, ends)
        ctx.cfg = (has_du, float(L) * float(h), 1.0 / float(h), ch, cv, offsets.shape[0])
        return z, pen

    @staticmethod
    def backward(ctx, dz, dpen):
        off, du_, pl, labels, perm, ends = ctx.saved_tensors
        has_du, Lh, inv_h, ch, cv, C = ctx.cfg
        H, W = pl.shape
        N = H * W
        if dz is None:
            dz = torch.zeros(H, W, device=pl.device, dtype=torch.float32)
        if dpen is None:
            dpen = torch.zeros((), device=pl.device, dtype=torch.float32)
        BLOCK = 256
        deff = torch.empty(H, W, device=pl.device, dtype=torch.float32)
        _hh_bwd_kernel[(triton.cdiv(N, BLOCK),)](
            pl, labels, off, du_, dz.contiguous().float(), dpen.reshape(1).float(), deff,
            H, W, Lh, inv_h, ch, cv, HAS_DU=has_du, BLOCK=BLOCK,
        )
        doff = torch.empty(C, device=pl.device, dtype=torch.float32)
        _segsum_kernel[(C,)](deff, perm, ends, doff, BLOCK=1024)
        return doff.view_as(off), (deff if has_du else None), None, None, None, None, None, None, None


def height_head(offsets, du, pixel_logits, labels_i32, perm, ends, max_layers: int, h: float, coef: float):
    """(z [H,W], penalty) - see _HeightHead. ``labels_i32`` int32 [H,W],
    ``perm``/``ends`` the label plan of DeterministicOps.gather_plan."""
    return _HeightHead.apply(
        offsets.reshape(-1), du, pixel_logits.contiguous().float(), labels_i32, perm, ends,
        int(max_layers), float(h), float(coef),
    )


if _HAS_TRITON:

    @triton.jit
    def _head_fwd_kernel(
        g_ptr, e_ptr, c_ptr, t_ptr, p_ptr, col_ptr, td_ptr, tdraw_ptr, rt_ptr, cont_ptr, scr_ptr,
        tp_ptr, rho_ptr, lim_ptr, lam_ptr, gate_ptr, cm_ptr, boh_ptr, viol_ptr, lossc_ptr, dgc_ptr,
        L, M, tau, h,
        CONSTR: tl.constexpr, BL: tl.constexpr, BM: tl.constexpr,
    ):
        """Per-layer material mix of a training step (Gumbel softmax of the
        logits, the mixed colour and TD, the runs of the most likely
        material) and, with CONSTR, the colour/swap constraint terms of
        _forward_backward with their gradient (for a unit upstream)."""
        rl = tl.arange(0, BL)
        rm = tl.arange(0, BM)
        lv = rl < L
        mv = rm < M
        m2 = lv[:, None] & mv[None, :]
        o2 = rl[:, None] * M + rm[None, :]
        g = tl.load(g_ptr + o2, mask=m2, other=0.0)
        e = tl.load(e_ptr + o2, mask=m2, other=1.0)
        x = tl.where(m2, (g + (-tl.log(e))) / tau, float("-inf"))
        x = tl.where(lv[:, None], x, 0.0)
        ex = tl.exp(x - tl.max(x, axis=1)[:, None])
        p = ex / tl.sum(ex, axis=1)[:, None]
        tl.store(p_ptr + o2, p, mask=m2)
        c0 = tl.load(c_ptr + rm * 3 + 0, mask=mv, other=0.0)
        c1 = tl.load(c_ptr + rm * 3 + 1, mask=mv, other=0.0)
        c2 = tl.load(c_ptr + rm * 3 + 2, mask=mv, other=0.0)
        tt = tl.load(t_ptr + rm, mask=mv, other=0.0)
        tl.store(col_ptr + rl * 3 + 0, tl.sum(p * c0[None, :], axis=1), mask=lv)
        tl.store(col_ptr + rl * 3 + 1, tl.sum(p * c1[None, :], axis=1), mask=lv)
        tl.store(col_ptr + rl * 3 + 2, tl.sum(p * c2[None, :], axis=1), mask=lv)
        tdraw = tl.sum(p * tt[None, :], axis=1)
        tl.store(tdraw_ptr + rl, tdraw, mask=lv)
        tl.store(td_ptr + rl, tl.minimum(tl.maximum(tdraw, 1e-8), 1e8), mask=lv)
        # runs of the most likely material (first maximum, as torch.argmax)
        idx = tl.argmax(tl.where(mv[None, :], p, -1.0), axis=1)
        tl.store(scr_ptr + rl, idx.to(tl.float32), mask=lv)
        tl.debug_barrier()
        prev = tl.load(scr_ptr + rl - 1, mask=lv & (rl >= 1), other=-1.0)
        tl.debug_barrier()
        change = (rl == 0) | (idx.to(tl.float32) != prev)
        rs = tl.associative_scan(tl.where(change & lv, rl, 0), 0, _imax)
        tl.store(rt_ptr + rl, (rl - rs + 1).to(tl.float32) * h, mask=lv)
        tl.store(cont_ptr + rl, (rs < rl).to(tl.float32), mask=lv)
        if CONSTR:
            gm = tl.where(m2, g, float("-inf"))
            gm = tl.where(lv[:, None], gm, 0.0)
            mx = tl.max(gm, axis=1)
            eg = tl.exp(gm - mx[:, None])
            sg = tl.sum(eg, axis=1)
            q = eg / sg[:, None]
            logq = tl.where(m2, (g - mx[:, None]) - tl.log(sg)[:, None], 0.0)
            tp = tl.load(tp_ptr + o2, mask=m2, other=0.0)
            rho = tl.load(rho_ptr)
            Lf = L.to(tl.float32)
            term1 = -rho * tl.sum(tl.sum(tp * logq, axis=1), axis=0) / Lf
            # the neighbouring layers' distributions (scratch rows)
            so = scr_ptr + BL + rl[:, None] * BM + rm[None, :]
            tl.store(so, q)
            tl.debug_barrier()
            q_prev = tl.load(so - BM, mask=(rl[:, None] >= 1) & lv[:, None] & mv[None, :], other=0.0)
            q_next = tl.load(so + BM, mask=(rl[:, None] < L - 1) & lv[:, None] & mv[None, :], other=0.0)
            tl.debug_barrier()
            boh = tl.load(boh_ptr + rm, mask=mv, other=0.0)
            cm = tl.load(cm_ptr + rm, mask=mv, other=0.0)
            first = (rl == 0) & lv
            e_sw = tl.sum(tl.where(lv & (rl >= 1), 1.0 - tl.sum(q * q_prev, axis=1), 0.0), axis=0)
            e_sw += tl.sum(tl.where(first, 1.0 - tl.sum(q * boh[None, :], axis=1), 0.0), axis=0)
            qc = tl.minimum(q, 1.0 - 1e-6)
            la = tl.sum(tl.where(m2, tl.log(1.0 - qc), 0.0), axis=0)  # log1p(-q)
            A = tl.exp(la)
            e_co = tl.sum(tl.where(mv, (1.0 - A) * cm, 0.0), axis=0)
            lim0 = tl.load(lim_ptr + 0)
            lim1 = tl.load(lim_ptr + 1)
            d0 = e_co - lim0
            d1 = e_sw - lim1
            ex0 = tl.maximum(d0, 0.0)
            ex1 = tl.maximum(d1, 0.0)
            tl.store(viol_ptr + 0, ex0)
            tl.store(viol_ptr + 1, ex1)
            gate = tl.load(gate_ptr)
            lam0 = tl.load(lam_ptr + 0)
            lam1 = tl.load(lam_ptr + 1)
            tl.store(lossc_ptr, term1 + gate * (lam0 * ex0 + lam1 * ex1))
            a_c = tl.where(d0 >= 0.0, gate * lam0, 0.0)
            a_s = tl.where(d1 >= 0.0, gate * lam1, 0.0)
            dg1 = (-rho / Lf) * (tp - q * tl.sum(tp, axis=1)[:, None])
            dq = a_s * (-(q_prev + q_next) - tl.where(first[:, None], boh[None, :], 0.0))
            dq += a_c * tl.where(q <= 1.0 - 1e-6, cm[None, :] * A[None, :] / (1.0 - qc), 0.0)
            dg2 = q * (dq - tl.sum(q * dq, axis=1)[:, None])
            tl.store(dgc_ptr + o2, dg1 + dg2, mask=m2)

    @triton.jit
    def _imax(a, b):
        return tl.maximum(a, b)

    @triton.jit
    def _head_bwd_kernel(
        dprm_ptr, p_ptr, td_ptr, tdraw_ptr, k_ptr, dark_ptr, c_ptr, t_ptr, gc_ptr, dgc_ptr, dg_ptr,
        L, M, tau,
        CONSTR: tl.constexpr, BL: tl.constexpr, BM: tl.constexpr,
    ):
        """Gradient of the logits from the composite's per-layer table
        gradients (colour, 1/sqrt(reach), 1/sqrt(slow reach), w) and the
        constraint terms' upstream gradient."""
        rl = tl.arange(0, BL)
        rm = tl.arange(0, BM)
        lv = rl < L
        mv = rm < M
        m2 = lv[:, None] & mv[None, :]
        o2 = rl[:, None] * M + rm[None, :]
        dc0 = tl.load(dprm_ptr + rl * 8 + 0, mask=lv, other=0.0)
        dc1 = tl.load(dprm_ptr + rl * 8 + 1, mask=lv, other=0.0)
        dc2 = tl.load(dprm_ptr + rl * 8 + 2, mask=lv, other=0.0)
        dir_ = tl.load(dprm_ptr + rl * 8 + 3, mask=lv, other=0.0)
        disl = tl.load(dprm_ptr + rl * 8 + 4, mask=lv, other=0.0)
        dw = tl.load(dprm_ptr + rl * 8 + 5, mask=lv, other=0.0)
        td = tl.load(td_ptr + rl, mask=lv, other=1.0)
        tdraw = tl.load(tdraw_ptr + rl, mask=lv, other=1.0)
        k = tl.load(k_ptr + rl, mask=lv, other=1.0)
        dark = tl.load(dark_ptr + rl, mask=lv, other=0.0)
        reach = tl.maximum(0.1 * td, 1e-9)
        ir = 1.0 / tl.sqrt(reach)
        isl = 1.0 / tl.sqrt(tl.sqrt(k * reach))
        d_reach = dir_ * (-0.5 * ir / reach) + disl * (-0.25 * isl / reach)
        wd = (1.0 - td) / 0.4
        dtd = tl.where(0.1 * td >= 1e-9, d_reach * 0.1, 0.0)
        dtd += tl.where((dark > 0.5) & (wd >= 0.0) & (wd <= 1.0), dw * -2.5, 0.0)
        dtd = tl.where((tdraw >= 1e-8) & (tdraw <= 1e8), dtd, 0.0)
        c0 = tl.load(c_ptr + rm * 3 + 0, mask=mv, other=0.0)
        c1 = tl.load(c_ptr + rm * 3 + 1, mask=mv, other=0.0)
        c2 = tl.load(c_ptr + rm * 3 + 2, mask=mv, other=0.0)
        tt = tl.load(t_ptr + rm, mask=mv, other=0.0)
        dp = dc0[:, None] * c0[None, :] + dc1[:, None] * c1[None, :] + dc2[:, None] * c2[None, :] + dtd[:, None] * tt[None, :]
        p = tl.load(p_ptr + o2, mask=m2, other=0.0)
        dgr = p * (dp - tl.sum(p * dp, axis=1)[:, None]) / tau
        if CONSTR:
            dgr += tl.load(gc_ptr) * tl.load(dgc_ptr + o2, mask=m2, other=0.0)
        tl.store(dg_ptr + o2, dgr, mask=m2)


class _ParamHead(torch.autograd.Function):
    """global logits -> the continuous composite's per-layer table [L,8] and
    the constraint terms (0 without), in two kernels forward (material mix
    + constraints, lightness fixed point + table) and one backward - the
    ~150 small ops of _composite_cont_mat, cont_coverage_params, the table
    packing, _expected_counts and the proximal term."""

    @staticmethod
    def forward(ctx, G, E, C, T, bg, h, tau, cons):
        L, M = G.shape
        dev = G.device
        BL = max(16, triton.next_power_of_2(L))
        BM = max(16, triton.next_power_of_2(M))
        f32 = dict(device=dev, dtype=torch.float32)
        p = torch.empty(L, M, **f32)
        col = torch.empty(L, 3, **f32)
        td = torch.empty(L, **f32)
        tdraw = torch.empty(L, **f32)
        rt = torch.empty(L, **f32)
        cont = torch.empty(L, **f32)
        scr = torch.empty(BL + BL * BM, **f32)
        lossc = torch.zeros(1, **f32)
        constr = cons is not None
        if constr:
            tp, rho, lim, lam, gate, cm, boh, viol = cons
            dgc = torch.empty(L, M, **f32)
        else:
            tp = rho = lim = lam = gate = cm = boh = viol = dgc = p
        Gc = G.detach().float().contiguous()
        _head_fwd_kernel[(1,)](
            Gc, E.contiguous(), C, T, p, col, td, tdraw, rt, cont, scr,
            tp, rho, lim, lam, gate, cm, boh, viol, lossc, dgc,
            L, M, float(tau), float(h), CONSTR=constr, BL=BL, BM=BM, num_warps=4,
        )
        k = torch.empty(L, **f32)
        dark = torch.empty(L, **f32)
        prm = torch.empty(L, 8, **f32)
        scr2 = torch.empty(1, BL, **f32)
        bgc = bg.detach().to(torch.float32).contiguous()
        _coverage_fp_kernel[(1,)](
            col, td, rt, cont, bgc, 0, k, dark, prm, scr2, L,
            STEPS=8, BL=BL, PALETTE=False, PRM=True, num_warps=1 if BL <= 128 else 4,
        )
        ctx.save_for_backward(p, td, tdraw, k, dark, C, T, dgc if constr else p)
        ctx.cfg = (L, M, float(tau), constr, BL, BM)
        return prm, lossc[0]

    @staticmethod
    def backward(ctx, dprm, dlossc):
        p, td, tdraw, k, dark, C, T, dgc = ctx.saved_tensors
        L, M, tau, constr, BL, BM = ctx.cfg
        if dprm is None:
            dprm = torch.zeros(L, 8, device=p.device, dtype=torch.float32)
        gc = dlossc.reshape(1).float() if dlossc is not None else torch.zeros(1, device=p.device)
        dG = torch.empty(L, M, device=p.device, dtype=torch.float32)
        _head_bwd_kernel[(1,)](
            dprm.contiguous().float(), p, td, tdraw, k, dark, C, T, gc, dgc, dG,
            L, M, tau, CONSTR=constr, BL=BL, BM=BM, num_warps=4,
        )
        return dG, None, None, None, None, None, None, None


def param_head(global_logits, gumbel_exp, material_colors, material_TDs, background, h, tau_global, constraint=None):
    """(table [L,8], constraint loss) - see _ParamHead. ``constraint``:
    (proximal target [L,M], rho, limits [2], multipliers [2], gate, colour
    mask [M], base one-hot [M], violation out [2]) or None."""
    return _ParamHead.apply(
        global_logits, gumbel_exp.float(), material_colors.float().contiguous(),
        material_TDs.float().contiguous(), background, float(h), float(tau_global), constraint,
    )


class _ContCompositeLossPrm(torch.autograd.Function):
    """_ContCompositeLoss on a ready per-layer table [L,8]."""

    @staticmethod
    def forward(ctx, z, prm, background, h, scale, tlab, wts, inv_norm):
        H, W = z.shape
        L = prm.shape[0]
        N = H * W
        z = z.contiguous().float()
        prm = prm.contiguous()
        bg = background.detach().float().contiguous()
        out = torch.empty(H, W, 3, device=z.device, dtype=torch.float32)
        grid = (triton.cdiv(N, _BLOCK),)
        _cont_fwd_kernel[grid](
            z, out, prm, bg, out, out, out, H, W, L, float(h), 1.0 / float(h), float(scale),
            STORE=False, BLOCK=_BLOCK,
        )
        BL = 256
        nblk = triton.cdiv(N, BL)
        part = torch.empty(1, nblk, device=z.device, dtype=torch.float32)
        g = torch.empty(H, W, 3, device=z.device, dtype=torch.float32)
        _lab_loss_kernel[(nblk,)](out, tlab, wts, g, part, N, float(inv_norm), BLOCK=BL)
        loss = torch.empty(1, device=z.device, dtype=torch.float32)
        _rowsum_kernel[(1,)](part, loss, nblk, BLOCK=1024)
        ctx.save_for_backward(z, prm, bg, g)
        ctx.h, ctx.scale = float(h), float(scale)
        return loss[0]

    @staticmethod
    def backward(ctx, grad_loss):
        z, prm, bg, g = ctx.saved_tensors
        L = prm.shape[0]
        g01 = (g * (grad_loss.float() * 255.0)).contiguous()
        dz, red = _cont_backward_raw(z, prm, bg, ctx.h, ctx.scale, g01)
        return dz, red[:L], red[L, 0:3], None, None, None, None, None


def composite_loss_prm(continuous_z, prm, background, h, tau_height, target, focus_map=None, alpha=None):
    """composite_loss_cont_fused with the per-layer table from param_head."""
    scale = 10.0 / (float(tau_height) + 1e-8)
    tlab, wts, inv = loss_weights(target, focus_map, alpha)
    return _ContCompositeLossPrm.apply(continuous_z, prm, background.float(), float(h), scale, tlab, wts, inv)


if _HAS_TRITON:

    @triton.jit
    def _cadamw_kernel(p_ptr, g_ptr, m_ptr, v_ptr, ema_ptr, nss_ptr, n, b1, b2, eps, decay,
                       HAS_EMA: tl.constexpr, BLOCK: tl.constexpr):
        """One CAdamW step of a whole (small) parameter in one program -
        moments, cautious mask normalised by its mean, update with the step
        size nss_ptr[0] (negative) - and its running average."""
        o = tl.arange(0, BLOCK)
        valid = o < n
        g = tl.load(g_ptr + o, mask=valid, other=0.0)
        m = tl.load(m_ptr + o, mask=valid, other=0.0) * b1 + g * (1.0 - b1)
        v = tl.load(v_ptr + o, mask=valid, other=0.0) * b2 + (1.0 - b2) * g * g
        denom = tl.sqrt(v) + eps
        mask = ((m * g) > 0.0).to(tl.float32)
        mean = tl.sum(tl.where(valid, mask, 0.0), axis=0) / n
        mask = mask / tl.maximum(mean, 1e-3)
        p = tl.load(p_ptr + o, mask=valid, other=0.0) + ((m * mask) / denom) * tl.load(nss_ptr)
        tl.store(m_ptr + o, m, mask=valid)
        tl.store(v_ptr + o, v, mask=valid)
        tl.store(p_ptr + o, p, mask=valid)
        if HAS_EMA:
            e = tl.load(ema_ptr + o, mask=valid, other=0.0) * decay + p * (1.0 - decay)
            tl.store(ema_ptr + o, e, mask=valid)


CADAMW_MAX = 16384


def cadamw_fused(p, grad, exp_avg, exp_avg_sq, ema, neg_step, beta1, beta2, eps, decay):
    """CAdamW update (+ running average) of a contiguous fp32 parameter of
    at most CADAMW_MAX elements in one launch."""
    n = p.numel()
    BLOCK = max(16, triton.next_power_of_2(n))
    _cadamw_kernel[(1,)](
        p, grad, exp_avg, exp_avg_sq, ema if ema is not None else p, neg_step, n,
        float(beta1), float(beta2), float(eps), float(decay),
        HAS_EMA=ema is not None, BLOCK=BLOCK, num_warps=4 if BLOCK <= 4096 else 8,
    )


def loss_weights(target: torch.Tensor, focus_map, alpha):
    """(target Lab fp32 [H,W,3], per-pixel weights [H,W], 1/(3 sum w)) as
    compute_loss weighs pixels, cached on the target tensor."""
    key = (id(focus_map), id(alpha))
    cache = getattr(target, "_af_cont_loss", None)
    if cache is None or cache[0] != key:
        from autoforge.Helper.ImageHelper import srgb_to_lab
        import torch.nn.functional as F

        with torch.autocast(target.device.type, enabled=False):
            tl32 = srgb_to_lab(target.to(torch.float32)).contiguous()
        H, W = target.shape[:2]
        w = torch.ones(H, W, device=target.device, dtype=torch.float32)
        if focus_map is not None:
            fm = focus_map.squeeze(-1) if focus_map.dim() == 3 and focus_map.shape[-1] == 1 else focus_map
            w = w * (0.1 + 0.9 * torch.clamp(fm.float(), min=0.0))
        if alpha is not None:
            a = alpha.squeeze(-1) if alpha.dim() == 3 and alpha.shape[-1] == 1 else alpha
            if a.shape != w.shape:
                a = F.interpolate(a[None, None].float(), size=w.shape, mode="nearest")[0, 0]
            w = w * (a >= 128).float()
        # compute_loss: mean(pixel mse * w) / max(mean(w), 1e-8)
        inv = 1.0 / (3.0 * max(float(w.sum()), 1e-8 * H * W))
        cache = (key, tl32, w.contiguous(), inv)
        target._af_cont_loss = cache
    return cache[1], cache[2], cache[3]


class _ContCompositeLoss(torch.autograd.Function):
    """compute_loss(composite_cont_fused(...)) without the penalty term: the
    Lab error and its gradient fused into one kernel after the composite."""

    @staticmethod
    def forward(ctx, z, colors, ir, isl, w, cont, background, h, scale, tlab, wts, inv_norm):
        H, W = z.shape
        L = colors.shape[0]
        N = H * W
        z = z.contiguous().float()
        prm = torch.stack(
            [colors[:, 0], colors[:, 1], colors[:, 2], ir, isl, w, cont, torch.zeros_like(w)], dim=1
        ).float().contiguous()
        bg = background.detach().float().contiguous()
        out = torch.empty(H, W, 3, device=z.device, dtype=torch.float32)
        grid = (triton.cdiv(N, _BLOCK),)
        _cont_fwd_kernel[grid](
            z, out, prm, bg, out, out, out, H, W, L, float(h), 1.0 / float(h), float(scale),
            STORE=False, BLOCK=_BLOCK,
        )
        BL = 256
        nblk = triton.cdiv(N, BL)
        part = torch.empty(1, nblk, device=z.device, dtype=torch.float32)
        g = torch.empty(H, W, 3, device=z.device, dtype=torch.float32)
        _lab_loss_kernel[(nblk,)](out, tlab, wts, g, part, N, float(inv_norm), BLOCK=BL)
        loss = torch.empty(1, device=z.device, dtype=torch.float32)
        _rowsum_kernel[(1,)](part, loss, nblk, BLOCK=1024)
        ctx.save_for_backward(z, prm, bg, g)
        ctx.h, ctx.scale = float(h), float(scale)
        return loss[0]

    @staticmethod
    def backward(ctx, grad_loss):
        z, prm, bg, g = ctx.saved_tensors
        g01 = (g * (grad_loss.float() * 255.0)).contiguous()
        return _cont_backward(z, prm, bg, ctx.h, ctx.scale, g01) + (None,) * 5


def composite_loss_cont_fused(continuous_z, layer_colors, reach, slow_reach, cov_w, run_start, background, h,
                              tau_height, target, focus_map=None, alpha=None):
    """compute_loss (without the height penalty) of composite_cont_fused's
    image, the Lab conversion and its gradient fused (fp32)."""
    L = layer_colors.shape[0]
    idx = torch.arange(L, device=layer_colors.device)
    cont = (run_start < idx).to(torch.float32)
    scale = 10.0 / (float(tau_height) + 1e-8)
    tlab, wts, inv = loss_weights(target, focus_map, alpha)
    return _ContCompositeLoss.apply(
        continuous_z, layer_colors.float(), torch.rsqrt(reach.float()), torch.rsqrt(slow_reach.float()),
        cov_w.float(), cont, background.float(), float(h), scale, tlab, wts, inv,
    )


def composite_cont_fused(continuous_z, layer_colors, reach, slow_reach, cov_w, run_start, background, h, tau_height):
    """[H,W,3] (0..255) continuous composite from the per-pixel continuous
    layer index and the per-layer coverage parameters (see
    ``composite_image_cont``); differentiable in all of them but run_start."""
    L = layer_colors.shape[0]
    idx = torch.arange(L, device=layer_colors.device)
    cont = (run_start < idx).to(torch.float32)
    scale = 10.0 / (float(tau_height) + 1e-8)
    return _ContComposite.apply(
        continuous_z, layer_colors.float(), torch.rsqrt(reach.float()), torch.rsqrt(slow_reach.float()),
        cov_w.float(), cont, background.float(), float(h), scale,
    )


if _HAS_TRITON:

    @triton.jit
    def _rowsum_kernel(x_ptr, out_ptr, N, BLOCK: tl.constexpr):
        # One program per row, fixed order: the sum of a row never depends on
        # how many rows are summed with it.
        r = tl.program_id(0)
        acc = tl.zeros([BLOCK], dtype=tl.float32)
        for lo in range(0, N, BLOCK):
            o = lo + tl.arange(0, BLOCK)
            acc += tl.load(x_ptr + r * N + o, mask=o < N, other=0.0)
        tl.store(out_ptr + r, tl.sum(acc, axis=0))


def batched_layer_params(cols: torch.Tensor, tds: torch.Tensor, background, h: float) -> torch.Tensor:
    """``layer_params`` for B stacks at once: cols [B,L,3], tds [B,L] ->
    [B,L,8] (one fixed-point kernel for all of them)."""
    cols = cols.to(torch.float32)
    tds = tds.to(torch.float32)
    B, L = tds.shape
    change = torch.ones(B, L, dtype=torch.bool, device=tds.device)
    change[:, 1:] = (cols[:, 1:] != cols[:, :-1]).any(-1) | (tds[:, 1:] != tds[:, :-1])
    idx = torch.arange(L, device=tds.device).view(1, L)
    run_start = torch.cummax(torch.where(change, idx, torch.zeros_like(idx)), dim=1)[0]
    k, dark = coverage_fixed_point(cols, tds, run_start, background, h)
    reach = torch.clamp(0.1 * tds, min=1e-9)
    w_dark = torch.clamp((1.0 - tds) / 0.4, 0.0, 1.0)
    p = torch.zeros(B, L, 8, device=cols.device, dtype=torch.float32)
    p[..., :3] = cols
    p[..., 3] = torch.rsqrt(reach)
    p[..., 4] = torch.rsqrt(torch.sqrt(k * reach))
    p[..., 5] = torch.where(dark, w_dark, torch.ones_like(w_dark))
    p[..., 6] = (run_start < idx).float()
    return p


def batched_heights_error_sum(z, params: torch.Tensor, bg: torch.Tensor, h: float, target_lab, weights) -> torch.Tensor:
    """[B] summed weighted Lab squared error of heights ``z`` [H,W] (or one
    map per stack, [B,H,W]) under each of the B stacks ``params`` [B,L,8] -
    deterministic and the same for a stack whatever else is in the batch."""
    H, W = z.shape[-2:]
    B, L = int(params.shape[0]), int(params.shape[1])
    zstride = H * W if z.dim() == 3 else 0
    zi = z.to(torch.int32).contiguous()
    BLOCK = 256
    nblk = triton.cdiv(H * W, BLOCK)
    part = torch.empty(B, nblk, device=z.device, dtype=torch.float32)
    _disc_composite_kernel[(nblk, B)](
        zi, part, params.contiguous(), bg, H, W, L,
        float(h), 1.0 / float(h), target_lab.contiguous(), weights.contiguous(), 0.0, zstride,
        ERR=True, BLOCK=BLOCK, LOSS=True,
    )
    out = torch.empty(B, device=z.device, dtype=torch.float32)
    _rowsum_kernel[(B,)](part, out, nblk, BLOCK=1024)
    return out


def batched_heights_loss(optimizer, z: torch.Tensor, cols: torch.Tensor, tds: torch.Tensor) -> torch.Tensor:
    """[B] ``heights_loss`` of B candidate stacks (cols [B,L,3], tds [B,L])."""
    tl32, w, wsum = _loss_inputs(optimizer)
    params = batched_layer_params(cols, tds, optimizer.background, optimizer.h)
    e = batched_heights_error_sum(z, params, background_tensor(optimizer.background), optimizer.h, tl32, w)
    return e / (3.0 * wsum)


def _loss_inputs(optimizer):
    """(target Lab, per-pixel loss weights, their sum) for the optimizer's
    current target, cached on the target tensor."""
    t = optimizer.target
    key = (id(optimizer.focus_map), id(optimizer.alpha))
    cache = getattr(t, "_af_fused_loss", None)
    if cache is None or cache[0] != key:
        from autoforge.Helper.ImageHelper import srgb_to_lab
        from autoforge.Helper.PixelHeightRefine import _pixel_weights

        tl32 = srgb_to_lab(t.to(torch.float32)).contiguous()
        w = _pixel_weights(optimizer, t.shape[:2]).contiguous()
        cache = (key, tl32, w, w.sum().clamp(min=1e-8))
        t._af_fused_loss = cache
    return cache[1], cache[2], cache[3]


def heights_loss(optimizer, z: torch.Tensor, cols: torch.Tensor, tds: torch.Tensor) -> torch.Tensor:
    """0-dim discrete loss (compute_loss: weighted mean Lab MSE) of the
    whole-layer heights ``z`` under per-layer colours/TDs, fused."""
    return batched_heights_loss(optimizer, z, cols[None], tds[None])[0]


# --------------------------------------------------------------------------
# Coverage parameters: the lightness fixed point of ``layer_coverage_params``
# (and ``batched_stack_palette``) as one kernel, one program per stack. The
# torch version is ~400 tiny launches per call (8 refinements of a
# [L+1,L+1] cumprod + matmul, the lightness and the coverage), which on a
# 75-layer stack is pure launch overhead. The stack colour after each layer
# is the linear recurrence c_l = (1 - o_l) c_{l-1} + o_l col_l, computed
# here with an associative scan; the one-layer shifts go through a small
# scratch row.
# --------------------------------------------------------------------------

if _HAS_TRITON:

    @triton.jit
    def _lightness(r, g, b):
        r = tl.floor(tl.minimum(tl.maximum(r, 0.0), 1.0) * 255.0 + 1e-4) / 255.0
        g = tl.floor(tl.minimum(tl.maximum(g, 0.0), 1.0) * 255.0 + 1e-4) / 255.0
        b = tl.floor(tl.minimum(tl.maximum(b, 0.0), 1.0) * 255.0 + 1e-4) / 255.0
        r = tl.where(r <= 0.04045, r / 12.92, tl.exp(tl.log((tl.maximum(r, 0.04045) + 0.055) / 1.055) * 2.4))
        g = tl.where(g <= 0.04045, g / 12.92, tl.exp(tl.log((tl.maximum(g, 0.04045) + 0.055) / 1.055) * 2.4))
        b = tl.where(b <= 0.04045, b / 12.92, tl.exp(tl.log((tl.maximum(b, 0.04045) + 0.055) / 1.055) * 2.4))
        y = r * 0.2126 + g * 0.7152 + b * 0.0722
        f = tl.where(
            y > 0.008856,
            tl.exp(tl.log(tl.maximum(y, 0.008856)) * (1.0 / 3.0)),
            7.787 * y + 0.137931,
        )
        return (116.0 * f - 16.0) / 100.0

    @triton.jit
    def _lin_combine(a1, r1, g1, b1, a2, r2, g2, b2):
        return a1 * a2, a2 * r1 + r2, a2 * g1 + g2, a2 * b1 + b2

    @triton.jit
    def _shift_up(x, scr, i, valid, fill):
        """x[i-1] (``fill`` at i == 0) through the scratch row."""
        tl.debug_barrier()
        tl.store(scr + i, x, mask=valid)
        tl.debug_barrier()
        return tl.load(scr + i - 1, mask=valid & (i >= 1), other=fill)

    @triton.jit
    def _stack_after(cr, cg, cb, op, bgr, bgg, bgb, valid):
        a = tl.where(valid, 1.0 - op, 1.0)
        pr = tl.where(valid, op * cr, 0.0)
        pg = tl.where(valid, op * cg, 0.0)
        pb = tl.where(valid, op * cb, 0.0)
        A, Br, Bg, Bb = tl.associative_scan((a, pr, pg, pb), 0, _lin_combine)
        return A * bgr + Br, A * bgg + Bg, A * bgb + Bb

    @triton.jit
    def _coverage_fp_kernel(
        col_ptr, td_ptr, rt_ptr, cont_ptr, bg_ptr, bg_stride, k_ptr, dark_ptr, pal_ptr, scr_ptr, L,
        STEPS: tl.constexpr, BL: tl.constexpr, PALETTE: tl.constexpr, PRM: tl.constexpr = False,
    ):
        b = tl.program_id(0)
        i = tl.arange(0, BL)
        valid = i < L
        base = b * L + i
        cr = tl.load(col_ptr + base * 3 + 0, mask=valid, other=0.0)
        cg = tl.load(col_ptr + base * 3 + 1, mask=valid, other=0.0)
        cb = tl.load(col_ptr + base * 3 + 2, mask=valid, other=0.0)
        td = tl.load(td_ptr + base, mask=valid, other=1.0)
        run_thick = tl.load(rt_ptr + base, mask=valid, other=1.0)
        cont = tl.load(cont_ptr + base, mask=valid, other=0.0)
        bgr = tl.load(bg_ptr + b * bg_stride + 0)
        bgg = tl.load(bg_ptr + b * bg_stride + 1)
        bgb = tl.load(bg_ptr + b * bg_stride + 2)
        scr = scr_ptr + b * BL
        reach = tl.maximum(0.1 * td, 1e-9)
        ir = 1.0 / tl.sqrt(reach)
        w_dark = tl.minimum(tl.maximum((1.0 - td) / 0.4, 0.0), 1.0)
        layer_light = _lightness(cr, cg, cb)
        darker = layer_light < 0.5
        bg_light = _lightness(bgr, bgg, bgb)
        dark = layer_light < -1.0  # all False
        k = tl.zeros([BL], dtype=tl.float32) + 1.0
        for _ in range(STEPS):
            w = tl.where(dark, w_dark, 1.0)
            cov = _cov(run_thick, ir, 1.0 / tl.sqrt(tl.sqrt(k * reach)), w)
            cov_prev = cont * _shift_up(cov, scr, i, valid, 0.0)
            op = tl.minimum(tl.maximum((cov - cov_prev) / tl.maximum(1.0 - cov_prev, 1e-6), 0.0), 1.0)
            sr, sg, sb = _stack_after(cr, cg, cb, op, bgr, bgg, bgb, valid)
            light_below = _shift_up(_lightness(sr, sg, sb), scr, i, valid, bg_light)
            dark = (light_below > layer_light) & darker
            denom = 1.0 - (light_below - layer_light)
            k = tl.where(denom > 0.0, light_below / tl.maximum(denom, 1e-12), 1.0)
            k = tl.maximum(k, 1.0)
        tl.store(k_ptr + base, k, mask=valid)
        tl.store(dark_ptr + base, dark.to(tl.float32), mask=valid)
        if PRM:
            # The continuous composite's per-layer table (_ContComposite):
            # colour, 1/sqrt(reach), 1/sqrt(slow reach), w, continues-run.
            pp = pal_ptr + base * 8
            tl.store(pp + 0, cr, mask=valid)
            tl.store(pp + 1, cg, mask=valid)
            tl.store(pp + 2, cb, mask=valid)
            tl.store(pp + 3, ir, mask=valid)
            tl.store(pp + 4, 1.0 / tl.sqrt(tl.sqrt(k * reach)), mask=valid)
            tl.store(pp + 5, tl.where(dark, w_dark, 1.0), mask=valid)
            tl.store(pp + 6, cont, mask=valid)
            tl.store(pp + 7, 0.0, mask=valid)
        if PALETTE:
            w = tl.where(dark, w_dark, 1.0)
            cov = _cov(run_thick, ir, 1.0 / tl.sqrt(tl.sqrt(k * reach)), w)
            cov_prev = cont * _shift_up(cov, scr, i, valid, 0.0)
            op = tl.minimum(tl.maximum((cov - cov_prev) / tl.maximum(1.0 - cov_prev, 1e-6), 0.0), 1.0)
            sr, sg, sb = _stack_after(cr, cg, cb, op, bgr, bgg, bgb, valid)
            pb = pal_ptr + b * (L + 1) * 3
            tl.store(pb + 0, bgr)
            tl.store(pb + 1, bgg)
            tl.store(pb + 2, bgb)
            tl.store(pb + (i + 1) * 3 + 0, sr, mask=valid)
            tl.store(pb + (i + 1) * 3 + 1, sg, mask=valid)
            tl.store(pb + (i + 1) * 3 + 2, sb, mask=valid)


def _run_geometry(run_start: torch.Tensor, h: float):
    """(run thickness, continues-the-run flag) per layer, fp32, [...,L]."""
    idx = torch.arange(run_start.shape[-1], device=run_start.device)
    return (idx - run_start + 1).to(torch.float32) * h, (run_start < idx).to(torch.float32)


def coverage_fixed_point(colors, tds, run_start, background, h: float, steps: int = 8, palette: bool = False):
    """Batched lightness fixed point: colors [B,L,3], tds [B,L], run_start
    [B,L], background [3] or [B,3]. Returns (k [B,L], dark [B,L] bool) and,
    with ``palette``, the [B,L+1,3] stack colour after 0..L layers (0..1)."""
    B, L = tds.shape
    dev = tds.device
    rt, cont = _run_geometry(run_start, h)
    bg = background.detach().to(torch.float32).contiguous()
    bg_stride = 3 if bg.dim() == 2 else 0
    BL = max(16, triton.next_power_of_2(L))
    k = torch.empty(B, L, device=dev, dtype=torch.float32)
    dark = torch.empty(B, L, device=dev, dtype=torch.float32)
    pal = torch.empty(B, L + 1, 3, device=dev, dtype=torch.float32) if palette else k
    scr = torch.empty(B, BL, device=dev, dtype=torch.float32)
    _coverage_fp_kernel[(B,)](
        colors.detach().to(torch.float32).contiguous(), tds.detach().to(torch.float32).contiguous(),
        rt.contiguous(), cont.contiguous(), bg, bg_stride, k, dark, pal, scr, L,
        STEPS=steps, BL=BL, PALETTE=palette, num_warps=1 if BL <= 128 else 4,
    )
    if palette:
        return k, dark > 0.5, pal
    return k, dark > 0.5


def coverage_params(layer_colors, layer_TDs, run_start, background, h: float):
    """``layer_coverage_params`` with the fixed point in one kernel; the
    returned (reach, slow_reach, w) are differentiable in ``layer_TDs``
    exactly as there (the lightness is a constant for gradients)."""
    reach = torch.clamp(0.1 * layer_TDs, min=1e-9)
    w_dark = torch.clamp((1.0 - layer_TDs) / 0.4, 0.0, 1.0)
    k, dark = coverage_fixed_point(layer_colors[None], layer_TDs[None], run_start[None], background, h)
    slow_reach = torch.sqrt(k[0] * reach)
    w = torch.where(dark[0], w_dark, torch.ones_like(w_dark))
    return reach, slow_reach, w


def cont_coverage_params(p_mat, colors_f32, tds_f32, background, h: float):
    """``_cont_coverage_params`` (runs from each layer's most likely
    material) with the fused fixed point."""
    mat_idx = torch.argmax(p_mat, dim=1)
    change = torch.ones_like(mat_idx, dtype=torch.bool)
    change[1:] = mat_idx[1:] != mat_idx[:-1]
    run_start = run_starts(change)
    reach, slow_reach, cov_w = coverage_params(colors_f32, tds_f32, run_start, background, h)
    return reach, slow_reach, cov_w, run_start


def stack_palettes(colors: torch.Tensor, tds: torch.Tensor, background, h: float) -> torch.Tensor:
    """``batched_stack_palette``: [B,L+1,3] (0..1) colour of B stacks printed
    up to 0..L full layers (runs from equal colour and TD)."""
    B, L = tds.shape
    colors = colors.to(torch.float32)
    tds = tds.to(torch.float32).clamp(1e-8, 1e8)
    change = torch.ones(B, L, dtype=torch.bool, device=tds.device)
    change[:, 1:] = (colors[:, 1:] != colors[:, :-1]).any(-1) | (tds[:, 1:] != tds[:, :-1])
    idx = torch.arange(L, device=tds.device).view(1, L)
    run_start = torch.cummax(torch.where(change, idx, torch.zeros_like(idx)), dim=1)[0]
    return coverage_fixed_point(colors, tds, run_start, background, h, palette=True)[2]


# --------------------------------------------------------------------------
# ICM class update of HeightAssign.assign_heights: every pixel of one
# (y mod 2, x mod 2) class takes the height k minimising
# err[k] + s * sum_q w_q |k - z_q| over its 8 neighbours (replicate border),
# in place - pixels of a class are never neighbours of each other.
# --------------------------------------------------------------------------

if _HAS_TRITON:

    @triton.jit
    def _icm_class_kernel(err_ptr, z_ptr, H, W, h, w, K, a, b, s, BLOCK: tl.constexpr):
        pid = tl.program_id(0)
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        valid = offs < h * w
        cy = offs // w
        cx = offs % w
        y = a + 2 * cy
        x = b + 2 * cx
        ym = tl.maximum(y - 1, 0)
        yp = tl.minimum(y + 1, H - 1)
        xm = tl.maximum(x - 1, 0)
        xp = tl.minimum(x + 1, W - 1)
        z_u = tl.load(z_ptr + ym * W + x, mask=valid, other=0).to(tl.float32)
        z_d = tl.load(z_ptr + yp * W + x, mask=valid, other=0).to(tl.float32)
        z_l = tl.load(z_ptr + y * W + xm, mask=valid, other=0).to(tl.float32)
        z_r = tl.load(z_ptr + y * W + xp, mask=valid, other=0).to(tl.float32)
        z_ul = tl.load(z_ptr + ym * W + xm, mask=valid, other=0).to(tl.float32)
        z_ur = tl.load(z_ptr + ym * W + xp, mask=valid, other=0).to(tl.float32)
        z_dl = tl.load(z_ptr + yp * W + xm, mask=valid, other=0).to(tl.float32)
        z_dr = tl.load(z_ptr + yp * W + xp, mask=valid, other=0).to(tl.float32)
        best = tl.full([BLOCK], float("inf"), tl.float32)
        best_k = tl.zeros([BLOCK], dtype=tl.int64)
        e_row = err_ptr + offs.to(tl.int64) * K
        for k in range(0, K):
            kf = k * 1.0
            nbc = (
                tl.abs(kf - z_u) + tl.abs(kf - z_d) + tl.abs(kf - z_l) + tl.abs(kf - z_r)
                + 0.7071 * (tl.abs(kf - z_ul) + tl.abs(kf - z_ur) + tl.abs(kf - z_dl) + tl.abs(kf - z_dr))
            )
            c = tl.load(e_row + k, mask=valid, other=0.0) + s * nbc
            take = c < best
            best = tl.where(take, c, best)
            best_k = tl.where(take, k, best_k)
        tl.store(z_ptr + y * W + x, best_k, mask=valid)


def icm_class_update(err_c: torch.Tensor, z: torch.Tensor, a: int, b: int, s: float) -> None:
    """In-place ICM update of class (a, b) of ``z`` (int64 [H,W]) with the
    class's error slice ``err_c`` [h,w,K]."""
    H, W = z.shape
    h, w, K = err_c.shape
    BLOCK = 128
    _icm_class_kernel[(triton.cdiv(h * w, BLOCK),)](
        err_c, z, H, W, h, w, K, a, b, float(s), BLOCK=BLOCK,
    )


# --------------------------------------------------------------------------
# 3x3 median (replicate border) of an integer height map, and the spike mask
# built on it (PixelHeightRefine._median3 / spike_mask): a 9-input sorting
# network per pixel instead of unfold + sort.
# --------------------------------------------------------------------------

if _HAS_TRITON:

    @triton.jit
    def _cs(a, b):
        return tl.minimum(a, b), tl.maximum(a, b)

    @triton.jit
    def _median9(p0, p1, p2, p3, p4, p5, p6, p7, p8):
        # Paeth's 19-exchange median-of-9 network.
        p1, p2 = _cs(p1, p2)
        p4, p5 = _cs(p4, p5)
        p7, p8 = _cs(p7, p8)
        p0, p1 = _cs(p0, p1)
        p3, p4 = _cs(p3, p4)
        p6, p7 = _cs(p6, p7)
        p1, p2 = _cs(p1, p2)
        p4, p5 = _cs(p4, p5)
        p7, p8 = _cs(p7, p8)
        p0, p3 = _cs(p0, p3)
        p5, p8 = _cs(p5, p8)
        p4, p7 = _cs(p4, p7)
        p3, p6 = _cs(p3, p6)
        p1, p4 = _cs(p1, p4)
        p2, p5 = _cs(p2, p5)
        p4, p7 = _cs(p4, p7)
        p4, p2 = _cs(p4, p2)
        p6, p4 = _cs(p6, p4)
        p4, p2 = _cs(p4, p2)
        return p4

    @triton.jit
    def _median3_kernel(z_ptr, out_ptr, spike_ptr, H, W, thr, max_out, SPIKE: tl.constexpr, BLOCK: tl.constexpr):
        pid = tl.program_id(0)
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        valid = offs < H * W
        y = offs // W
        x = offs % W
        ym = tl.maximum(y - 1, 0) * W
        yc = y * W
        yp = tl.minimum(y + 1, H - 1) * W
        xm = tl.maximum(x - 1, 0)
        xp = tl.minimum(x + 1, W - 1)
        v0 = tl.load(z_ptr + ym + xm, mask=valid, other=0).to(tl.float32)
        v1 = tl.load(z_ptr + ym + x, mask=valid, other=0).to(tl.float32)
        v2 = tl.load(z_ptr + ym + xp, mask=valid, other=0).to(tl.float32)
        v3 = tl.load(z_ptr + yc + xm, mask=valid, other=0).to(tl.float32)
        v4 = tl.load(z_ptr + yc + x, mask=valid, other=0).to(tl.float32)
        v5 = tl.load(z_ptr + yc + xp, mask=valid, other=0).to(tl.float32)
        v6 = tl.load(z_ptr + yp + xm, mask=valid, other=0).to(tl.float32)
        v7 = tl.load(z_ptr + yp + x, mask=valid, other=0).to(tl.float32)
        v8 = tl.load(z_ptr + yp + xp, mask=valid, other=0).to(tl.float32)
        med = _median9(v0, v1, v2, v3, v4, v5, v6, v7, v8)
        if SPIKE:
            cnt = (
                ((v0 - med) >= thr).to(tl.int32) + ((v1 - med) >= thr).to(tl.int32)
                + ((v2 - med) >= thr).to(tl.int32) + ((v3 - med) >= thr).to(tl.int32)
                + ((v4 - med) >= thr).to(tl.int32) + ((v5 - med) >= thr).to(tl.int32)
                + ((v6 - med) >= thr).to(tl.int32) + ((v7 - med) >= thr).to(tl.int32)
                + ((v8 - med) >= thr).to(tl.int32)
            )
            sp = ((v4 - med) >= thr) & (cnt <= max_out) & (cnt > 0)
            tl.store(spike_ptr + offs, sp.to(tl.int8), mask=valid)
        else:
            tl.store(out_ptr + offs, med, mask=valid)


def median3(z: torch.Tensor) -> torch.Tensor:
    """3x3 median (replicate border) of an integer map, same dtype."""
    H, W = z.shape
    zc = z.contiguous()
    out = torch.empty(H, W, device=z.device, dtype=torch.float32)
    _median3_kernel[(triton.cdiv(H * W, 256),)](zc, out, out, H, W, 0.0, 0, SPIKE=False, BLOCK=256)
    return out.to(z.dtype)


def spike_mask(z: torch.Tensor, threshold: float, max_outliers: int = 2) -> torch.Tensor:
    """PixelHeightRefine.spike_mask, fused."""
    H, W = z.shape
    zc = z.contiguous()
    out = torch.empty(H, W, device=z.device, dtype=torch.int8)
    _median3_kernel[(triton.cdiv(H * W, 256),)](
        zc, out, out, H, W, float(threshold), int(max_outliers), SPIKE=True, BLOCK=256
    )
    return out.bool()


# --------------------------------------------------------------------------
# Palette proxy distances (PixelHeightRefine.PaletteProxy): for B stacks'
# Lab palettes [B,n,3] and N weighted target points, sum_i w_i min_k
# |t_i - p_k|^2. The torch version builds the whole [B,N,n] distance tensor;
# here each program keeps a block of points in registers and walks the
# palette, writing one partial sum per block (summed afterwards).
# --------------------------------------------------------------------------

if _HAS_TRITON:

    @triton.jit
    def _proxy_kernel(pal_ptr, t_ptr, w_ptr, part_ptr, N, n, nblk, BLOCK: tl.constexpr):
        b = tl.program_id(0)
        blk = tl.program_id(1)
        offs = blk * BLOCK + tl.arange(0, BLOCK)
        valid = offs < N
        t0 = tl.load(t_ptr + offs * 3 + 0, mask=valid, other=0.0)
        t1 = tl.load(t_ptr + offs * 3 + 1, mask=valid, other=0.0)
        t2 = tl.load(t_ptr + offs * 3 + 2, mask=valid, other=0.0)
        best = tl.full([BLOCK], float("inf"), tl.float32)
        p = pal_ptr + b * n * 3
        for k in range(0, n):
            d0 = t0 - tl.load(p + k * 3 + 0)
            d1 = t1 - tl.load(p + k * 3 + 1)
            d2 = t2 - tl.load(p + k * 3 + 2)
            best = tl.minimum(best, d0 * d0 + d1 * d1 + d2 * d2)
        w = tl.load(w_ptr + offs, mask=valid, other=0.0)
        tl.store(part_ptr + b * nblk + blk, tl.sum(tl.where(valid, best * w, 0.0), axis=0))


def proxy_losses(pal_lab: torch.Tensor, t_lab: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """[B] sum_i w_i min_k |t_i - pal_lab[b,k]|^2 / 3."""
    B, n, _ = pal_lab.shape
    N = t_lab.shape[0]
    BLOCK = 256
    nblk = triton.cdiv(N, BLOCK)
    part = torch.empty(B, nblk, device=pal_lab.device, dtype=torch.float32)
    _proxy_kernel[(B, nblk)](
        pal_lab.to(torch.float32).contiguous(), t_lab, w, part, N, n, nblk, BLOCK=BLOCK,
    )
    return part.sum(1) / 3.0
