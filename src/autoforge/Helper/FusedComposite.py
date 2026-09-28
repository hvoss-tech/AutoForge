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
        tlab_ptr, wt_ptr, smooth,
        ERR: tl.constexpr, BLOCK: tl.constexpr,
    ):
        pid = tl.program_id(0)
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
        for l in range(0, L):
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
        float(h), 1.0 / float(h), out, out, 0.0, ERR=False, BLOCK=BLOCK,
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
        float(h), 1.0 / float(h), target_lab.contiguous(), weights.contiguous(), float(smooth),
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
        h, scale = ctx.h, ctx.scale
        H, W = z.shape
        L = prm.shape[0]
        N = H * W
        g = (grad_out.float() * 255.0).contiguous()
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
        return (
            dz, red[:L, 0:3], red[:L, 3], red[:L, 4], red[:L, 5], None, red[L, 0:3], None, None,
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
    tl32, w, wsum = _loss_inputs(optimizer)
    params = layer_params(cols, tds, optimizer.background, optimizer.h)
    e = heights_error(z, params, background_tensor(optimizer.background), optimizer.h, tl32, w, 0.0)
    return e.sum() / (3.0 * wsum)


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
        STEPS: tl.constexpr, BL: tl.constexpr, PALETTE: tl.constexpr,
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
