import os
import threading
from collections import OrderedDict
from contextlib import contextmanager
from typing import Final, Optional

import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from autoforge.Helper.AmpUtils import get_selected_autocast

# --------------------------------------------------------------------------
# Ablation switches (paper experiments only).
#
# Each flag is read once, at import time, from an environment variable that
# is unset in normal/production use - so with no env var set, every branch
# below compiles to exactly the original behavior (TorchScript treats these
# as compile-time constants and dead-code-eliminates the unused branch).
# Set via e.g. AUTOFORGE_ABLATION_ROUNDING=hard before invoking the CLI; see
# paper/experiments/ for the scripts that exercise these.
# --------------------------------------------------------------------------
ABLATION_NO_ADAPTIVE_ROUNDING: bool = (
    os.environ.get("AUTOFORGE_ABLATION_ROUNDING", "") == "hard"
)
ABLATION_BEER_LAMBERT_OPACITY: bool = (
    os.environ.get("AUTOFORGE_ABLATION_OPACITY", "") == "beer_lambert"
)
ABLATION_NO_GUMBEL_NOISE: bool = (
    os.environ.get("AUTOFORGE_ABLATION_GUMBEL", "") == "off"
)
# TorchScript can't close over a plain Python global inside a compiled
# function body, so the flags above are threaded through as ordinary boolean
# parameters instead, defaulted to these module-level values (evaluated once,
# at import time, exactly like the flags themselves) - every call site below
# that doesn't explicitly pass the parameter gets the ablation behavior
# selected by the environment at process start.


@torch.jit.script
def adaptive_round(
    x: torch.Tensor,
    tau: float,
    high_tau: float,
    low_tau: float,
    temp: float,
    no_adaptive: bool = ABLATION_NO_ADAPTIVE_ROUNDING,
) -> torch.Tensor:
    """
    Smooth rounding based on temperature 'tau'.

    Args:
        x (torch.Tensor): The input tensor to be rounded.
        tau (float): The current temperature parameter.
        high_tau (float): The high threshold for the temperature.
        low_tau (float): The low threshold for the temperature.
        temp (float): The temperature parameter for the sigmoid function.
        no_adaptive (bool): "Without Adaptive Rounding" ablation - when set,
            always hard-round regardless of tau (see
            ``ABLATION_NO_ADAPTIVE_ROUNDING`` above).

    Returns:
        torch.Tensor: The rounded tensor.
    """
    if no_adaptive:
        return torch.round(x)
    if tau <= low_tau:
        return torch.round(x)
    elif tau >= high_tau:
        floor_val = torch.floor(x)
        diff = x - floor_val
        soft_round = floor_val + torch.sigmoid((diff - 0.5) / temp)
        return soft_round
    else:
        ratio = (tau - low_tau) / (high_tau - low_tau)
        hard_round = torch.round(x)
        floor_val = torch.floor(x)
        diff = x - floor_val
        soft_round = floor_val + torch.sigmoid((diff - 0.5) / temp)
        return ratio * soft_round + (1 - ratio) * hard_round


# A deterministic random generator that mimics torch.rand_like.
@torch.jit.script
def deterministic_rand_like(tensor: torch.Tensor, seed: int) -> torch.Tensor:
    """
    Generate a deterministic random tensor that mimics torch.rand_like.

    Args:
        tensor (torch.Tensor): The input tensor whose shape and device will be used.
        seed (int): The seed for the deterministic random generator.

    Returns:
        torch.Tensor: A tensor with the same shape as the input tensor, filled with deterministic random values.
    """
    # Compute the total number of elements.
    n: int = 1
    for d in tensor.shape:
        n = n * d
    # Create a 1D tensor of indices [0, 1, 2, ..., n-1].
    indices = torch.arange(n, dtype=torch.float32, device=tensor.device)
    # Offset the indices by the seed.
    indices = indices + seed
    # Use a simple hash function: sin(x)*constant, then take the fractional part.
    r = torch.sin(indices) * 43758.5453123
    r = r - torch.floor(r)
    # Reshape to the shape of the original tensor.
    return r.view(tensor.shape)


@torch.jit.script
def deterministic_gumbel_softmax(
    logits: torch.Tensor, tau: float, hard: bool, rng_seed: int
) -> torch.Tensor:
    """
    Apply the Gumbel-Softmax trick in a deterministic manner using a fixed random seed.

    Args:
        logits (torch.Tensor): The input logits tensor.
        tau (float): The temperature parameter for the Gumbel-Softmax.
        hard (bool): If True, the output will be one-hot encoded.
        rng_seed (int): The seed for the deterministic random generator.

    Returns:
        torch.Tensor: The resulting tensor after applying the Gumbel-Softmax trick.
    """
    eps: float = 1e-20
    # Instead of torch.rand_like(..., generator=...), use our deterministic_rand_like.
    U = deterministic_rand_like(logits, rng_seed)
    # Compute Gumbel noise.
    gumbel_noise = -torch.log(-torch.log(U + eps) + eps)
    y = (logits + gumbel_noise) / tau
    y_soft = F.softmax(y, dim=-1)
    if hard:
        # Compute one-hot using argmax and scatter.
        index = torch.argmax(y_soft, dim=-1, keepdim=True)
        y_hard = torch.zeros_like(y_soft).scatter_(-1, index, 1.0)
        # Use the straight-through estimator.
        y = (y_hard - y_soft).detach() + y_soft
    return y


@torch.jit.script
def deterministic_gumbel_noise(seeds: torch.Tensor, n_mat: int) -> torch.Tensor:
    """Gumbel noise for a batch of ``deterministic_rand_like`` draws at once.

    ``deterministic_rand_like(x_of_shape_[M], seed)`` is
    ``frac(sin(arange(M) + seed) * 43758.5453123)``; this computes that for
    every seed in ``seeds`` (int64, any shape - flattened to [N]) and returns
    the corresponding Gumbel noise, shape [N, n_mat].

    Bit-exact vs. the scalar path: the seed is kept in int64 through any
    offset arithmetic and converted to fp32 exactly once, matching
    ``torch.arange(M, dtype=torch.float32) + <python int seed>``.
    """
    m_idx = torch.arange(n_mat, dtype=torch.float32, device=seeds.device).view(1, n_mat)
    r = torch.sin(m_idx + seeds.to(torch.float32).view(-1, 1)) * 43758.5453123
    U = r - torch.floor(r)
    eps: float = 1e-20
    return -torch.log(-torch.log(U + eps) + eps)


@torch.jit.script
def batched_layer_material_indices(
    global_logits: torch.Tensor,  # [L, M]
    tau: float,
    seed_base: int,
) -> torch.Tensor:
    """Vectorized replacement for the per-layer material-selection loop.

    Exactly reproduces
    ``argmax(deterministic_gumbel_softmax(global_logits[j], tau, True, seed_base + j))``
    for every layer ``j``, but in a single kernel instead of ``L`` separate
    tiny ones. The per-layer loop was measured at ~7.3ms of
    ``composite_image_disc``'s ~9.8ms forward pass (L=75) - pure Python/launch
    overhead, since each iteration only touches an [M]-sized tensor.

    The ``softmax`` is kept rather than taking the argmax of the raw
    ``(logits + gumbel) / tau`` scores. It is mathematically redundant
    (softmax is monotonic) but preserves the saturation/NaN behaviour at very
    small tau on large-scale logits that the original path had - see
    ``material_select_from_logits``'s docstring in PruningHelper.

    Returns:
        torch.Tensor: int64 [L] of chosen material indices.
    """
    L = int(global_logits.shape[0])
    M = int(global_logits.shape[1])
    seeds = torch.arange(L, dtype=torch.int64, device=global_logits.device) + seed_base
    gumbel = deterministic_gumbel_noise(seeds, M)  # [L, M]
    y_soft = F.softmax((global_logits + gumbel) / tau, dim=-1)
    return torch.argmax(y_soft, dim=-1)


@torch.jit.script
def bleed_layer_effect(mask: torch.Tensor, strength: float = 0.1) -> torch.Tensor:
    """
    Applies a simple 2D 3x3 average blur to simulate edge bleeding.

    Args:
        mask (torch.Tensor): [H,W] or [L,H,W] tensor of masks.
        strength (float): Amount of the bleed to spread to neighbors.

    Returns:
        torch.Tensor: Mask with neighboring bleed added.
    """
    if mask.dim() == 2:
        mask = mask.unsqueeze(0)  # [1,H,W]
    L, H, W = mask.shape

    # 3x3 average kernel over the 8 neighbours (centre excluded).
    #
    # Built entirely on-device. The obvious `torch.tensor([[1,1,1],[1,0,1],
    # [1,1,1]], device=mask.device)` spelling allocates on the CPU and issues
    # a host->device copy on *every* call - this function runs once per
    # composite, i.e. once per training step and once per pruning candidate.
    # It is also an outright blocker for CUDA graph capture ("Cannot copy
    # between CPU and CUDA tensors during CUDA graph capture"). Values are
    # identical: 1/8 in the eight neighbour taps, 0 in the centre.
    taps = torch.full((9,), 0.125, dtype=mask.dtype, device=mask.device)
    centre = torch.arange(9, device=mask.device) == 4
    kernel = torch.where(centre, torch.zeros_like(taps), taps).view(1, 1, 3, 3)

    # Apply conv2d to each layer independently
    blurred = F.conv2d(mask.unsqueeze(1), kernel, padding=1, groups=1).squeeze(
        1
    )  # [L,H,W]

    # Combine original mask with bleed from neighbors
    return mask + strength * blurred


# --------------------------------------------------------------------------
# Layer opacity model
#
# Consecutive layers of the same material form a run. A run covers whatever
# lies beneath it by ``coverage(run thickness)``: the colour seen on top is a
# straight sRGB blend from the colour under the run towards the material's
# colour. The compositing below is per layer (front to back), so every layer
# gets the opacity that turns the run's coverage so far into exactly that
# blend: ``1 - (1 - coverage_k) / (1 - coverage_{k-1})``.
# --------------------------------------------------------------------------

# Fixed-point refinements of the stack lightness used by the dark-over-light
# correction (``layer_coverage_params``). A static count keeps the composite
# free of host syncs and CUDA-graph capturable.
LIGHTNESS_REFINE_STEPS: Final[int] = 8


@torch.jit.script
def srgb_lightness(rgb: torch.Tensor) -> torch.Tensor:
    """CIELAB lightness L*/100 (D65 white) of sRGB colours in [0, 1].

    ``rgb`` is [..., 3]; returns [...]. Colours are quantised to 8 bits per
    channel first, as a colour stored as a hex code would be.
    """
    c = torch.floor(torch.clamp(rgb, 0.0, 1.0) * 255.0 + 1e-4) / 255.0
    lin = torch.where(
        c <= 0.04045,
        c / 12.92,
        ((torch.clamp(c, min=0.04045) + 0.055) / 1.055) ** 2.4,
    )
    y = lin[..., 0] * 0.2126 + lin[..., 1] * 0.7152 + lin[..., 2] * 0.0722
    f = torch.where(
        y > 0.008856,
        torch.clamp(y, min=0.008856) ** (1.0 / 3.0),
        7.787 * y + 0.137931,
    )
    return (116.0 * f - 16.0) / 100.0


@torch.jit.script
def run_starts(change: torch.Tensor) -> torch.Tensor:
    """[L] bool "layer k starts a new run" -> [L] int64 index of the first
    layer of the run each layer belongs to (layer 0 always starts one)."""
    idx = torch.arange(change.shape[0], dtype=torch.int64, device=change.device)
    starts = torch.where(change, idx, torch.zeros_like(idx))
    return torch.cummax(starts, dim=0)[0]


@torch.jit.script
def material_run_starts(
    layer_colors: torch.Tensor, layer_TDs: torch.Tensor
) -> torch.Tensor:
    """``run_starts`` for a hard per-layer material assignment: a new run
    begins wherever the colour or TD differs from the layer below."""
    differs = (layer_colors[1:] != layer_colors[:-1]).any(dim=1) | (
        layer_TDs[1:] != layer_TDs[:-1]
    )
    first = torch.ones(1, dtype=torch.bool, device=layer_colors.device)
    return run_starts(torch.cat([first, differs], dim=0))


@torch.jit.script
def _coverage(
    run_thick: torch.Tensor,
    reach: torch.Tensor,
    slow_reach: torch.Tensor,
    w: torch.Tensor,
) -> torch.Tensor:
    """How much of what lies beneath a run it covers at ``run_thick``.

    ``sqrt(thickness / reach)`` up to full coverage at ``reach``, blended
    with the same curve over ``slow_reach`` by ``w`` (see
    ``layer_coverage_params``). The lower clamp keeps sqrt's gradient finite
    at zero thickness; the square root is shared by both curves.
    """
    root = torch.sqrt(torch.clamp(run_thick, min=1e-12))
    plain = torch.clamp(root * torch.rsqrt(reach), max=1.0)
    slow = torch.clamp(root * torch.rsqrt(slow_reach), max=1.0)
    return slow + w * (plain - slow)


@torch.jit.script
def _stack_masks(n: int, device: torch.device):
    """Layer-order masks for ``_stack_after_layers`` over ``n`` = L+1 slots
    (slot 0 is the background): ``above[j,i]`` - slot i lies above slot j;
    ``covered[j,i]`` - slot j is at or below slot i."""
    i = torch.arange(n, device=device)
    above = i.view(1, n) > i.view(n, 1)
    covered = above | (i.view(1, n) == i.view(n, 1))
    return above, covered


@torch.jit.script
def _stack_after_layers(
    opac: torch.Tensor,  # [L] per-layer opacity at full layer thickness
    colors_ext: torch.Tensor,  # [L+1,3] background, then the layers bottom to top
    above: torch.Tensor,  # [L+1,L+1] bool, see _stack_masks
    covered: torch.Tensor,  # [L+1,L+1] bool, see _stack_masks
    one: torch.Tensor,  # [1] 1.0
    zero: torch.Tensor,  # [1] 0.0
) -> torch.Tensor:
    """[L+1,3] colour of the stack after 0..L fully printed layers.

    ``one``/``zero`` are passed in so repeated calls do not materialise the
    constants on the device every time."""
    n = int(opac.shape[0]) + 1
    o_ext = torch.cat([one, opac], dim=0)
    trans = torch.where(above, (1.0 - o_ext).view(1, n).expand(n, n), one)
    # through[j,i] = transmittance of layers j+1..i
    through = torch.cumprod(trans, dim=1)
    weight = torch.where(covered, through * o_ext.view(n, 1), zero)
    return weight.t() @ colors_ext


@torch.jit.script
def layer_coverage_params(
    layer_colors: torch.Tensor,  # [L,3] bottom to top
    layer_TDs: torch.Tensor,  # [L]
    run_start: torch.Tensor,  # [L] int64, see run_starts
    background: torch.Tensor,  # [3]
    h: float,
    refine_steps: int = LIGHTNESS_REFINE_STEPS,
):
    """Per-layer ``(reach, slow_reach, w)`` for ``_coverage``.

    A run reaches full coverage at a tenth of its transmission distance
    (``reach``), so a single thin layer already covers a good part of the
    stack below.

    A material darker than mid-grey laid over a lighter stack covers more
    slowly: its reach is stretched to ``sqrt(k * reach)`` with ``k`` growing
    with the lightness gap, and near-opaque materials (TD below 1) fade back
    to the plain curve through ``w``. Which layers that applies to depends on
    the lightness of the stack below them, which in turn depends on the
    opacities below - that is solved by fixed-point iteration over the
    [L]-sized stack (after n refinements the bottom n layers are exact).
    The lightness itself is treated as a constant for gradients.
    """
    reach = torch.clamp(0.1 * layer_TDs, min=1e-9)
    w_dark = torch.clamp((1.0 - layer_TDs) / 0.4, 0.0, 1.0)

    colors = layer_colors.detach().to(torch.float32)
    tds = layer_TDs.detach().to(torch.float32)
    reach_c = torch.clamp(0.1 * tds, min=1e-9)
    w_dark_c = torch.clamp((1.0 - tds) / 0.4, 0.0, 1.0)
    bg = background.detach().to(torch.float32)
    layer_light = srgb_lightness(colors)
    colors_ext = torch.cat([bg.view(1, 3), colors], dim=0)
    above, covered = _stack_masks(int(colors.shape[0]) + 1, colors.device)

    # Device constants reused by every pass (a Python scalar operand would be
    # materialised on the device on each use).
    one = torch.ones(1, dtype=torch.float32, device=colors.device)
    zero = torch.zeros(1, dtype=torch.float32, device=colors.device)

    idx = torch.arange(colors.shape[0], dtype=torch.int64, device=colors.device)
    run_thick = (idx - run_start + 1).to(torch.float32) * h
    continues = run_start < idx
    dark = torch.zeros_like(continues)
    k = torch.ones_like(run_thick)
    darker_than_mid = layer_light < 0.5
    for _ in range(refine_steps):
        w = torch.where(dark, w_dark_c, one)
        cov = _coverage(run_thick, reach_c, torch.sqrt(k * reach_c), w)
        cov_prev = torch.where(continues, torch.cat([zero, cov[:-1]], dim=0), zero)
        opac = torch.clamp((cov - cov_prev) / torch.clamp(1.0 - cov_prev, min=1e-6), 0.0, 1.0)
        light_below = srgb_lightness(
            _stack_after_layers(opac, colors_ext, above, covered, one, zero)[:-1]
        )
        dark = (light_below > layer_light) & darker_than_mid
        denom = 1.0 - (light_below - layer_light)
        k = torch.where(denom > 0.0, light_below / torch.clamp(denom, min=1e-12), one)
        k = torch.clamp(k, min=1.0)

    slow_reach = torch.sqrt(k * reach)
    w = torch.where(dark, w_dark, torch.ones_like(w_dark))
    return reach, slow_reach, w


@torch.jit.script
def _layer_opacity(
    eff_thick: torch.Tensor,  # [k,H,W] effective thickness, layers lo..lo+k-1 bottom to top
    reach: torch.Tensor,  # [k]
    slow_reach: torch.Tensor,  # [k]
    w: torch.Tensor,  # [k]
    run_start: torch.Tensor,  # [k] int64 (absolute layer indices)
    lo: int,
    h: float,
    carry_thick: torch.Tensor,  # [H,W] run thickness at layer lo-1
    carry_cov: torch.Tensor,  # [H,W] coverage at layer lo-1
    beer_lambert: bool = ABLATION_BEER_LAMBERT_OPACITY,
):
    """Per-layer opacity [k,H,W] (bottom to top) for a slice of the stack.

    Returns ``(opac, run_thick_top, coverage_top)``; the last two are the
    carries for the slice directly above. Computed in fp32 whatever the
    input dtype: the opacity is a ratio of ``1 - coverage`` terms that are
    tiny near full coverage.

    Under ``AUTOFORGE_ABLATION_OPACITY=beer_lambert`` the textbook
    single-scattering law ``1 - exp(-thickness / TD)`` replaces the model -
    the "Without Opacity Calibration" ablation.
    """
    eff = eff_thick.to(torch.float32)
    n = int(eff.shape[0])
    layer_idx = torch.arange(lo, lo + n, dtype=torch.int64, device=eff.device)
    if beer_lambert:
        opac = torch.clamp(1.0 - torch.exp(-eff / (10.0 * reach.view(-1, 1, 1))), 0.0, 1.0)
        return opac, carry_thick, carry_cov

    # Thickness of each layer's run up to and including the layer.
    csum = torch.cumsum(eff, dim=0)
    local_start = torch.clamp(run_start - lo, min=0)
    # Cumulative sum just below each run's first layer (0 at the slice bottom),
    # gathered straight from csum instead of from a zero-padded copy of it.
    before_run = csum.index_select(0, torch.clamp(local_start - 1, min=0)) * (
        local_start > 0
    ).to(torch.float32).view(-1, 1, 1)
    from_below = (run_start < lo).to(torch.float32).view(-1, 1, 1)
    run_thick = csum - before_run + from_below * carry_thick

    cov = _coverage(
        run_thick, reach.view(-1, 1, 1), slow_reach.view(-1, 1, 1), w.view(-1, 1, 1)
    )
    continues = (run_start < layer_idx).to(torch.float32).view(-1, 1, 1)
    cov_prev = torch.cat([carry_cov.unsqueeze(0), cov[:-1]], dim=0) * continues
    # 1 - (1 - cov) / (1 - cov_prev), written with one division.
    opac = torch.clamp((cov - cov_prev) / torch.clamp(1.0 - cov_prev, min=1e-6), 0.0, 1.0)
    # Scaled by how much of the layer is actually printed here (eff <= h),
    # so layers above a pixel's height stay fully transparent.
    opac = opac * (eff * (1.0 / h))
    return opac, run_thick[-1], cov[-1]


class _CumprodDim0(torch.autograd.Function):
    """``torch.cumprod(x, dim=0, dtype=torch.float32)`` with a sync-free backward.

    ATen's ``cumprod_backward`` unconditionally runs ``(input == 0).any().item()``
    to decide between its zero-free fast path and a general slow path. That
    ``.item()`` is a device->host copy, i.e. a **full CUDA sync inside the
    backward pass of every single training step** - it drains the pipeline and
    serializes CPU kernel-launch with GPU execution, on a step that is
    otherwise entirely launch-bound. It is also a hard blocker for CUDA graph
    capture ("operation not permitted when stream is capturing").

    The backward here is ATen's own zero-free formula,
    ``reverse_cumsum(grad * y) / x``, with the ``x == 0`` positions forced to
    zero instead of producing 0/0 = NaN. That substitution is safe *for this
    pipeline* rather than in general: ``x`` here is ``1 - opac`` where ``opac``
    comes straight out of a ``clamp(..., 0.0, 1.0)``, so ``x == 0`` implies the
    clamp saturated at its upper bound, and clamp's own backward multiplies
    that position's gradient by zero anyway. Writing 0 there produces the same
    downstream gradient as the true value would, while avoiding a NaN that
    ``0 * NaN`` would otherwise propagate.
    """

    @staticmethod
    def forward(ctx, x: torch.Tensor) -> torch.Tensor:
        y = torch.cumprod(x, dim=0, dtype=torch.float32)
        ctx.save_for_backward(x, y)
        return y

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        x, y = ctx.saved_tensors
        rev_cumsum = (grad_out * y).flip(0).cumsum(0).flip(0)
        # Where x[i] == 0 every y[j>=i] is 0 too, so rev_cumsum[i] is exactly
        # 0; substituting a 1 in the denominator therefore yields 0 rather
        # than 0/0 = NaN, without needing a separate select afterwards.
        grad_x = rev_cumsum / x.masked_fill(x == 0, 1.0)
        return grad_x.to(x.dtype)


@torch.jit.script
def _composite_cont_pre(
    pixel_height_logits: torch.Tensor,  # [H,W]
    global_logits: torch.Tensor,  # [L,M]
    tau_height: float,
    tau_global: float,
    h: float,
    max_layers: int,
    material_colors: torch.Tensor,  # [M,3]
    material_TDs: torch.Tensor,  # [M]
    background: torch.Tensor,  # [3]
    compute_dtype: Optional[torch.dtype] = None,
    gumbel_exp: Optional[torch.Tensor] = None,  # [L,M] Exponential(1) samples
    no_gumbel: bool = ABLATION_NO_GUMBEL_NOISE,
):
    """Everything in the continuous composite up to the per-layer opacity.

    The composite is cut into scripted pieces so that the opacity step can
    run under activation checkpointing and the ``cumprod`` can go through
    ``_CumprodDim0`` - neither can be called from TorchScript, and dropping
    ``@torch.jit.script`` from the whole composite costs ~40% (2.85ms ->
    3.98ms fwd+bwd at L=75, H=W=250).

    Returns ``(continuous_z, p_mat, colors_f32, tds_f32)``.
    """
    # 1. per-pixel continuous layer index
    pixel_height = (max_layers * h) * torch.sigmoid(pixel_height_logits)  # [H,W]
    continuous_z = pixel_height / h  # [H,W]
    continuous_z = adaptive_round(continuous_z, tau_height, 1.0, 0.0, 0.1)

    # 2. global material weights with Gumbel-Softmax
    #
    # ``gumbel_exp`` lets the caller supply the Exponential(1) draw that
    # F.gumbel_softmax would otherwise make internally. That is what keeps
    # the RNG *outside* a CUDA-graph-captured region: a captured graph gets
    # its own philox offset sequence on replay, which would silently give a
    # different noise stream than the eager path. Both branches compute the
    # identical expression - F.gumbel_softmax(logits, tau, hard=False) is
    # exactly softmax((logits + -log(Exponential(1))) / tau).
    if no_gumbel:
        # "Without Gumbel Softmax" ablation: plain temperature-scaled softmax,
        # no stochastic relaxation noise (isolates the noise's contribution
        # from the softmax reparameterization itself).
        p_mat = F.softmax(global_logits / tau_global, dim=1)  # [L,M]
    elif gumbel_exp is None:
        p_mat = F.gumbel_softmax(global_logits, tau_global, hard=False, dim=1)  # [L,M]
    else:
        p_mat = F.softmax(
            (global_logits + (-torch.log(gumbel_exp))) / tau_global, dim=1
        )  # [L,M]

    layer_colors = p_mat @ material_colors  # [L,3]
    layer_TDs = (p_mat @ material_TDs).clamp(1e-8, 1e8)  # [L]

    # fp32 copies for the per-layer coverage parameters (worked out below).
    colors_f32 = layer_colors
    tds_f32 = layer_TDs

    return continuous_z, p_mat, colors_f32, tds_f32


@torch.jit.script
def _print_mask_segment(
    continuous_z: torch.Tensor,  # [H,W]
    tau_height: float,
    h: float,
    max_layers: int,
    compute_dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:
    """[L,H,W] effective thickness of every layer (soft print mask with
    neighbour bleed, times the layer height). Run under activation
    checkpointing by ``composite_image_cont``: its input is a single [H,W]
    map, so recomputing it is cheaper than keeping its [L,H,W] tape."""
    # 3. soft print mask for all layers (layer 0 = bottom, layer L-1 = top)
    #    Small τ  -> large scale (steep transition)
    #    Large τ  -> small scale (smooth transition)
    eps = 1e-8
    scale = 10.0 / (tau_height + eps)
    layer_idx = torch.arange(
        max_layers, dtype=torch.float32, device=continuous_z.device
    ).view(-1, 1, 1)  # [L,1,1]
    p_print = torch.sigmoid(
        (continuous_z.unsqueeze(0) - (layer_idx + 0.5)) * scale
    )  # [L,H,W]

    # `@torch.jit.script` functions do not observe an ambient `torch.autocast`
    # context (verified empirically: output stayed float32 even inside an
    # active bf16 autocast region) - every large [L,H,W]/[L,H,W,3] tensor from
    # here down was silently computed and held in full fp32, roughly doubling
    # both memory and bandwidth versus the caller's intended precision. Cast
    # explicitly once here (after the precision-sensitive height/tau logic
    # above has already produced its fp32 layer mask) so the rest of the
    # per-layer compositing pipeline - the actual memory-dominant part - runs
    # in the caller-selected lower precision instead.
    if compute_dtype is not None:
        p_print = p_print.to(compute_dtype)

    # 4. thickness and opacity
    p_print_bleed = bleed_layer_effect(p_print, strength=0.1)  # [L,H,W]
    del p_print
    eff_thick = torch.clamp(p_print_bleed, 0.0, 1.0) * h
    del p_print_bleed
    return eff_thick


@torch.jit.script
def _cont_coverage_params(
    p_mat: torch.Tensor,  # [L,M]
    colors_f32: torch.Tensor,  # [L,3]
    tds_f32: torch.Tensor,  # [L]
    background: torch.Tensor,  # [3]
    h: float,
):
    """Runs (from each layer's most likely material) and coverage
    parameters for the continuous composite."""
    # Runs follow each layer's most likely material. The coverage parameters
    # are a long chain of tiny per-layer kernels, so they are issued after the
    # big [L,H,W] work above: the GPU is busy with that while they launch.
    mat_idx = torch.argmax(p_mat, dim=1)
    run_start = run_starts(
        torch.cat(
            [
                torch.ones(1, dtype=torch.bool, device=p_mat.device),
                mat_idx[1:] != mat_idx[:-1],
            ],
            dim=0,
        )
    )
    reach, slow_reach, cov_w = layer_coverage_params(
        colors_f32, tds_f32, run_start, background, h
    )

    return reach, slow_reach, cov_w, run_start


@torch.jit.script
def _run_thickness_segment(
    eff: torch.Tensor,  # [L,H,W] fp32 effective thickness
    run_start: torch.Tensor,
) -> torch.Tensor:
    """First part of ``_layer_opacity`` (whole stack, nothing carried in):
    [L,H,W] fp32 thickness of each layer's run up to and including it."""
    zero_hw = torch.zeros_like(eff[0])
    csum = torch.cumsum(eff, dim=0)
    local_start = torch.clamp(run_start, min=0)
    before_run = csum.index_select(0, torch.clamp(local_start - 1, min=0)) * (
        local_start > 0
    ).to(torch.float32).view(-1, 1, 1)
    from_below = (run_start < 0).to(torch.float32).view(-1, 1, 1)
    return csum - before_run + from_below * zero_hw


@torch.jit.script
def _coverage_segment(
    run_thick: torch.Tensor,  # [L,H,W] fp32, from _run_thickness_segment
    reach: torch.Tensor,
    slow_reach: torch.Tensor,
    cov_w: torch.Tensor,
) -> torch.Tensor:
    """Second part of ``_layer_opacity``: [L,H,W] fp32 run coverage."""
    return _coverage(
        run_thick,
        reach.view(-1, 1, 1),
        slow_reach.view(-1, 1, 1),
        cov_w.view(-1, 1, 1),
    )


@torch.jit.script
def _ratio_from_coverage(
    cov: torch.Tensor,  # [L,H,W] fp32, from _coverage_segment
    run_start: torch.Tensor,
) -> torch.Tensor:
    """Second part of ``_layer_opacity``: each layer's opacity before scaling
    by how much of it is printed, [L,H,W] fp32."""
    zero_hw = torch.zeros_like(cov[0])
    n = int(cov.shape[0])
    layer_idx = torch.arange(0, n, dtype=torch.int64, device=cov.device)
    continues = (run_start < layer_idx).to(torch.float32).view(-1, 1, 1)
    cov_prev = torch.cat([zero_hw.unsqueeze(0), cov[:-1]], dim=0) * continues
    return torch.clamp((cov - cov_prev) / torch.clamp(1.0 - cov_prev, min=1e-6), 0.0, 1.0)


@torch.jit.script
def _opacity_from_ratio(
    ratio: torch.Tensor,  # [L,H,W] fp32, from _ratio_from_coverage
    eff: torch.Tensor,  # [L,H,W] fp32 effective thickness
    h: float,
    out_dtype: torch.dtype,
) -> torch.Tensor:
    """Last part of ``_layer_opacity``: scale by the printed fraction and cast."""
    opac = ratio * (eff * (1.0 / h))
    return opac.to(out_dtype)


@torch.jit.script
def _composite_cont_mid(opac: torch.Tensor, layer_colors: torch.Tensor):
    """Flip to top->bottom order and build the shifted transmittance."""
    opac_fb = torch.flip(opac, dims=[0])  # [L,H,W]
    colors_fb = torch.flip(layer_colors, dims=[0])  # [L,3]
    trans_fb = 1.0 - opac_fb  # [L,H,W]
    trans_shift = torch.cat([torch.ones_like(trans_fb[:1]), trans_fb[:-1]], dim=0)
    return opac_fb, colors_fb, trans_shift


@torch.jit.script
def _composite_cont_post(
    remain_fb: torch.Tensor,  # [L,H,W] fp32 exclusive cumulative transmittance
    opac_fb: torch.Tensor,  # [L,H,W]
    colors_fb: torch.Tensor,  # [L,3]
    background: torch.Tensor,  # [3]
    compute_dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:
    """Top-to-bottom accumulation, given the cumulative transmittance."""
    # cumprod over up to max_layers factors accumulates rounding error each
    # step; it is accumulated in fp32 regardless of compute_dtype, then dropped
    # back down here so the (larger) downstream tensors still get the memory
    # win.
    if compute_dtype is not None:
        remain_fb = remain_fb.to(compute_dtype)

    comp_layers = (remain_fb * opac_fb).unsqueeze(-1) * colors_fb.view(
        -1, 1, 1, 3
    )  # [L,H,W,3]

    # Sum-reduce over layers in fp32 (same reasoning as cumprod above); this
    # also gives us the function's fp32 return dtype for free.
    comp = comp_layers.sum(dim=0, dtype=torch.float32)  # [H,W,3]
    del comp_layers

    # 6. background
    rem_after = remain_fb[-1] * (1.0 - opac_fb[-1])  # remaining after bottom layer
    comp = comp + rem_after.to(torch.float32).unsqueeze(-1) * background  # [H,W,3]
    return comp * 255.0


def composite_image_cont(
    pixel_height_logits: torch.Tensor,  # [H,W]
    global_logits: torch.Tensor,  # [L,M]
    tau_height: float,
    tau_global: float,
    h: float,
    max_layers: int,
    material_colors: torch.Tensor,  # [M,3]
    material_TDs: torch.Tensor,  # [M]
    background: torch.Tensor,  # [3]
    compute_dtype: Optional[torch.dtype] = None,
    gumbel_exp: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    continuous_z, p_mat, colors_f32, tds_f32 = _composite_cont_pre(
        pixel_height_logits,
        global_logits,
        tau_height,
        tau_global,
        h,
        max_layers,
        material_colors,
        material_TDs,
        background,
        compute_dtype,
        gumbel_exp,
    )
    if continuous_z.requires_grad:
        eff_thick = torch.utils.checkpoint.checkpoint(
            _print_mask_segment, continuous_z, tau_height, h, max_layers, compute_dtype,
            use_reentrant=True, preserve_rng_state=False,
        )
    else:
        eff_thick = _print_mask_segment(continuous_z, tau_height, h, max_layers, compute_dtype)
    # Issued after the big mask kernels: this chain of tiny per-layer kernels
    # launches while the GPU works on those.
    reach, slow_reach, cov_w, run_start = _cont_coverage_params(
        p_mat, colors_f32, tds_f32, background, h
    )
    layer_colors = colors_f32
    if compute_dtype is not None:
        layer_colors = layer_colors.to(compute_dtype)
    # Per-layer opacity (same ops as _layer_opacity) in two checkpointed
    # halves: their fp32 intermediates are the largest part of the autograd
    # tape, so they are recomputed in the backward pass - one half at a time -
    # instead of being kept. use_reentrant=True / preserve_rng_state=False:
    # see composite_image_cont_lowmem.
    eff = eff_thick.to(torch.float32)
    if eff.requires_grad:
        # Not checkpointed: its backward needs nothing large (cumsum saves
        # nothing, the gather only its indices), so keeping it on the plain
        # tape lets ``eff`` be released as soon as the last segment using it
        # has run its backward.
        run_thick = _run_thickness_segment(eff, run_start)
        cov = torch.utils.checkpoint.checkpoint(
            _coverage_segment, run_thick, reach, slow_reach, cov_w,
            use_reentrant=True, preserve_rng_state=False,
        )
        del run_thick
        ratio = torch.utils.checkpoint.checkpoint(
            _ratio_from_coverage, cov, run_start,
            use_reentrant=True, preserve_rng_state=False,
        )
        del cov
        opac = torch.utils.checkpoint.checkpoint(
            _opacity_from_ratio, ratio, eff, h, eff_thick.dtype,
            use_reentrant=True, preserve_rng_state=False,
        )
    else:
        cov = _coverage_segment(
            _run_thickness_segment(eff, run_start), reach, slow_reach, cov_w
        )
        ratio = _ratio_from_coverage(cov, run_start)
        del cov
        opac = _opacity_from_ratio(ratio, eff, h, eff_thick.dtype)
    del ratio, eff
    del eff_thick
    opac_fb, colors_fb, trans_shift = _composite_cont_mid(opac, layer_colors)
    del opac
    remain_fb = _CumprodDim0.apply(trans_shift)
    del trans_shift
    return _composite_cont_post(
        remain_fb, opac_fb, colors_fb, background, compute_dtype
    )


@torch.jit.script
def _composite_cont_chunk(
    continuous_z: torch.Tensor,  # [H,W] fp32, already adaptive_round-ed
    layer_ids: torch.Tensor,  # [k] fp32, ascending original layer indices lo..lo+k-1
    layer_colors: torch.Tensor,  # [k,3]
    reach: torch.Tensor,  # [k]
    slow_reach: torch.Tensor,  # [k]
    cov_w: torch.Tensor,  # [k]
    run_start: torch.Tensor,  # [k] int64
    lo: int,
    tau_height: float,
    h: float,
    carry_thick: torch.Tensor,  # [H,W]
    carry_cov: torch.Tensor,  # [H,W]
    compute_dtype: Optional[torch.dtype] = None,
):
    """One slice (layers lo..lo+k-1) of the continuous composite.

    Returns ``(contrib, tail, carry_thick, carry_cov)``: the slice's colour
    contribution assuming full incoming transmittance, the transmittance it
    passes on to what lies below it, and the run state for the slice above
    (see ``_layer_opacity``). Slices are walked bottom to top and compose as
    ``acc = contrib + tail * acc``, exactly like the unchunked layer loop.

    The print mask and bleed are per-layer independent (the bleed
    convolution is a per-layer 2D blur); only the [H,W] run state has to
    cross a slice boundary, which is what makes the layer axis chunkable.
    """
    eps: float = 1e-8
    scale = 10.0 / (tau_height + eps)
    p_print = torch.sigmoid(
        (continuous_z.unsqueeze(0) - (layer_ids.view(-1, 1, 1) + 0.5)) * scale
    )  # [k,H,W]

    if compute_dtype is not None:
        p_print = p_print.to(compute_dtype)
        layer_colors = layer_colors.to(compute_dtype)

    p_print_bleed = bleed_layer_effect(p_print, strength=0.1)
    del p_print
    eff_thick = torch.clamp(p_print_bleed, 0.0, 1.0) * h
    del p_print_bleed

    opac, carry_thick, carry_cov = _layer_opacity(
        eff_thick, reach, slow_reach, cov_w, run_start, lo, h, carry_thick, carry_cov
    )
    opac = torch.flip(opac.to(eff_thick.dtype), dims=[0])  # top to bottom
    del eff_thick
    colors_fb = torch.flip(layer_colors, dims=[0])

    trans = 1.0 - opac
    rem_local = torch.cumprod(
        torch.cat([torch.ones_like(trans[:1]), trans[:-1]], dim=0),
        dim=0,
        dtype=torch.float32,
    )
    if compute_dtype is not None:
        rem_local = rem_local.to(compute_dtype)

    comp_layers = (rem_local * opac).unsqueeze(-1) * colors_fb.view(-1, 1, 1, 3)
    contrib = comp_layers.sum(dim=0, dtype=torch.float32)  # [H,W,3]
    del comp_layers
    tail = (rem_local[-1] * trans[-1]).to(torch.float32)  # [H,W]
    return contrib, tail, carry_thick, carry_cov


def composite_image_cont_lowmem(
    pixel_height_logits: torch.Tensor,  # [H,W]
    global_logits: torch.Tensor,  # [L,M]
    tau_height: float,
    tau_global: float,
    h: float,
    max_layers: int,
    material_colors: torch.Tensor,  # [M,3]
    material_TDs: torch.Tensor,  # [M]
    background: torch.Tensor,  # [3]
    compute_dtype: Optional[torch.dtype] = None,
    gumbel_exp: Optional[torch.Tensor] = None,
    layer_chunk: int = 25,
) -> torch.Tensor:
    """Memory-lean equivalent of ``composite_image_cont``.

    ``composite_image_cont`` keeps every [L,H,W] intermediate on the autograd
    tape, and at full output resolution that sets the whole pipeline's VRAM
    high-water mark. Measured at L=75, H=W~250: 125MB of tape, 158MB forward
    peak, 204MB across forward+backward.

    Checkpointing the composite as a whole - or either of its two halves -
    does *not* help: the peak is in the backward, so the recompute simply
    rebuilds the tape it was supposed to avoid, on top of the inputs the
    checkpoint still holds (measured: peak allocated went up, not down).
    What does work is chunking the layer axis and checkpointing each chunk:
    only [H,W]-sized running state crosses a chunk boundary, so no single
    tape ever covers more than ``layer_chunk`` layers.

    Not bit-identical to the unchunked version - splitting the transmittance
    cumprod, the run-thickness cumsum and the layer sum at chunk boundaries
    regroups all three - so callers
    should be ones whose result is verified before being kept.
    """
    # 1. per-pixel continuous layer index (identical to composite_image_cont)
    pixel_height = (max_layers * h) * torch.sigmoid(pixel_height_logits)
    continuous_z = adaptive_round(pixel_height / h, tau_height, 1.0, 0.0, 0.1)

    # 2. global material weights - see composite_image_cont on gumbel_exp
    if ABLATION_NO_GUMBEL_NOISE:
        p_mat = F.softmax(global_logits / tau_global, dim=1)
    elif gumbel_exp is None:
        p_mat = F.gumbel_softmax(global_logits, tau_global, hard=False, dim=1)
    else:
        p_mat = F.softmax(
            (global_logits + (-torch.log(gumbel_exp))) / tau_global, dim=1
        )
    layer_colors = p_mat @ material_colors  # [L,3]
    layer_TDs = (p_mat @ material_TDs).clamp(1e-8, 1e8)  # [L]

    # Runs and coverage parameters, as in composite_image_cont.
    mat_idx = torch.argmax(p_mat, dim=1)
    run_start = run_starts(
        torch.cat(
            [
                torch.ones(1, dtype=torch.bool, device=p_mat.device),
                mat_idx[1:] != mat_idx[:-1],
            ],
            dim=0,
        )
    )
    reach, slow_reach, cov_w = layer_coverage_params(
        layer_colors, layer_TDs, run_start, background, h
    )
    layer_ids = torch.arange(
        max_layers, dtype=torch.float32, device=pixel_height.device
    )

    # Bottom to top, so each chunk composes over everything below it.
    comp = background.to(torch.float32).view(1, 1, 3).expand(
        pixel_height.shape[0], pixel_height.shape[1], 3
    )
    carry_thick = torch.zeros_like(pixel_height, dtype=torch.float32)
    carry_cov = torch.zeros_like(pixel_height, dtype=torch.float32)

    for start in range(0, max_layers, layer_chunk):
        stop = min(start + layer_chunk, max_layers)
        # preserve_rng_state=False: the chunk body draws no random numbers
        # (the Gumbel sample is taken above), so there is nothing to restore
        # and saving/restoring the CUDA generator per chunk per step would be
        # pure overhead. use_reentrant=True because the non-reentrant
        # implementation wraps saved tensors in hooks that the TorchScript
        # interpreter rejects on re-entry.
        contrib, tail, carry_thick, carry_cov = torch.utils.checkpoint.checkpoint(
            _composite_cont_chunk,
            continuous_z,
            layer_ids[start:stop],
            layer_colors[start:stop],
            reach[start:stop],
            slow_reach[start:stop],
            cov_w[start:stop],
            run_start[start:stop],
            start,
            tau_height,
            h,
            carry_thick,
            carry_cov,
            compute_dtype,
            use_reentrant=True,
            preserve_rng_state=False,
        )
        comp = contrib + tail.unsqueeze(-1) * comp

    return comp * 255.0


@torch.jit.script
def _runs_from_materials(mats: torch.Tensor):
    """
    Given a 1D int tensor of per-layer materials (top to bottom),
    return the start indices, end indices (exclusive) and material id
    for each run of equal values.

    Returns:
        run_starts  [R] int64
        run_ends    [R] int64
        run_mats    [R] same dtype as mats
    """
    L = int(mats.shape[0])
    if L == 0:
        empty_i = torch.empty(0, dtype=torch.int64, device=mats.device)
        return empty_i, empty_i, torch.empty(0, dtype=mats.dtype, device=mats.device)

    change = torch.ones(L, dtype=torch.bool, device=mats.device)
    change[1:] = mats[1:] != mats[:-1]

    # TorchScript friendly: no keyword args
    run_starts = torch.nonzero(change).squeeze(1).to(torch.int64)  # [R]
    run_ends = torch.cat([run_starts[1:], torch.tensor([L], device=mats.device)])
    run_mats = mats[run_starts]  # [R]

    return run_starts, run_ends, run_mats


# --------------------------------------------------------------------------
# CUDA graph replay for repeated no-grad composites
#
# Pruning and seed search evaluate the same discrete composite thousands of
# times with inputs of identical shape. Each call is a few hundred small
# kernels, so it is bound by launch overhead rather than GPU work. Inside a
# ``composite_graph_scope``, after a few eager calls of the same shape the
# call is captured into a CUDA graph and replayed from then on: the replay
# runs exactly the kernels of an eager call, so the results are identical.
# A captured graph keeps its working memory, so graphs only live for the
# duration of the scope (a pruning phase), never alongside training.
# --------------------------------------------------------------------------
_GRAPH_WARM_CALLS: Final[int] = 3
_GRAPH_CACHE_SIZE: Final[int] = 4
# Scope depth, cache and memory pool are per thread. They used to be
# process-wide: while pruning held a scope in its thread, every other thread
# that composited (the webui's preview callback, slider re-renders from the
# page) went through the same unlocked cache and shared pool, and the two
# threads' captures and evictions corrupted each other - "it->second->
# use_count > 0 INTERNAL ASSERT FAILED" in capture_begin, after which the
# process's CUDA RNG was unusable for every later run ("Offset increment
# outside graph capture encountered unexpectedly"). A scope now only ever
# affects the thread that opened it; any other thread composites eagerly.
_graph_state = threading.local()
# Captures are serialised process-wide, and so are composites from threads
# without a scope of their own: CUDA does not let another thread capture, or
# launch the allocations/kernels of a composite, in the middle of a capture
# (cudaErrorIllegalState). A capture takes milliseconds, so a preview render
# waiting for one is not noticeable; replays never take the lock.
_capture_lock = threading.RLock()


def _graph_thread_state():
    st = _graph_state
    if not hasattr(st, "depth"):
        st.depth = 0
        st.cache = OrderedDict()
        st.pool = None
    return st


@contextmanager
def composite_graph_scope():
    """Enable CUDA graph replay of repeated composites for the enclosed block
    in the current thread (also usable as a decorator). Captured graphs and
    their memory are released when the thread's outermost scope exits."""
    st = _graph_thread_state()
    st.depth += 1
    try:
        yield
    finally:
        st.depth -= 1
        if st.depth == 0:
            st.cache.clear()
            st.pool = None


def _replay_captured(fn, key: str, tensors: list, *rest):
    """``fn(*tensors, *rest)``, replayed from a captured CUDA graph once the
    same call (shapes, dtypes and non-tensor arguments) has been seen
    ``_GRAPH_WARM_CALLS`` times inside a ``composite_graph_scope`` of this
    thread. Falls back to a plain call outside a scope, off CUDA, while a
    graph is already being captured, or when gradients are needed."""
    st = _graph_thread_state()
    if not tensors[0].is_cuda or torch.cuda.is_current_stream_capturing():
        return fn(*tensors, *rest)
    if st.depth == 0 or (torch.is_grad_enabled() and any(t.requires_grad for t in tensors)):
        with _capture_lock:
            return fn(*tensors, *rest)
    full_key = (
        key,
        tuple((tuple(t.shape), t.dtype, t.device, t.stride()) for t in tensors),
        rest,
    )
    cache = st.cache
    entry = cache.get(full_key)
    if entry is None:
        entry = {"calls": 0, "graph": None}
        cache[full_key] = entry
        while len(cache) > _GRAPH_CACHE_SIZE:
            cache.popitem(last=False)
    else:
        cache.move_to_end(full_key)

    if entry["graph"] is None:
        entry["calls"] += 1
        # Capture only after the call has run eagerly in this thread: cuBLAS
        # (and other libraries) create their per-thread handles lazily on
        # first use, and creating one during capture fails
        # (CUBLAS_STATUS_NOT_INITIALIZED, which also invalidates the capture).
        if entry["calls"] <= _GRAPH_WARM_CALLS:
            return fn(*tensors, *rest)
        with _capture_lock:
            _capture(fn, entry, st, tensors, rest)

    for dst, src in zip(entry["inputs"], tensors):
        dst.copy_(src)
    entry["graph"].replay()
    return entry["output"].clone()


def _capture(fn, entry: dict, st, tensors: list, rest: tuple) -> None:
    """Capture ``fn(*tensors, *rest)`` into ``entry`` (caller holds
    ``_capture_lock``)."""
    # A pool lives only as long as a graph captured into it: once the cache
    # has evicted the last of them (layer pruning changes the shape after
    # every removal, so shapes that never get captured push the only
    # captured one out) the pool's use count is zero and capturing into it
    # again trips "it->second->use_count > 0 INTERNAL ASSERT FAILED". Start
    # a fresh pool then.
    if st.pool is None or not any(e["graph"] is not None for e in st.cache.values()):
        st.pool = torch.cuda.graph_pool_handle()
    static_in = [t.clone() for t in tensors]
    graph = torch.cuda.CUDAGraph()
    with torch.no_grad():
        # The call has already run eagerly _GRAPH_WARM_CALLS times, so all
        # lazy initialisation is done; no separate warm-up run is needed.
        # All of a thread's captured composites share one memory pool: they
        # are only ever replayed one at a time and their outputs are copied
        # out.
        with torch.cuda.graph(
            graph, pool=st.pool, capture_error_mode="thread_local"
        ):
            static_out = fn(*static_in, *rest)
    entry.update(graph=graph, inputs=static_in, output=static_out)


def composite_image_disc(
    pixel_height_logits: torch.Tensor,  # [H,W]
    global_logits: torch.Tensor,  # [max_layers, n_materials]
    tau_height: float,
    tau_global: float,
    h: float,
    max_layers: int,
    material_colors: torch.Tensor,  # [n_materials, 3]
    material_TDs: torch.Tensor,  # [n_materials]
    background: torch.Tensor,  # [3]
    rng_seed: int = -1,
    compute_dtype: Optional[torch.dtype] = None,
    layer_chunk: int = 25,
) -> torch.Tensor:
    """Discrete counterpart of `composite_image_cont` (see
    ``_composite_image_disc``); repeated calls replay a CUDA graph."""

    def run(pl, gl, mc, mt, bg, tau_h, tau_g, h_, n_layers, seed, dtype, chunk):
        return _composite_image_disc(
            pl, gl, tau_h, tau_g, h_, n_layers, mc, mt, bg, seed, dtype, chunk
        )

    return _replay_captured(
        run,
        "composite_image_disc",
        [pixel_height_logits, global_logits, material_colors, material_TDs, background],
        float(tau_height),
        float(tau_global),
        float(h),
        int(max_layers),
        int(rng_seed),
        compute_dtype,
        int(layer_chunk),
    )


@torch.jit.script
def _composite_image_disc(
    pixel_height_logits: torch.Tensor,  # [H,W]
    global_logits: torch.Tensor,  # [max_layers, n_materials]
    tau_height: float,
    tau_global: float,
    h: float,
    max_layers: int,
    material_colors: torch.Tensor,  # [n_materials, 3]
    material_TDs: torch.Tensor,  # [n_materials]
    background: torch.Tensor,  # [3]
    rng_seed: int = -1,
    compute_dtype: Optional[torch.dtype] = None,
    layer_chunk: int = 25,
) -> torch.Tensor:
    """
    Discrete counterpart of `composite_image_cont`.

    * Heights are snapped to whole layers with `adaptive_round`.
    * Each layer gets exactly one material chosen with
      `deterministic_gumbel_softmax`, making the result pixel-wise
      discrete in both height and color while gradients still flow
      through the soft procedures when temperatures are >0.
    """
    eps: float = 1e-8

    # 1. Discretise per-pixel heights (top of printed stack in units of h).
    pixel_height: torch.Tensor = (float(max_layers) * h) * torch.sigmoid(
        pixel_height_logits
    )
    z_cont: torch.Tensor = pixel_height / h
    #   Adaptive rounding: low_tau=0 means perfectly hard when tau_height→0,
    #   high_tau=1 gives fully soft when tau_height→1.  temp=0.1 sets the
    #   sharpness of the sigmoid used inside adaptive_round.
    z_disc: torch.Tensor = adaptive_round(z_cont, tau_height, 1.0, 0.0, 0.1)
    z_disc = torch.clamp(z_disc, 0.0, float(max_layers))
    z_int: torch.Tensor = torch.round(z_disc).to(torch.int64)  # [H, W]

    # 2. Pick one material for every layer with a deterministic Gumbel-Softmax.
    L: int = int(global_logits.shape[0])
    n_mat: int = int(global_logits.shape[1])

    seed_base: int = rng_seed if rng_seed >= 0 else 0
    sel = batched_layer_material_indices(global_logits, tau_global, seed_base)  # [L]
    layer_colors = material_colors.index_select(0, sel)  # [L,3]
    layer_TDs = material_TDs.index_select(0, sel).clamp(1e-8, 1e8)  # [L]

    # Runs of the same material and their coverage parameters (per layer).
    run_start = material_run_starts(layer_colors, layer_TDs)
    reach, slow_reach, cov_w = layer_coverage_params(
        layer_colors, layer_TDs, run_start, background, h
    )

    # 3-6. Walk the stack bottom-to-top in chunks of `layer_chunk` layers.
    #
    # Every step from the binary print mask through the bleed is per-layer
    # independent (the bleed is a per-layer 2D blur), so the layer axis
    # chunks cleanly: only the [H,W,3] colour of everything below, and the
    # [H,W] run thickness/coverage of the layer below, cross a chunk
    # boundary. That bounds the working set by the chunk instead of by
    # max_layers, which matters because this function runs at full *output*
    # resolution in every pruning phase - a single [L,H,W] tensor is 18MB at
    # stl_output_size=50 but 169MB at the tool's default 150.
    H_out: int = int(z_int.shape[0])
    W_out: int = int(z_int.shape[1])
    comp = background.to(torch.float32).view(1, 1, 3).expand(H_out, W_out, 3)
    carry_thick = torch.zeros(
        (H_out, W_out), dtype=torch.float32, device=pixel_height.device
    )
    carry_cov = torch.zeros_like(carry_thick)

    lo: int = 0
    while lo < max_layers:
        hi: int = lo + layer_chunk
        if hi > max_layers:
            hi = max_layers
        idx = torch.arange(lo, hi, dtype=torch.int64, device=pixel_height.device)
        p_print = (idx.view(-1, 1, 1) < z_int.unsqueeze(0)).to(
            pixel_height.dtype
        )  # [k,H,W] bottom to top

        # See composite_image_cont: @torch.jit.script does not observe
        # ambient torch.autocast, so cast explicitly here.
        if compute_dtype is not None:
            p_print = p_print.to(compute_dtype)

        p_print_bleed = bleed_layer_effect(p_print, strength=0.1)
        eff_thick = torch.clamp(p_print_bleed, 0.0, 1.0) * h
        opac, carry_thick, carry_cov = _layer_opacity(
            eff_thick,
            reach[lo:hi],
            slow_reach[lo:hi],
            cov_w[lo:hi],
            run_start[lo:hi],
            lo,
            h,
            carry_thick,
            carry_cov,
        )
        # Top-to-bottom within the chunk, the direction the transmittance
        # accumulates in.
        opac = opac.flip([0]).to(eff_thick.dtype)
        cols_c = layer_colors[lo:hi].flip([0]).to(eff_thick.dtype)
        trans = 1.0 - opac
        # Accumulate cumprod/sum in fp32 regardless of compute_dtype
        # (rounding error compounds across layers), then drop back down so
        # the larger downstream tensor still gets the memory win.
        rem_local = torch.cumprod(
            torch.cat([torch.ones_like(trans[:1]), trans[:-1]], dim=0),
            dim=0,
            dtype=torch.float32,
        )
        if compute_dtype is not None:
            rem_local = rem_local.to(compute_dtype)

        contrib = ((rem_local * opac).unsqueeze(-1) * cols_c.view(-1, 1, 1, 3)).sum(
            dim=0, dtype=torch.float32
        )  # [H,W,3]
        tail = (rem_local[-1] * trans[-1]).to(torch.float32)
        comp = contrib + tail.unsqueeze(-1) * comp
        lo = hi

    return comp * 255.0


def _gpu_capability(device):
    """CUDA compute capability as an int (80, 61, …); 0 on non-CUDA backends."""
    if getattr(device, "type", None) != "cuda":
        return 0
    try:
        major, minor = torch.cuda.get_device_capability(device)
    except Exception:
        return 0
    return major * 10 + minor  # 80, 61, …


def _has_fp16(device):
    return _gpu_capability(device) >= 53  # CC 5.3 or newer


class PrecisionManager:
    """
    Usage
    -----
    prec = PrecisionManager(device)
    with prec.autocast():
        loss = model(...)
    prec.backward_and_step(loss, optimizer)
    """

    def __init__(self, device):
        self.device = device
        self.scaler = None
        self.autocast_dtype = None
        self.enabled = False

        # Decide dtype once using the shared runtime probe
        dtype, _reason = get_selected_autocast(device)
        self.autocast_dtype = dtype
        # GPU backends only. CPU bf16 is deliberately excluded: the probe may
        # accept it, but enabling it here would also switch
        # `composite_compute_dtype` to bf16 and change CPU-run results, which
        # is not what a CPU fallback run should silently do.
        # `device.type == "cuda"` covers ROCm too - PyTorch reports AMD GPUs
        # as CUDA devices - and MPS only reaches here via an explicit
        # AUTOFORGE_AMP override that already passed the runtime probe.
        self.enabled = dtype is not None and device.type in ("cuda", "mps")

        # Use GradScaler only for float16; bf16 does not need scaling.
        # torch.amp.GradScaler takes the device type, so the same call works
        # for CUDA, ROCm and MPS (the old torch.cuda.amp.GradScaler was both
        # deprecated and hard-wired to CUDA).
        if self.enabled and dtype == torch.float16:
            self.scaler = torch.amp.GradScaler(device.type)
        else:
            self.scaler = None

        # Optional: If a CUDA-family GPU but no AMP selected, allow TF32 for
        # speed on Ampere+ (silently ignored on ROCm).
        if device.type == "cuda" and dtype is None:
            try:
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True
            except Exception:
                pass

    @contextmanager
    def autocast(self):
        if self.enabled:
            with torch.amp.autocast(self.device.type, dtype=self.autocast_dtype):
                yield
        else:
            yield  # FP32 path

    def backward_and_step(self, loss, optimizer):
        if self.scaler is not None:  # FP16 path
            self.scaler.scale(loss).backward()
            self.scaler.step(optimizer)
            self.scaler.update()
        else:
            loss.backward()
            optimizer.step()
