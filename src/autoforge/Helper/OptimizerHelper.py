import os
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


@torch.jit.script
def _compute_opacity(
    thick_ratio: torch.Tensor, beer_lambert: bool = ABLATION_BEER_LAMBERT_OPACITY
) -> torch.Tensor:
    """Opacity as a function of thickness/TD.

    Default: the empirically-fitted 4-parameter model (see Appendix A of the
    design doc). Under ``AUTOFORGE_ABLATION_OPACITY=beer_lambert`` this is
    replaced with the textbook single-scattering Beer-Lambert law
    ``1 - exp(-thick_ratio)`` instead - the "Without Opacity Calibration"
    ablation.
    """
    if beer_lambert:
        return torch.clamp(1.0 - torch.exp(-thick_ratio), 0.0, 1.0)
    o, A, k, b = -2.9864511e-02, 4.0532556e-01, 8.2597107e+01, 1.2547257e+00
    return torch.clamp(o + (A * torch.log1p(k * thick_ratio) + b * thick_ratio), 0.0, 1.0)


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
    compute_dtype: Optional[torch.dtype] = None,
    gumbel_exp: Optional[torch.Tensor] = None,  # [L,M] Exponential(1) samples
    no_gumbel: bool = ABLATION_NO_GUMBEL_NOISE,
):
    """Everything in the continuous composite up to (and including) the
    shifted top-to-bottom transmittance stack.

    Split out of ``composite_image_cont`` so the ``cumprod`` in between can go
    through ``_CumprodDim0`` - a ``torch.autograd.Function`` can't be called
    from TorchScript, and dropping ``@torch.jit.script`` from the whole
    composite costs ~40% (2.85ms -> 3.98ms fwd+bwd at L=75, H=W=250), so the
    scripted region is kept and merely cut in two around it.

    Returns ``(opac_fb, colors_fb, trans_shift)``.
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

    # 3. soft print mask for all layers (layer 0 = bottom, layer L-1 = top)
    #    Small τ  -> large scale (steep transition)
    #    Large τ  -> small scale (smooth transition)
    eps = 1e-8
    scale = 10.0 / (tau_height + eps)
    layer_idx = torch.arange(
        max_layers, dtype=torch.float32, device=pixel_height.device
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
        layer_colors = layer_colors.to(compute_dtype)
        layer_TDs = layer_TDs.to(compute_dtype)

    # 4. thickness and opacity
    p_print_bleed = bleed_layer_effect(p_print, strength=0.1)  # [L,H,W]
    del p_print
    eff_thick = torch.clamp(p_print_bleed, 0.0, 1.0) * h
    del p_print_bleed
    thick_ratio = eff_thick / layer_TDs.view(-1, 1, 1)  # [L,H,W]
    del eff_thick

    opac = _compute_opacity(thick_ratio)  # [L,H,W]
    del thick_ratio

    # 5. flip to top->bottom order before compositing
    opac_fb = torch.flip(opac, dims=[0])  # [L,H,W]
    del opac
    colors_fb = torch.flip(layer_colors, dims=[0])  # [L,3]
    del layer_colors

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
    opac_fb, colors_fb, trans_shift = _composite_cont_pre(
        pixel_height_logits,
        global_logits,
        tau_height,
        tau_global,
        h,
        max_layers,
        material_colors,
        material_TDs,
        compute_dtype,
        gumbel_exp,
    )
    remain_fb = _CumprodDim0.apply(trans_shift)
    del trans_shift
    return _composite_cont_post(
        remain_fb, opac_fb, colors_fb, background, compute_dtype
    )


@torch.jit.script
def _composite_cont_chunk(
    continuous_z: torch.Tensor,  # [H,W] fp32, already adaptive_round-ed
    layer_ids_fb: torch.Tensor,  # [k] fp32, descending original layer indices
    layer_colors_fb: torch.Tensor,  # [k,3]
    layer_TDs_fb: torch.Tensor,  # [k]
    tau_height: float,
    h: float,
    compute_dtype: Optional[torch.dtype] = None,
):
    """One top-to-bottom slice of the continuous composite.

    Returns ``(contrib, tail)``: the slice's colour contribution assuming full
    incoming transmittance, and the transmittance it passes on to the slice
    below. The caller scales ``contrib`` by the running transmittance and
    multiplies ``tail`` into it, so slices compose exactly the way the
    unchunked layer loop does.

    Every step here is per-layer independent (the bleed convolution is a
    per-layer 2D blur), which is what makes the layer axis chunkable at all.
    """
    eps: float = 1e-8
    scale = 10.0 / (tau_height + eps)
    p_print = torch.sigmoid(
        (continuous_z.unsqueeze(0) - (layer_ids_fb.view(-1, 1, 1) + 0.5)) * scale
    )  # [k,H,W]

    if compute_dtype is not None:
        p_print = p_print.to(compute_dtype)
        layer_colors_fb = layer_colors_fb.to(compute_dtype)
        layer_TDs_fb = layer_TDs_fb.to(compute_dtype)

    p_print_bleed = bleed_layer_effect(p_print, strength=0.1)
    del p_print
    eff_thick = torch.clamp(p_print_bleed, 0.0, 1.0) * h
    del p_print_bleed
    thick_ratio = eff_thick / layer_TDs_fb.view(-1, 1, 1)
    del eff_thick

    opac = _compute_opacity(thick_ratio)  # [k,H,W]
    del thick_ratio

    trans = 1.0 - opac
    rem_local = torch.cumprod(
        torch.cat([torch.ones_like(trans[:1]), trans[:-1]], dim=0),
        dim=0,
        dtype=torch.float32,
    )
    if compute_dtype is not None:
        rem_local = rem_local.to(compute_dtype)

    comp_layers = (rem_local * opac).unsqueeze(-1) * layer_colors_fb.view(-1, 1, 1, 3)
    contrib = comp_layers.sum(dim=0, dtype=torch.float32)  # [H,W,3]
    del comp_layers
    tail = (rem_local[-1] * trans[-1]).to(torch.float32)  # [H,W]
    return contrib, tail


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
    cumprod and the layer sum at chunk boundaries regroups both - so callers
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

    # Top-to-bottom order, so chunk i sits above chunk i+1.
    colors_fb = torch.flip(layer_colors, dims=[0])
    TDs_fb = torch.flip(layer_TDs, dims=[0])
    ids_fb = torch.arange(
        max_layers - 1, -1, -1, dtype=torch.float32, device=pixel_height.device
    )

    comp = torch.zeros(
        pixel_height.shape[0],
        pixel_height.shape[1],
        3,
        dtype=torch.float32,
        device=pixel_height.device,
    )
    remain = torch.ones_like(pixel_height, dtype=torch.float32)

    for start in range(0, max_layers, layer_chunk):
        stop = min(start + layer_chunk, max_layers)
        # preserve_rng_state=False: the chunk body draws no random numbers
        # (the Gumbel sample is taken above), so there is nothing to restore
        # and saving/restoring the CUDA generator per chunk per step would be
        # pure overhead. use_reentrant=True because the non-reentrant
        # implementation wraps saved tensors in hooks that the TorchScript
        # interpreter rejects on re-entry.
        contrib, tail = torch.utils.checkpoint.checkpoint(
            _composite_cont_chunk,
            continuous_z,
            ids_fb[start:stop],
            colors_fb[start:stop],
            TDs_fb[start:stop],
            tau_height,
            h,
            compute_dtype,
            use_reentrant=True,
            preserve_rng_state=False,
        )
        comp = comp + remain.unsqueeze(-1) * contrib
        remain = remain * tail

    comp = comp + remain.unsqueeze(-1) * background
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


@torch.jit.script
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

    # 3-6. Walk the stack top-to-bottom in chunks of `layer_chunk` layers.
    #
    # Every step from the binary print mask through the opacity curve is
    # per-layer independent (the bleed is a per-layer 2D blur), so the layer
    # axis chunks cleanly: only the [H,W] running transmittance and the
    # [H,W,3] accumulated colour cross a chunk boundary. That bounds the
    # working set by the chunk instead of by max_layers, which matters
    # because this function runs at full *output* resolution in every pruning
    # phase - a single [L,H,W] tensor is 18MB at stl_output_size=50 but
    # 169MB at the tool's default 150.
    H_out: int = int(z_int.shape[0])
    W_out: int = int(z_int.shape[1])
    comp = torch.zeros(
        (H_out, W_out, 3), dtype=torch.float32, device=pixel_height.device
    )
    remain = torch.ones(
        (H_out, W_out), dtype=torch.float32, device=pixel_height.device
    )

    hi: int = max_layers
    while hi > 0:
        lo: int = hi - layer_chunk
        if lo < 0:
            lo = 0
        # Descending layer indices so each chunk is ordered top-to-bottom,
        # matching the direction the transmittance accumulates in.
        idx = torch.arange(
            lo, hi, dtype=torch.int64, device=pixel_height.device
        ).flip([0])
        p_print = (idx.view(-1, 1, 1) < z_int.unsqueeze(0)).to(
            pixel_height.dtype
        )  # [k,H,W]
        cols_c = layer_colors[lo:hi].flip([0])
        tds_c = layer_TDs[lo:hi].flip([0])

        # See composite_image_cont: @torch.jit.script does not observe
        # ambient torch.autocast, so cast explicitly here.
        if compute_dtype is not None:
            p_print = p_print.to(compute_dtype)
            cols_c = cols_c.to(compute_dtype)
            tds_c = tds_c.to(compute_dtype)

        p_print_bleed = bleed_layer_effect(p_print, strength=0.1)
        eff_thick = torch.clamp(p_print_bleed, 0.0, 1.0) * h
        thick_ratio = eff_thick / tds_c.view(-1, 1, 1)
        opac = _compute_opacity(thick_ratio)
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
        comp = comp + remain.unsqueeze(-1) * contrib
        remain = remain * (rem_local[-1] * trans[-1]).to(torch.float32)
        hi = lo

    comp = comp + remain.unsqueeze(-1) * background  # [H,W,3]
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
