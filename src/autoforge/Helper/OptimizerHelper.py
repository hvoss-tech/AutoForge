from contextlib import contextmanager

import torch
import torch.nn.functional as F
from autoforge.Helper.AmpUtils import get_selected_autocast


@torch.jit.script
def adaptive_round(
    x: torch.Tensor, tau: float, high_tau: float, low_tau: float, temp: float
) -> torch.Tensor:
    """
    Smooth rounding based on temperature 'tau'.

    Args:
        x (torch.Tensor): The input tensor to be rounded.
        tau (float): The current temperature parameter.
        high_tau (float): The high threshold for the temperature.
        low_tau (float): The low threshold for the temperature.
        temp (float): The temperature parameter for the sigmoid function.

    Returns:
        torch.Tensor: The rounded tensor.
    """
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

    # 3x3 average kernel
    kernel = (
        torch.tensor(
            [[1, 1, 1], [1, 0, 1], [1, 1, 1]], dtype=mask.dtype, device=mask.device
        )
        / 8.0
    )  # 8 neighbors

    kernel = kernel.view(1, 1, 3, 3)

    # Apply conv2d to each layer independently
    blurred = F.conv2d(mask.unsqueeze(1), kernel, padding=1, groups=1).squeeze(
        1
    )  # [L,H,W]

    # Combine original mask with bleed from neighbors
    return mask + strength * blurred


@torch.jit.script
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
) -> torch.Tensor:
    # 1. per-pixel continuous layer index
    pixel_height = (max_layers * h) * torch.sigmoid(pixel_height_logits)  # [H,W]
    continuous_z = pixel_height / h  # [H,W]
    continuous_z = adaptive_round(continuous_z, tau_height, 1.0, 0.0, 0.1)

    # 2. global material weights with Gumbel-Softmax
    p_mat = F.gumbel_softmax(global_logits, tau_global, hard=False, dim=1)  # [L,M]

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

    # 4. thickness and opacity
    p_print_bleed = bleed_layer_effect(p_print, strength=0.1)  # [L,H,W]
    del p_print
    eff_thick = torch.clamp(p_print_bleed, 0.0, 1.0) * h
    del p_print_bleed
    thick_ratio = eff_thick / layer_TDs.view(-1, 1, 1)  # [L,H,W]
    del eff_thick

    o, A, k, b = -2.9864511e-02, 4.0532556e-01, 8.2597107e+01, 1.2547257e+00
    opac = o + (A * torch.log1p(k * thick_ratio) + b * thick_ratio)
    del thick_ratio
    opac = torch.clamp(opac, 0.0, 1.0)  # [L,H,W]

    # 5. flip to top->bottom order before compositing
    opac_fb = torch.flip(opac, dims=[0])  # [L,H,W]
    del opac
    colors_fb = torch.flip(layer_colors, dims=[0])  # [L,3]
    del layer_colors

    trans_fb = 1.0 - opac_fb  # [L,H,W]
    trans_shift = torch.cat([torch.ones_like(trans_fb[:1]), trans_fb[:-1]], dim=0)
    remain_fb = torch.cumprod(trans_shift, dim=0)  # remaining before each layer [L,H,W]
    del trans_shift

    comp_layers = (remain_fb * opac_fb).unsqueeze(-1) * colors_fb.view(
        -1, 1, 1, 3
    )  # [L,H,W,3]
    del opac_fb, colors_fb

    comp = comp_layers.sum(dim=0)  # [H,W,3]
    del comp_layers

    # 6. background
    rem_after = remain_fb[-1] * trans_fb[-1]  # remaining after bottom layer
    del remain_fb, trans_fb
    comp = comp + rem_after.unsqueeze(-1) * background  # [H,W,3]
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

    layer_colors: torch.Tensor = torch.empty(
        (L, 3), dtype=material_colors.dtype, device=material_colors.device
    )
    layer_TDs: torch.Tensor = torch.empty(
        (L,), dtype=material_TDs.dtype, device=material_TDs.device
    )

    seed_base: int = rng_seed if rng_seed >= 0 else 0
    hard_flag: bool = True  # always one-hot
    for j in range(L):
        seed_j: int = seed_base + j
        one_hot: torch.Tensor = deterministic_gumbel_softmax(
            global_logits[j], tau_global, hard_flag, seed_j
        )  # [n_materials]
        idx: int = int(torch.argmax(one_hot).item())
        layer_colors[j] = material_colors[idx]
        layer_TDs[j] = material_TDs[idx].clamp(1e-8, 1e8)

    # 3. Binary print mask: a layer is present iff its index < z_int.
    layer_idx: torch.Tensor = torch.arange(
        max_layers, dtype=torch.int64, device=pixel_height.device
    ).view(-1, 1, 1)  # [L,1,1]
    p_print: torch.Tensor = (layer_idx < z_int.unsqueeze(0)).to(
        pixel_height.dtype
    )  # [L,H,W]

    # 4. Thickness, opacity and the rest exactly as in the continuous version.
    p_print_bleed = bleed_layer_effect(p_print, strength=0.1)  # [L,H,W]
    del p_print
    eff_thick = torch.clamp(p_print_bleed, 0.0, 1.0) * h
    del p_print_bleed
    thick_ratio: torch.Tensor = eff_thick / layer_TDs.view(-1, 1, 1)  # [L,H,W]
    del eff_thick

    o, A, k, b = -2.9864511e-02, 4.0532556e-01, 8.2597107e+01, 1.2547257e+00
    opac: torch.Tensor = o + (A * torch.log1p(k * thick_ratio) + b * thick_ratio)
    del thick_ratio
    opac = torch.clamp(opac, 0.0, 1.0)  # [L,H,W]

    # 5. Top-to-bottom compositing (same flipping trick as before).
    opac_fb = torch.flip(opac, dims=[0])  # [L,H,W]
    del opac
    colors_fb = torch.flip(layer_colors, dims=[0])  # [L,3]
    del layer_colors

    trans_fb = 1.0 - opac_fb  # [L,H,W]
    trans_prev = torch.cat([torch.ones_like(trans_fb[:1]), trans_fb[:-1]], dim=0)
    remain_fb = torch.cumprod(trans_prev, dim=0)  # [L,H,W]
    del trans_prev

    comp_layers = (remain_fb * opac_fb).unsqueeze(-1) * colors_fb.view(
        -1, 1, 1, 3
    )  # [L,H,W,3]
    del opac_fb, colors_fb

    comp = comp_layers.sum(dim=0)  # [H,W,3]
    del comp_layers

    # 6. Background
    rem_after = remain_fb[-1] * trans_fb[-1]
    del remain_fb, trans_fb
    comp = comp + rem_after.unsqueeze(-1) * background  # [H,W,3]

    return comp * 255.0


def _gpu_capability(device):
    major, minor = torch.cuda.get_device_capability(device)
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
        # Enable only when a non-None dtype is selected and device supports native CUDA autocast
        self.enabled = dtype is not None and device.type == "cuda"

        # Use GradScaler only for CUDA float16; bf16 does not need scaling
        if self.enabled and dtype == torch.float16:
            self.scaler = torch.cuda.amp.GradScaler()
        else:
            self.scaler = None

        # Optional: If CUDA but no AMP selected, allow TF32 for speed on Ampere+
        if device.type == "cuda" and dtype is None:
            try:
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True
            except Exception:
                pass

    @contextmanager
    def autocast(self):
        if self.enabled:
            with torch.cuda.amp.autocast(dtype=self.autocast_dtype):
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
