from typing import Optional

import torch
import torch.nn.functional as F

from autoforge.Helper.ImageHelper import srgb_to_lab
from autoforge.Helper.OptimizerHelper import (
    composite_image_cont,
    composite_image_cont_lowmem,
)


def loss_fn(
    params: dict,
    target: torch.Tensor,
    tau_height: float,
    tau_global: float,
    h: float,
    max_layers: int,
    material_colors: torch.Tensor,
    material_TDs: torch.Tensor,
    background: torch.Tensor,
    add_penalty_loss: float = 0.0,
    focus_map: torch.Tensor = None,
    alpha: torch.Tensor = None,
    compute_dtype: Optional[torch.dtype] = None,
    gumbel_exp: Optional[torch.Tensor] = None,
    low_memory: bool = False,
) -> torch.Tensor:
    """
    Full forward pass for continuous assignment:
    composite, then compute unified loss.
    focus_map acts as a priority mask (values in [0,1]) where 1.0 means full weight and 0 means low weight.
    alpha (optional) is a per-pixel alpha mask [H,W] or [H,W,1] in 0-255 range;
        pixels with alpha < 128 are masked out of the loss.
    compute_dtype (optional): if set (e.g. torch.bfloat16), the memory-heavy
        per-layer compositing math runs in this precision instead of fp32 -
        see composite_image_cont for why this has to be threaded through
        explicitly rather than relying on ambient torch.autocast.
    gumbel_exp (optional): pre-drawn Exponential(1) noise for the material
        Gumbel-Softmax, so the RNG draw can be kept outside a CUDA-graph
        capture region - see composite_image_cont.
    """
    composite = composite_image_cont_lowmem if low_memory else composite_image_cont
    comp = composite(
        params["pixel_height_logits"],
        params["global_logits"],
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
    return compute_loss(
        comp=comp,
        target=target,
        pixel_height_logits=params.get("pixel_height_logits", None),
        tau_height=tau_height,
        add_penalty_loss=add_penalty_loss,
        focus_map=focus_map,
        alpha=alpha,
    )


def compute_loss(
    comp: torch.Tensor,
    target: torch.Tensor,
    pixel_height_logits: torch.Tensor = None,
    tau_height: float = 1.0,
    add_penalty_loss: float = 0.0,
    focus_map: torch.Tensor = None,
    alpha: torch.Tensor = None,
) -> torch.Tensor:
    """
    Compute loss between composite and target.

    If focus_map (priority mask) is provided (shape [H,W], >= 0), we apply per-pixel weights:
        weight = 0.1 + 0.9 * focus_map
    (So outside mask -> 0.1, a plain 0..1 mask at full -> 1.0, gradients respected.)
    Values above 1 weight a pixel more than 10x the unmasked ones: that is how
    --priority_mask_strength is applied (see auto_forge._load_priority_mask).

    If alpha is provided (shape [H,W] or [H,W,1], values 0-255), transparent pixels
    (alpha < 128) are masked out entirely (weight = 0). When both focus_map and alpha
    are provided, the masks are combined multiplicatively.

    The final loss is the weighted mean of per-pixel Lab-space MSE.
    We normalize by the mean weight to keep the magnitude comparable with the unweighted loss.
    """
    comp_lab = srgb_to_lab(comp)

    # `target` is a fixed image for many consecutive calls (every training
    # step, all pruning phases operating at one resolution). Cache its Lab
    # conversion on the tensor object itself so we don't redo the (fairly
    # expensive, pow/log-heavy) color conversion every single call - it only
    # needs recomputing when the target tensor is swapped out (e.g. going
    # from solver resolution to full output resolution during post-processing).
    target_lab = getattr(target, "_af_lab_cache", None)
    if (
        target_lab is None
        or target_lab.dtype != comp_lab.dtype
        or target_lab.device != comp_lab.device
    ):
        target_lab = srgb_to_lab(target)
        try:
            target._af_lab_cache = target_lab
        except Exception:
            pass

    if focus_map is None and alpha is None:
        mse_loss = F.mse_loss(comp_lab, target_lab)
        total_loss = mse_loss
    else:
        per_pixel_mse = (comp_lab - target_lab).pow(2).mean(dim=2)  # [H,W]
        weights = torch.ones_like(per_pixel_mse)

        if focus_map is not None:
            if focus_map.dim() == 3 and focus_map.shape[-1] == 1:
                focus_map_proc = focus_map.squeeze(-1)
            else:
                focus_map_proc = focus_map
            focus_map_proc = torch.clamp(focus_map_proc, min=0.0)
            weights = weights * (0.1 + 0.9 * focus_map_proc)

        if alpha is not None:
            if alpha.dim() == 3 and alpha.shape[-1] == 1:
                alpha_proc = alpha.squeeze(-1)
            else:
                alpha_proc = alpha
            # Resize alpha to match per_pixel_mse spatial dims (safety net
            # for any resolution mismatch in the pipeline)
            if alpha_proc.shape != per_pixel_mse.shape:
                alpha_proc = F.interpolate(
                    alpha_proc.unsqueeze(0).unsqueeze(0),
                    size=per_pixel_mse.shape[-2:],
                    mode="nearest",
                ).squeeze(0).squeeze(0)
            alpha_mask = (alpha_proc >= 128).float()
            weights = weights * alpha_mask

        weighted_loss = per_pixel_mse * weights
        total_loss = weighted_loss.mean() / weights.mean().clamp(min=1e-8)

    # Height-map smoothness penalty (Laplacian / total variation)
    if add_penalty_loss > 0 and pixel_height_logits is not None and pixel_height_logits.dim() == 2:
        dy = (pixel_height_logits[:, 1:] - pixel_height_logits[:, :-1]).pow(2).mean()
        dx = (pixel_height_logits[1:, :] - pixel_height_logits[:-1, :]).pow(2).mean()
        total_loss = total_loss + (dx + dy) * add_penalty_loss

    return total_loss
