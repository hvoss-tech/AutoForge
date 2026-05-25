import torch
import torch.nn.functional as F

from autoforge.Helper.ImageHelper import srgb_to_lab
from autoforge.Helper.OptimizerHelper import composite_image_cont


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
    focus_strength: float = None,
) -> torch.Tensor:
    """
    Full forward pass for continuous assignment:
    composite, then compute unified loss on (global_logits).
    focus_map acts as a priority mask (values in [0,1]) where 1.0 means full weight and 0 means low weight.
    """
    comp = composite_image_cont(
        params["pixel_height_logits"],
        params["global_logits"],
        tau_height,
        tau_global,
        h,
        max_layers,
        material_colors,
        material_TDs,
        background,
    )
    return compute_loss(
        comp=comp,
        target=target,
        pixel_height_logits=params.get("pixel_height_logits", None),
        tau_height=tau_height,
        add_penalty_loss=add_penalty_loss,
        focus_map=focus_map,
        focus_strength=focus_strength,
    )


def compute_loss(
    comp: torch.Tensor,
    target: torch.Tensor,
    pixel_height_logits: torch.Tensor = None,
    tau_height: float = 1.0,
    add_penalty_loss: float = 0.0,
    focus_map: torch.Tensor = None,
    focus_strength: float = None,
) -> torch.Tensor:
    """
    Compute loss between composite and target.

    If focus_map (priority mask) is provided (shape [H,W] normalized 0..1), we apply per-pixel weights:
        weight = 0.1 + 0.9 * focus_map
    (So outside mask -> 0.1, fully prioritized -> 1.0, gradients respected.)
    The final loss is the weighted mean of per-pixel Lab-space MSE.
    We normalize by the mean weight to keep the magnitude comparable with the unweighted loss.
    """
    comp_lab = srgb_to_lab(comp)
    target_lab = srgb_to_lab(target)

    if focus_map is None:
        # standard mean squared error in Lab space
        mse_loss = F.mse_loss(comp_lab, target_lab)
        total_loss = mse_loss
        return total_loss

    # Ensure focus_map shape compatibility: [H,W]
    if focus_map.dim() == 3 and focus_map.shape[-1] == 1:
        focus_map_proc = focus_map.squeeze(-1)
    else:
        focus_map_proc = focus_map
    # Clamp/normalize safety
    focus_map_proc = torch.clamp(focus_map_proc, 0.0, 1.0)

    # Per-pixel MSE over Lab channels
    per_pixel_mse = (comp_lab - target_lab).pow(2).mean(dim=2)  # [H,W]

    weights = 0.1 + 0.9 * focus_map_proc  # [H,W] range [0.1, 1.0]
    weighted_loss = per_pixel_mse * weights
    # Normalize by average weight so scale comparable to original MSE
    total_loss = weighted_loss.mean() / weights.mean()

    return total_loss

    # (Additional smoothness and penalties are currently disabled.)
