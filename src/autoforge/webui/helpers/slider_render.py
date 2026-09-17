"""Recompute the composite preview image + colored mesh from an edited
color-slider stack, without re-running optimization.

The color sliders represent a HueForge-style stack: each enabled slider owns
a contiguous run of print layers (from the previous slider's layer + 1 up to
its own ``layer``) and assigns one filament (color + transmission distance)
to that run. Editing a slider only changes *which material occupies which
layer band* — the per-pixel height solution from the last completed
optimization is untouched. That means a full re-render only needs the
Beer-Lambert-style top-down compositing pass (the same physics as
``OptimizerHelper.composite_image_disc``), driven by the layer→material
mapping implied by the slider stack instead of the optimizer's learned
``global_logits``.
"""

from __future__ import annotations

import os
import threading
from typing import Any, Optional

import cv2
import numpy as np
import torch

from autoforge.Helper.FilamentHelper import hex_to_rgb
from autoforge.Helper.OptimizerHelper import bleed_layer_effect
from autoforge.webui.helpers.colored_mesh import generate_colored_preview_mesh

# Same empirical opacity-vs-thickness curve used by composite_image_disc /
# composite_image_cont — keep in sync with OptimizerHelper.py.
_OPAC_O, _OPAC_A, _OPAC_K, _OPAC_B = (
    -2.9864511e-02,
    4.0532556e-01,
    8.2597107e01,
    1.2547257e00,
)


def _atomic_write_bytes(path: str, data: bytes) -> None:
    tmp = f"{path}.{os.getpid()}.{threading.get_ident()}.tmp"
    with open(tmp, "wb") as f:
        f.write(data)
    os.replace(tmp, path)


def _dedupe_overlapping_layers(sliders: list[dict]) -> list[dict]:
    """Sliders may share a ``layer`` value — the frontend lets you drag one
    on top of another instead of blocking it. Only one can actually own that
    print layer though: mirroring the frontend's ColorCore/ColorSliders
    tie-break, the slider that appears last in column order (the highest
    original index, i.e. furthest right) wins; the rest are dropped here so
    the compositor never has two candidate materials for the same layer."""
    winner_by_layer: dict[int, int] = {}
    for orig_idx, s in enumerate(sliders):
        if not s.get("enabled") or int(s.get("layer", 0)) <= 0:
            continue
        layer = int(s["layer"])
        if layer not in winner_by_layer or orig_idx > winner_by_layer[layer]:
            winner_by_layer[layer] = orig_idx
    winners = set(winner_by_layer.values())
    return [s for i, s in enumerate(sliders) if i in winners]


def _layer_material_indices(sliders: list[dict], max_layers: int) -> np.ndarray:
    """Map each stack layer (0-indexed, length ``max_layers``) to a slot in
    ``sliders`` (already filtered to enabled, sorted by ``layer`` ascending).

    Layer band ownership mirrors the frontend ColorCore/derive_sliders logic:
    slider ``i`` owns print layers ``(sliders[i-1].layer, sliders[i].layer]``
    (1-indexed), with the first slider starting at layer 1. Any layers above
    the topmost slider's ``layer`` (e.g. the user dragged it down) keep that
    slider's material — there is no "no material" layer in a physical print.
    """
    idx = np.zeros(max_layers, dtype=np.int64)
    if not sliders:
        return idx
    boundaries = [int(s["layer"]) for s in sliders]
    for layer in range(1, max_layers + 1):
        owner = len(boundaries) - 1
        for i, b in enumerate(boundaries):
            if layer <= b:
                owner = i
                break
        idx[layer - 1] = owner
    return idx


def _resolve_slider_materials(
    sliders: list[dict], filament_lookup: dict[str, dict]
) -> tuple[np.ndarray, np.ndarray]:
    """Build (colors[N,3] in [0,1], tds[N]) arrays, one row per slider, in
    the same order as the (already sorted+filtered) ``sliders`` list."""
    colors = np.zeros((len(sliders), 3), dtype=np.float32)
    tds = np.zeros(len(sliders), dtype=np.float32)
    for i, s in enumerate(sliders):
        filament = filament_lookup.get(str(s.get("filament_uuid", "")))
        # The slider's own TD wins: it starts as the filament's TD (set on
        # drop / derived from the optimizer) and the TD box above each slider
        # column edits it. Preferring the library value made that box a
        # no-op for the render.
        slider_td = float(s.get("td") or 0.0)
        if filament is not None:
            colors[i] = hex_to_rgb(filament["color"])
            tds[i] = slider_td if slider_td > 0 else float(filament.get("td", 5.0))
        else:
            # No matching filament (e.g. a filament not in the library
            # anymore) — fall back to the slider's own td and a neutral
            # gray so the render still succeeds instead of erroring out.
            colors[i] = [0.5, 0.5, 0.5]
            tds[i] = float(s.get("td", 5.0))
    return colors, tds


def composite_from_slider_stack(
    disc_height_image: torch.Tensor,
    layer_material_idx: np.ndarray,
    material_colors: torch.Tensor,
    material_TDs: torch.Tensor,
    background: torch.Tensor,
    h: float,
    max_layers: int,
) -> torch.Tensor:
    """Beer-Lambert top-down compositing for an explicit, already-discrete
    layer→material assignment (no Gumbel-softmax sampling needed since the
    assignment is fully determined by the slider stack)."""
    device = disc_height_image.device
    idx_t = torch.as_tensor(layer_material_idx, dtype=torch.long, device=device)
    layer_colors = material_colors[idx_t]  # [L,3]
    layer_TDs = material_TDs[idx_t].clamp(1e-8, 1e8)  # [L]

    z_int = disc_height_image.to(torch.int64).clamp(0, max_layers)  # [H,W]
    layer_idx = torch.arange(max_layers, device=device).view(-1, 1, 1)
    p_print = (layer_idx < z_int.unsqueeze(0)).to(material_colors.dtype)  # [L,H,W]

    p_print_bleed = bleed_layer_effect(p_print, strength=0.1)
    eff_thick = torch.clamp(p_print_bleed, 0.0, 1.0) * h
    thick_ratio = eff_thick / layer_TDs.view(-1, 1, 1)

    opac = _OPAC_O + (_OPAC_A * torch.log1p(_OPAC_K * thick_ratio) + _OPAC_B * thick_ratio)
    opac = torch.clamp(opac, 0.0, 1.0)

    opac_fb = torch.flip(opac, dims=[0])
    colors_fb = torch.flip(layer_colors, dims=[0])
    trans_fb = 1.0 - opac_fb
    trans_prev = torch.cat([torch.ones_like(trans_fb[:1]), trans_fb[:-1]], dim=0)
    remain_fb = torch.cumprod(trans_prev, dim=0)

    comp_layers = (remain_fb * opac_fb).unsqueeze(-1) * colors_fb.view(-1, 1, 1, 3)
    comp = comp_layers.sum(dim=0)

    rem_after = remain_fb[-1] * trans_fb[-1]
    comp = comp + rem_after.unsqueeze(-1) * background

    return comp * 255.0


def render_with_sliders(
    pipeline_result: dict[str, Any],
    sliders: list[dict],
    filament_lookup: dict[str, dict],
    output_dir: str,
    png_name: str = "final_model.png",
    ply_name: str = "final_model_colored.ply",
) -> Optional[dict[str, Any]]:
    """Recompute the composite preview + colored PLY from an edited slider
    stack, writing ``png_name`` / ``ply_name`` into ``output_dir``.

    Returns ``{"image_b64": str, "preview_png": path, "colored_ply": path}``
    or ``None`` if there is no discretized solution to render yet.
    """
    optimizer = pipeline_result["optimizer"]
    args = pipeline_result["args"]
    alpha = pipeline_result.get("alpha")
    background: torch.Tensor = pipeline_result["background"]

    disc_global, disc_height_image = optimizer.get_discretized_solution(best=True)
    if disc_global is None or disc_height_image is None:
        return None

    max_layers = int(optimizer.max_layers)
    h = float(getattr(args, "layer_height", 0.04))

    enabled = sorted(_dedupe_overlapping_layers(sliders), key=lambda s: int(s["layer"]))
    if not enabled:
        return None

    layer_material_idx = _layer_material_indices(enabled, max_layers)
    colors_np, tds_np = _resolve_slider_materials(enabled, filament_lookup)

    device = disc_height_image.device
    material_colors = torch.tensor(colors_np, dtype=torch.float32, device=device)
    material_TDs = torch.tensor(tds_np, dtype=torch.float32, device=device)

    with torch.no_grad():
        comp = composite_from_slider_stack(
            disc_height_image,
            layer_material_idx,
            material_colors,
            material_TDs,
            background,
            h,
            max_layers,
        )

    comp_np = comp.detach().cpu().numpy().astype(np.uint8)
    comp_bgr = cv2.cvtColor(comp_np, cv2.COLOR_RGB2BGR)

    _ok, buf = cv2.imencode(".png", comp_bgr)
    image_b64 = None
    if _ok:
        import base64

        image_b64 = base64.b64encode(buf.tobytes()).decode("utf-8")

    os.makedirs(output_dir, exist_ok=True)
    preview_path = os.path.join(output_dir, png_name)
    # Written to a temp file and renamed into place: the 3D view / image panel
    # fetch these files right after every edit, and reading one while it was
    # being rewritten in place gave truncated or aborted responses.
    if _ok:
        _atomic_write_bytes(preview_path, buf.tobytes())
    else:
        cv2.imwrite(preview_path, comp_bgr)

    height_map_mm = disc_height_image.detach().cpu().numpy().astype(np.float32) * h
    color_image_np = np.ascontiguousarray(comp_np)
    colored_mesh = generate_colored_preview_mesh(
        height_map=height_map_mm,
        color_image=color_image_np,
        background_height=float(args.background_height),
        maximum_x_y_size=float(args.stl_output_size),
        alpha_mask=alpha,
    )
    ply_path = os.path.join(output_dir, ply_name)
    tmp_ply = f"{ply_path}.{os.getpid()}.{threading.get_ident()}.tmp.ply"
    colored_mesh.export(tmp_ply, encoding="binary")
    os.replace(tmp_ply, ply_path)

    return {"image_b64": image_b64, "preview_png": preview_path, "colored_ply": ply_path}
