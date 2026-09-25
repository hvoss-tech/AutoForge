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

import base64
import os
import threading
from typing import Any, Optional

import cv2
import numpy as np
import torch

from autoforge.Helper.FilamentHelper import hex_to_rgb
from autoforge.Helper.OptimizerHelper import bleed_layer_effect
from autoforge.webui.helpers.colored_mesh import (
    generate_colored_preview_mesh,
    preview_mesh_max_dim,
    top_vertex_pixel_indices,
)

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


def _alpha_matching_shape(pipeline_result: dict[str, Any], target_hw: tuple[int, int]) -> Optional[np.ndarray]:
    """Return an (H,W[,1]) uint8/float alpha mask matching ``target_hw``.

    ``disc_height_image`` is at *processing* resolution for a not-yet-trained
    auto-preview result (``build_init_preview`` seeds it from
    ``get_current_parameters()`` without ever calling ``export_results()``,
    which is what restores the full-resolution height map for a real
    optimization job). ``pipeline_result["alpha"]`` is always full/output
    resolution though, so blindly using it against a processing-resolution
    height map produced an alpha mask larger than the height grid --
    ``generate_colored_preview_mesh`` then emitted face indices past the end
    of its vertex array (IndexError, or a silently mangled mesh when the
    sizes happened to be close enough not to crash). Prefer whichever stored
    alpha already matches, falling back to a nearest-neighbor resize of the
    full-res one so a mismatch can never reach the mesh builder.
    """

    def _to_np(a: Any) -> Optional[np.ndarray]:
        if a is None:
            return None
        return a.detach().cpu().numpy() if torch.is_tensor(a) else np.asarray(a)

    alpha_proc_np = _to_np(pipeline_result.get("alpha_proc"))
    if alpha_proc_np is not None and tuple(alpha_proc_np.shape[:2]) == target_hw:
        return alpha_proc_np

    alpha_np = _to_np(pipeline_result.get("alpha"))
    if alpha_np is None:
        return None
    if tuple(alpha_np.shape[:2]) == target_hw:
        return alpha_np

    resized = cv2.resize(
        alpha_np.astype(np.float32),
        (target_hw[1], target_hw[0]),
        interpolation=cv2.INTER_NEAREST,
    )
    if resized.ndim == 2:
        resized = resized[..., None]
    return resized


def compute_slider_render(
    pipeline_result: dict[str, Any],
    sliders: list[dict],
    filament_lookup: dict[str, dict],
    background_rgb: Optional[tuple[float, float, float]] = None,
) -> Optional[dict[str, Any]]:
    """The fast part of a slider edit: composite the edited stack and encode
    the preview PNG — no files written, no mesh built.

    ``background_rgb`` (0-1 floats) overrides the pipeline's own resolved
    background color — without it, editing the base color in the UI had no
    effect on the composite, since the render always fell back to whatever
    color the optimization run happened to settle on.

    Returns None when there is no discretized solution (or nothing enabled).
    Everything in the result is plain numpy/bytes, so it can outlive the
    pipeline result (a background PLY write keeps no GPU memory alive).
    """
    optimizer = pipeline_result["optimizer"]
    args = pipeline_result["args"]
    background: torch.Tensor = pipeline_result["background"]
    if background_rgb is not None:
        background = torch.tensor(background_rgb, dtype=background.dtype, device=background.device)

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

    comp_np = np.ascontiguousarray(comp.detach().cpu().numpy().astype(np.uint8))
    comp_bgr = cv2.cvtColor(comp_np, cv2.COLOR_RGB2BGR)
    ok, buf = cv2.imencode(".png", comp_bgr)
    png_bytes = buf.tobytes() if ok else None

    height_map_mm = disc_height_image.detach().cpu().numpy().astype(np.float32) * h
    alpha_for_mesh = _alpha_matching_shape(pipeline_result, height_map_mm.shape[:2])
    return {
        "comp_np": comp_np,
        "png_bytes": png_bytes,
        "image_b64": base64.b64encode(png_bytes).decode("utf-8") if png_bytes else None,
        "height_map_mm": height_map_mm,
        "alpha": alpha_for_mesh,
        "background_height": float(args.background_height),
        "stl_output_size": float(args.stl_output_size),
        "top_vertex_pixels": _cached_top_vertex_pixels(pipeline_result, height_map_mm.shape[:2], alpha_for_mesh),
    }


def _cached_top_vertex_pixels(pipeline_result: dict[str, Any], hw: tuple[int, int], alpha) -> np.ndarray:
    """Which pixel each top mesh vertex shows. It depends only on the grid
    size and the alpha mask — never on heights or colors — so it is computed
    once per result instead of on every edit."""
    key = (int(hw[0]), int(hw[1]), preview_mesh_max_dim(),
           id(pipeline_result.get("alpha")), id(pipeline_result.get("alpha_proc")))
    cached = pipeline_result.get("_top_vertex_pixels")
    if cached is not None and cached[0] == key:
        return cached[1]
    idx = top_vertex_pixel_indices(int(hw[0]), int(hw[1]), alpha)
    pipeline_result["_top_vertex_pixels"] = (key, idx)
    return idx


def top_vertex_colors(render: dict[str, Any]) -> bytes:
    """RGB bytes for the mesh's top vertices, in vertex order (see
    ``top_vertex_pixel_indices``) — all a client needs to recolor the mesh
    it already shows. The bottom vertices never change color."""
    return render["comp_np"].reshape(-1, 3)[render["top_vertex_pixels"]].tobytes()


def write_slider_png(render: dict[str, Any], output_dir: str, png_name: str) -> str:
    os.makedirs(output_dir, exist_ok=True)
    preview_path = os.path.join(output_dir, png_name)
    # Written to a temp file and renamed into place: the 3D view / image panel
    # fetch these files right after every edit, and reading one while it was
    # being rewritten in place gave truncated or aborted responses.
    if render["png_bytes"] is not None:
        _atomic_write_bytes(preview_path, render["png_bytes"])
    else:
        cv2.imwrite(preview_path, cv2.cvtColor(render["comp_np"], cv2.COLOR_RGB2BGR))
    return preview_path


def write_slider_ply(render: dict[str, Any], output_dir: str, ply_name: str) -> str:
    """The slow part: build and export the colored mesh (~150ms and ~25MB at
    default settings). Only needed so the edit survives a reload/undo — a
    live edit reaches the 3D view as vertex colors instead."""
    colored_mesh = generate_colored_preview_mesh(
        height_map=render["height_map_mm"],
        color_image=render["comp_np"],
        background_height=render["background_height"],
        maximum_x_y_size=render["stl_output_size"],
        alpha_mask=render["alpha"],
    )
    os.makedirs(output_dir, exist_ok=True)
    ply_path = os.path.join(output_dir, ply_name)
    tmp_ply = f"{ply_path}.{os.getpid()}.{threading.get_ident()}.tmp.ply"
    colored_mesh.export(tmp_ply, encoding="binary")
    os.replace(tmp_ply, ply_path)
    return ply_path


def render_with_sliders(
    pipeline_result: dict[str, Any],
    sliders: list[dict],
    filament_lookup: dict[str, dict],
    output_dir: str,
    png_name: str = "final_model.png",
    ply_name: str = "final_model_colored.ply",
    background_rgb: Optional[tuple[float, float, float]] = None,
) -> Optional[dict[str, Any]]:
    """Recompute the composite preview + colored PLY from an edited slider
    stack, writing ``png_name`` / ``ply_name`` into ``output_dir``.

    Returns ``{"image_b64": str, "preview_png": path, "colored_ply": path}``
    or ``None`` if there is no discretized solution to render yet.
    """
    render = compute_slider_render(pipeline_result, sliders, filament_lookup, background_rgb=background_rgb)
    if render is None:
        return None
    preview_path = write_slider_png(render, output_dir, png_name)
    ply_path = write_slider_ply(render, output_dir, ply_name)
    return {"image_b64": render["image_b64"], "preview_png": preview_path, "colored_ply": ply_path}
