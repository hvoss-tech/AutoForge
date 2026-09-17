"""Derive the ColorSlider stack from a discrete optimization solution.

The WebUI represents the print as a HueForge-style stack of material
"sliders" — one per contiguous run of print layers using the same material.
The stack is read off the optimizer's discrete solution:

* ``disc_global`` is a 1D array of material indices, one per print layer.
* ``disc_height_image`` is the per-pixel height map (in layers); its
  min/max bound the layer range the sliders are allowed to occupy.

The number of sliders is *not* capped: the optimizer can legitimately
produce more (or fewer) than any fixed UI column count — a material can
recur in several non-contiguous layer bands, and pruning further changes
the band count. Truncating or merging bands here would silently discard
part of the real result, and since ``render_with_sliders`` reconstructs the
whole composite/mesh from exactly this slider list, a truncated stack
doesn't just mis-render the UI — it produces a genuinely different (wrong)
3D preview the moment anything re-triggers that reconstruction. The
frontend is responsible for displaying however many columns this returns
(with horizontal scrolling), not this module for pretending there are
fewer than there really are.
"""

import numpy as np


def _slider_dict(
    material: int,
    end_layer: int,
    layer_height: float,
    material_tds: np.ndarray,
    material_uuids: list,
) -> dict:
    return {
        "td": float(material_tds[material]) if material < len(material_tds) else 5.0,
        "layer": int(end_layer),
        "depth_mm": round(float(end_layer) * float(layer_height), 2),
        "filament_uuid": (
            material_uuids[material] if material < len(material_uuids) else ""
        ),
        "enabled": True,
    }


def derive_sliders_from_optimizer(
    optimizer,
    material_tds: np.ndarray,
    material_uuids: list,
    layer_height: float,
):
    """Return ``{"sliders": [...], "min_layer": int, "max_layer": int}``.

    The sliders use snake_case keys matching the frontend ``ColorSliderConfig``
    type.  Returns ``None`` when the optimizer has no discretized solution yet.
    """
    disc_global, disc_height_image = optimizer.get_discretized_solution(best=True)
    if disc_global is None or disc_height_image is None:
        return None

    disc_global = disc_global.detach().cpu().numpy().reshape(-1).astype(int)
    height_map = disc_height_image.detach().cpu().numpy()
    min_layer = int(height_map.min())
    max_layer = int(height_map.max())

    if max_layer <= 0:
        return {"sliders": [], "min_layer": min_layer, "max_layer": max_layer}

    # Walk the printed stack (print layers 1..max_layer) and group
    # consecutive layers that use the same material into segments.
    n_stack = min(int(disc_global.shape[0]), max_layer)
    segments: list[list[int]] = []  # [material, start_layer, end_layer]
    current_material = None
    start = 1
    for layer in range(1, n_stack + 1):
        material = int(disc_global[layer - 1])
        if material != current_material:
            if current_material is not None:
                segments.append([current_material, start, layer - 1])
            current_material = material
            start = layer
    if current_material is not None:
        segments.append([current_material, start, n_stack])

    sliders = [
        _slider_dict(material, end, layer_height, material_tds, material_uuids)
        for material, _start, end in segments
    ]

    return {"sliders": sliders, "min_layer": min_layer, "max_layer": max_layer}


def should_repeat_prune(
    pass_index: int,
    max_passes: int,
    loss: float | None,
    best_loss: float | None,
    epsilon: float = 1e-4,
) -> tuple[bool, str]:
    """Decide whether auto-repeat pruning should run another pass.

    Returns ``(repeat, reason)``; ``reason`` is what to tell the user when it
    stops. A pass has to beat the best loss so far by more than ``epsilon`` to
    count — the loss is a mean over ~10^5 pixels, so a difference smaller than
    that is the same solution re-scored, and treating it as progress would
    loop until the pass limit for nothing.

    A pass whose loss couldn't be measured counts as progress: stopping on a
    missing measurement would silently cut the run short.
    """
    if pass_index >= max_passes:
        return False, f"reached the {max_passes}-pass limit"
    if loss is None:
        return True, ""
    if best_loss is not None and loss >= best_loss - epsilon:
        return False, "no further improvement"
    return True, ""


def result_counts_from_optimizer(optimizer) -> dict | None:
    """``{"colors", "swaps", "layers"}`` for the optimizer's current discrete
    solution, counted exactly the way the pruning dialog counts them:

    * ``colors`` — distinct materials in the printed stack, **plus one** for
      the base/background color (this is the number the pruner's
      ``max_colors`` limits).
    * ``swaps`` — filament changes after the first color layer, i.e. one
      fewer than the number of contiguous same-material runs.
    * ``layers`` — the top print layer.

    This is what makes the pruning overlay live: pruning mutates the
    optimizer's solution in place through several phases, and without asking
    the solution itself the UI could only show the counts it started with
    until the whole run was over.
    """
    disc_global, disc_height_image = optimizer.get_discretized_solution(best=True)
    if disc_global is None or disc_height_image is None:
        return None

    materials = disc_global.detach().cpu().numpy().reshape(-1).astype(int)
    max_layer = int(disc_height_image.detach().cpu().numpy().max())
    stack = materials[: max(0, min(len(materials), max_layer))]
    if stack.size == 0:
        return {"colors": 1, "swaps": 0, "layers": max(0, max_layer)}

    runs = 1 + int(np.count_nonzero(stack[1:] != stack[:-1]))
    return {
        "colors": int(np.unique(stack).size) + 1,
        "swaps": runs - 1,
        "layers": max_layer,
    }


def derive_layer_range_from_result(result: dict):
    """Return just ``{"min_layer": int, "max_layer": int}`` for a pipeline
    result, without grouping the discrete solution into material segments.

    Used for the post-upload "auto-preview" flow (``api/init.py``): at that
    point the optimizer hasn't been trained, so ``disc_global`` (the
    layer->material assignment ``derive_sliders_from_result`` groups into
    segments) is meaningless — but ``disc_height_image`` (pure per-pixel
    height, from the heightmap-initialization algorithm) is already the
    real result, and the slider track's range should reflect it immediately
    rather than staying at a placeholder default.
    """
    optimizer = result["optimizer"]
    disc_global, disc_height_image = optimizer.get_discretized_solution(best=True)
    if disc_height_image is None:
        return None
    height_map = disc_height_image.detach().cpu().numpy()
    return {"min_layer": int(height_map.min()), "max_layer": int(height_map.max())}


def derive_base_from_result(result: dict) -> dict:
    """Describe the base/background slab the color layers are printed on.

    The base is a real part of the print — it is the first thing that comes
    off the plate and its color shows through every translucent layer above
    it — but it is not one of the optimizer's layer bands, so it never
    appeared in ``derive_sliders_*``. The UI still needs to show it (and let
    the user swap which filament it is), and it specifically needs the
    *resolved* color: with ``auto_background_color`` on, the pipeline picks
    the active filament closest to the image's dominant color and overwrites
    ``args.background_color`` with it, so the setting the frontend sent is
    not what was actually printed.

    Returns ``{"color", "height_mm", "layers", "filament_uuid", "auto"}``.
    ``filament_uuid`` is empty when the color doesn't correspond to one of
    the active filaments (a hand-picked hex, or auto-selection failing).
    """
    args = result.get("args")
    material_uuids = result.get("material_uuids", []) or []

    color = str(getattr(args, "background_color", "#000000") or "#000000")
    height_mm = float(getattr(args, "background_height", 0.0) or 0.0)
    layer_height = float(getattr(args, "layer_height", 0.04) or 0.04)

    index = getattr(args, "background_material_index", None)
    filament_uuid = ""
    if isinstance(index, (int, np.integer)) and 0 <= int(index) < len(material_uuids):
        filament_uuid = str(material_uuids[int(index)])
    else:
        # Not auto-selected (or the index didn't survive): fall back to
        # matching the hex against the active filaments, so a background that
        # *is* one of them still shows up as that filament in the UI.
        colors = result.get("colors_list") or []
        target = color.strip().lower()
        for i, hex_color in enumerate(colors):
            if str(hex_color).strip().lower() == target and i < len(material_uuids):
                filament_uuid = str(material_uuids[i])
                break

    return {
        "color": color,
        "height_mm": round(height_mm, 4),
        "layers": int(round(height_mm / layer_height)) if layer_height > 0 else 0,
        "filament_uuid": filament_uuid,
        "auto": bool(getattr(args, "auto_background_color", False)),
    }


def derive_sliders_from_result(result: dict):
    """Derive sliders from a pipeline result dict (see ``run_pipeline``)."""
    optimizer = result["optimizer"]
    material_tds = result.get("material_TDs_np")
    material_uuids = result.get("material_uuids", [])
    args = result.get("args")
    layer_height = float(getattr(args, "layer_height", 0.04)) if args else 0.04
    if material_tds is None:
        num_materials = int(optimizer.material_colors.shape[0])
        material_tds = np.zeros(num_materials, dtype=np.float64)
    return derive_sliders_from_optimizer(
        optimizer, material_tds, material_uuids, layer_height
    )
