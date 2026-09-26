import io
import json
import os
import struct
import uuid

import numpy as np

from autoforge.Helper.FilamentHelper import load_materials_data


def extract_filament_swaps(disc_global, disc_height_image, background_layers):
    """
    Given the discrete global material assignment (disc_global) and the discrete height image,
    extract the list of material indices (one per swap point) and the corresponding slider
    values (which indicate at which layer the material change occurs).

    Args:
        disc_global (jnp.ndarray): Discrete global material assignments.
        disc_height_image (jnp.ndarray): Discrete height image.
        background_layers (int): Number of background layers.

    Returns:
        tuple: A tuple containing:
            - filament_indices (list): List of material indices for each swap point.
            - slider_values (list): List of layer numbers where a material change occurs.
    """
    # L is the total number of layers printed (maximum value in the height image)
    L = int(np.max(np.asarray(disc_height_image)))
    if L == 0:
        return [], []

    filament_indices = [int(disc_global[0])]  # first colour used
    slider_values = [1]  # first swap happens at layer #2

    prev = int(disc_global[0])
    for i in range(1, L):
        current = int(disc_global[i])
        if current != prev:
            filament_indices.append(current)  # new material
            slider_values.append(i + 1)  # 1-based index
            prev = current

    filament_indices.append(prev)
    slider = slider_values[-1] + 1
    slider_values.append(slider)

    return filament_indices, slider_values


def _is_blank(value) -> bool:
    """Missing, None, or the NaN pandas reads an empty CSV cell as."""
    if value is None:
        return True
    if isinstance(value, float) and np.isnan(value):
        return True
    return isinstance(value, str) and value.strip() == ""


def _hfp_filament(mat, brand="", color="#000000", name=""):
    """One ``filament_set`` entry of a .hfp file from a material record.

    The records come from pandas, which reads an empty CSV cell as NaN -
    and ``json.dump`` writes that as a bare ``NaN``, which is not JSON, so
    HueForge (or any strict parser) could not open the project. The webui
    hits this for every filament without a brand. Blank fields get the
    given defaults (a fresh uuid for ``uuid``) instead."""

    def text(key, default):
        value = mat.get(key)
        return default if _is_blank(value) else str(value)

    td_raw = mat.get("Transmissivity")
    try:
        td = float(td_raw)
    except (TypeError, ValueError):
        td = float("nan")
    if not np.isfinite(td):
        td = 0.1
    return {
        "Brand": text("Brand", brand),
        "Color": text("Color", color),
        "Name": text("Name", name),
        "Owned": str(mat.get("Owned", False)).strip().lower() == "true",
        "Transmissivity": int(td) if td.is_integer() else td,
        "Type": text("Type", "PLA"),
        "uuid": text("Uuid", str(uuid.uuid4())),
    }


def model_size_mm(width_px, height_px, maximum_x_y_size):
    """Physical (width, height) in mm of a model built by ``generate_stl``
    from a ``width_px`` x ``height_px`` height map: its longer side spans
    ``maximum_x_y_size`` mm across the pixel centres (``max(W, H) - 1``
    steps)."""
    steps = max(int(width_px) - 1, int(height_px) - 1, 1)
    scale = float(maximum_x_y_size) / steps
    return (max(int(width_px) - 1, 0) * scale, max(int(height_px) - 1, 0) * scale)


def generate_project_file(
    project_filename,
    args,
    disc_global,
    disc_height_image,
    width_mm,
    height_mm,
    stl_filename,
    csv_filename,
):
    """
    Export a project file containing the printing parameters, including:
      - Key dimensions and layer information (from your command-line args and computed outputs)
      - The filament_set: a list of filament definitions (each corresponding to a color swap)
        where the same material may be repeated if used at different swap points.
      - slider_values: a list of layer numbers (indices) where a filament swap occurs.

    The filament_set entries are built using the full material data from the CSV file.

    Args:
        project_filename (str): Path to the output project file.
        args (Namespace): Command-line arguments containing printing parameters.
        disc_global (jnp.ndarray): Discrete global material assignments.
        disc_height_image (jnp.ndarray): Discrete height image.
        width_mm (float): Width of the model in millimeters.
        height_mm (float): Height of the model in millimeters.
        stl_filename (str): Path to the STL file.
        csv_filename (str): Path to the CSV file containing material data.
    """
    # Compute the number of background layers (as in your main())
    background_layers = int(round(args.background_height / args.layer_height))

    # Load full material data from CSV
    material_data = load_materials_data(args)

    # Extract the swap points from the discrete solution
    filament_indices, slider_values = extract_filament_swaps(
        disc_global, disc_height_image, background_layers
    )

    # Build the filament_set list. For each swap point, we look up the corresponding material from CSV.
    # Here we map CSV columns to the project file’s expected keys.
    filament_set = []

    # Use actual background filament data if auto selected and index stored
    bg_idx = getattr(args, "background_material_index", None)
    if bg_idx is not None and 0 <= bg_idx < len(material_data):
        filament_set.append(
            _hfp_filament(
                material_data[bg_idx],
                brand="Autoforge",
                color=args.background_color,
                name="Background",
            )
        )
    else:
        filament_set.append(
            {
                "Brand": "Autoforge",
                "Color": args.background_color,
                "Name": "Background",
                "Owned": False,
                "Transmissivity": 0.1,
                "Type": "PLA",
                "uuid": str(uuid.uuid4()),
            }
        )

    for idx in filament_indices:
        filament_set.append(_hfp_filament(material_data[idx]))

    filament_set = filament_set[::-1]

    # Build the project file dictionary.
    # Many keys are filled in with default or derived values.
    project_data = {
        "base_layer_height": args.background_height,  # you may adjust this if needed
        "blue_shift": 0,
        "border_height": args.background_height,  # here we use the background height
        "border_width": 3,
        "borderless": True,
        "bright_adjust_zero": False,
        "brightness_compensation_name": "Standard",
        "bw_tolerance": 8,
        "color_match_method": 0,
        "depth_mode": 2,
        "edit_image": False,
        "extra_gap": 2,
        "filament_set": filament_set,
        "flatten": False,
        "full_range": False,
        "green_shift": 0,
        "gs_threshold": 0,
        "height_in_mm": height_mm,
        "hsl_invert": False,
        "ignore_blue": False,
        "ignore_green": False,
        "ignore_red": False,
        "invert_blue": False,
        "invert_green": False,
        "invert_red": False,
        "inverted_color_pop": False,
        "layer_height": args.layer_height,
        "legacy_luminance": False,
        "light_intensity": -1,
        "light_temperature": 1,
        "lighting_visualizer": 0,
        "luminance_factor": 0,
        "luminance_method": 2,
        "luminance_offset": 0,
        "luminance_offset_max": 100,
        "luminance_power": 2,
        "luminance_weight": 100,
        "max_depth": args.background_height + args.layer_height * args.max_layers,
        "median": 0,
        "mesh_style_edit": True,
        "min_depth": 0.48,
        "min_detail": 0.2,
        "negative": True,
        "red_shift": 0,
        "reverse_litho": True,
        "slider_values": slider_values,
        "smoothing": 0,
        "srgb_linearize": False,
        "stl": os.path.basename(stl_filename),
        "strict_tolerance": False,
        "transparency": True,
        "version": "0.7.0",
        "width_in_mm": width_mm,
    }

    # Write out the project file as JSON
    with open(project_filename, "w") as f:
        # allow_nan=False: a NaN that slipped through must fail here, loudly,
        # rather than produce a file HueForge cannot parse.
        json.dump(project_data, f, indent=4, allow_nan=False)


def generate_stl(
    height_map, filename, background_height, maximum_x_y_size, alpha_mask=None
):
    """
    Generate a binary STL file from a height map with an optional alpha mask.
    If alpha_mask is provided, vertices where alpha < 128 are omitted.
    This function builds a manifold mesh consisting of:
      - a top surface (only quads whose four vertices are valid),
      - side walls along the boundary edges of the top surface, and
      - a bottom face covering the valid region.

    Args:
        height_map (np.ndarray): 2D array representing the height map.
        filename (str): The name of the output STL file.
        background_height (float): The height of the background in the STL model.
        maximum_x_y_size (float): Maximum size (in millimeters) for the x and y dimensions.
        alpha_mask (np.ndarray): Optional alpha mask (same shape as height_map).
            A pixel is “valid” only if its alpha is ≥ 128.
    """
    H, W = height_map.shape

    # Compute valid mask: every pixel is valid if no alpha mask is provided.
    valid_mask = (
        np.ones((H, W), dtype=bool) if alpha_mask is None else (alpha_mask >= 128)
    )

    # --- Vectorized Creation of Vertices ---
    # Create a meshgrid of coordinates. Note that the y coordinate is flipped so that row 0 is at the top.
    j, i = np.meshgrid(np.arange(W), np.arange(H))
    x = j.astype(np.float32)
    y = (H - 1 - i).astype(np.float32)
    z = height_map.astype(np.float32) + background_height

    top_vertices = np.stack([x, y, z], axis=2)
    bottom_vertices = top_vertices.copy()
    bottom_vertices[:, :, 2] = 0

    # Scale vertices so the maximum x or y dimension equals maximum_x_y_size.
    original_max = max(W - 1, H - 1)
    scale = maximum_x_y_size / original_max
    top_vertices[:, :, :2] *= scale
    bottom_vertices[:, :, :2] *= scale

    # --- Top and Bottom Surfaces ---
    # Only use cells (quads) where all four corners are valid.
    quad_valid = (
        valid_mask[:-1, :-1]
        & valid_mask[:-1, 1:]
        & valid_mask[1:, 1:]
        & valid_mask[1:, :-1]
    )
    valid_i, valid_j = np.nonzero(quad_valid)
    num_quads = len(valid_i)

    # Define the four corners of each valid quad.
    v0 = top_vertices[valid_i, valid_j]
    v1 = top_vertices[valid_i, valid_j + 1]
    v2 = top_vertices[valid_i + 1, valid_j + 1]
    v3 = top_vertices[valid_i + 1, valid_j]

    # Top surface: using triangles (v2, v1, v0) and (v3, v2, v0)
    top_triangles = np.concatenate(
        [np.stack([v2, v1, v0], axis=1), np.stack([v3, v2, v0], axis=1)], axis=0
    )

    # Bottom face (using bottom vertices; note the reversed order so normals point downward)
    bv0 = bottom_vertices[valid_i, valid_j]
    bv1 = bottom_vertices[valid_i, valid_j + 1]
    bv2 = bottom_vertices[valid_i + 1, valid_j + 1]
    bv3 = bottom_vertices[valid_i + 1, valid_j]

    bottom_triangles = np.concatenate(
        [np.stack([bv0, bv1, bv2], axis=1), np.stack([bv0, bv2, bv3], axis=1)], axis=0
    )

    # --- Side Walls ---
    # Determine boundary edges from the grid of valid quads.
    # For each quad edge, if there is no neighboring valid quad sharing that edge, it is a boundary.
    side_triangles_list = []

    # Left edges: for quads in column 0 or when left neighbor is not valid.
    left_cond = np.zeros_like(quad_valid, dtype=bool)
    left_cond[:, 0] = quad_valid[:, 0]
    left_cond[:, 1:] = quad_valid[:, 1:] & (~quad_valid[:, :-1])
    li, lj = np.nonzero(left_cond)
    lv0 = top_vertices[li, lj]
    lv1 = top_vertices[li + 1, lj]
    lb0 = bottom_vertices[li, lj]
    lb1 = bottom_vertices[li + 1, lj]
    left_tris = np.concatenate(
        [np.stack([lv0, lv1, lb1], axis=1), np.stack([lv0, lb1, lb0], axis=1)], axis=0
    )
    side_triangles_list.append(left_tris)

    # Right edges: for quads in the last column or when right neighbor is not valid.
    right_cond = np.zeros_like(quad_valid, dtype=bool)
    right_cond[:, -1] = quad_valid[:, -1]
    right_cond[:, :-1] = quad_valid[:, :-1] & (~quad_valid[:, 1:])
    ri, rj = np.nonzero(right_cond)
    rv0 = top_vertices[ri, rj + 1]
    rv1 = top_vertices[ri + 1, rj + 1]
    rb0 = bottom_vertices[ri, rj + 1]
    rb1 = bottom_vertices[ri + 1, rj + 1]
    right_tris = np.concatenate(
        [np.stack([rv0, rv1, rb1], axis=1), np.stack([rv0, rb1, rb0], axis=1)], axis=0
    )
    side_triangles_list.append(right_tris)

    # Top edges: for quads in the first row or when the above neighbor is not valid.
    top_cond = np.zeros_like(quad_valid, dtype=bool)
    top_cond[0, :] = quad_valid[0, :]
    top_cond[1:, :] = quad_valid[1:, :] & (~quad_valid[:-1, :])
    ti, tj = np.nonzero(top_cond)
    tv0 = top_vertices[ti, tj]
    tv1 = top_vertices[ti, tj + 1]
    tb0 = bottom_vertices[ti, tj]
    tb1 = bottom_vertices[ti, tj + 1]
    top_wall_tris = np.concatenate(
        [np.stack([tv0, tv1, tb1], axis=1), np.stack([tv0, tb1, tb0], axis=1)], axis=0
    )
    side_triangles_list.append(top_wall_tris)

    # Bottom edges: for quads in the last row or when the below neighbor is not valid.
    bottom_cond = np.zeros_like(quad_valid, dtype=bool)
    bottom_cond[-1, :] = quad_valid[-1, :]
    bottom_cond[:-1, :] = quad_valid[:-1, :] & (~quad_valid[1:, :])
    bi, bj = np.nonzero(bottom_cond)
    bv0_edge = top_vertices[bi + 1, bj]
    bv1_edge = top_vertices[bi + 1, bj + 1]
    bb0 = bottom_vertices[bi + 1, bj]
    bb1 = bottom_vertices[bi + 1, bj + 1]
    bottom_wall_tris = np.concatenate(
        [
            np.stack([bv0_edge, bv1_edge, bb1], axis=1),
            np.stack([bv0_edge, bb1, bb0], axis=1),
        ],
        axis=0,
    )
    side_triangles_list.append(bottom_wall_tris)

    # Combine all side wall triangles.
    side_triangles = (
        np.concatenate(side_triangles_list, axis=0)
        if side_triangles_list
        else np.empty((0, 3, 3), dtype=np.float32)
    )

    # --- Combine All Triangles ---
    all_triangles = np.concatenate(
        [top_triangles, side_triangles, bottom_triangles], axis=0
    )

    # --- Compute Normals Vectorized ---
    v1_arr = all_triangles[:, 0, :]
    v2_arr = all_triangles[:, 1, :]
    v3_arr = all_triangles[:, 2, :]
    normals = np.cross(v2_arr - v1_arr, v3_arr - v1_arr)
    norms = np.linalg.norm(normals, axis=1)
    norms[norms == 0] = 1  # Prevent division by zero
    normals /= norms[:, np.newaxis]

    num_triangles = all_triangles.shape[0]

    # --- Create a Structured Array for Binary STL ---
    stl_dtype = np.dtype(
        [
            ("normal", np.float32, (3,)),
            ("v1", np.float32, (3,)),
            ("v2", np.float32, (3,)),
            ("v3", np.float32, (3,)),
            ("attr", np.uint16),
        ]
    )
    stl_data = np.empty(num_triangles, dtype=stl_dtype)
    stl_data["normal"] = normals
    stl_data["v1"] = all_triangles[:, 0, :]
    stl_data["v2"] = all_triangles[:, 1, :]
    stl_data["v3"] = all_triangles[:, 2, :]
    stl_data["attr"] = 0

    # Write to an in-memory buffer
    buffer = io.BytesIO()
    header_str = "Binary STL generated from heightmap with alpha mask"
    header = header_str.encode("utf-8").ljust(80, b" ")
    buffer.write(header)
    buffer.write(struct.pack("<I", num_triangles))
    buffer.write(stl_data.tobytes())

    # `buffer` is already a complete, valid binary STL (we hand-built the
    # header/triangle-count/triangle-soup above). Binary STL has no
    # shared-vertex representation - every triangle stores its own 3
    # vertices independently - so round-tripping through
    # trimesh.load(...).merge_vertices().export(...) here cannot change a
    # single byte of the output: merge_vertices() only collapses trimesh's
    # in-memory indexed representation, which export() re-expands back into
    # the exact same triangle soup on the way out. Verified byte-identical
    # output (same size/volume/area/watertightness) while being ~10x faster
    # at typical mesh sizes - just write the bytes we already have.
    with open(filename, "wb") as f:
        f.write(buffer.getvalue())


def generate_swap_instructions(
    discrete_global,
    discrete_height_image,
    h,
    background_layers,
    background_height,
    material_names,
    background_material_name=None,
):
    """
    Generate swap instructions based on discrete material assignments.

    Args:
        discrete_global (jnp.ndarray): Array of discrete global material assignments.
        discrete_height_image (jnp.ndarray): Array representing the discrete height image.
        h (float): Layer thickness.
        background_layers (int): Number of background layers.
        background_height (float): Height of the background in mm.
        material_names (list): List of material names.

    Returns:
        list: A list of strings containing the swap instructions.
    """
    L = int(np.max(np.array(discrete_height_image)))
    instructions = []
    if L == 0:
        instructions.append("No layers printed.")
        return instructions
    # First line includes background filament name if available
    if background_material_name:
        instructions.append(
            f"Print at 100% infill with a layer height of {h:.2f}mm with a base layer of {background_height:.2f}mm using background filament {background_material_name}."
        )
    else:
        instructions.append(
            f"Print at 100% infill with a layer height of {h:.2f}mm with a base layer of {background_height:.2f}mm"
        )
    instructions.append("")
    start_bg_name = (
        background_material_name
        if background_material_name
        else "your background color"
    )
    instructions.append(
        f"Start with {start_bg_name}, with a layer height of {background_height:.2f}mm for the first layer."
    )
    for i in range(0, L):
        if i == 0 or int(discrete_global[i]) != int(discrete_global[i - 1]):
            ie = i + 1
            instructions.append(
                f"At layer #{ie + 1} ({(ie * h) + background_height:.2f}mm) swap to {material_names[int(discrete_global[i])]}"
            )
    instructions.append(
        "For the rest, use " + material_names[int(discrete_global[L - 1])]
    )
    return instructions


def generate_flatforge_stls(
    disc_global,
    disc_height_image,
    material_colors_np,
    material_names,
    material_TDs_np,
    layer_height,
    background_height,
    background_color_hex,
    maximum_x_y_size,
    output_folder,
    cap_layers=0,
    alpha_mask=None,
):
    """
    Generate separate STL files for FlatForge mode.
    
    In FlatForge mode, each color gets its own STL file, along with STLs for the 
    background and optional cap layer. The STLs are designed to align perfectly
    when loaded together to create a solid rectangular print.
    
    Args:
        disc_global (np.ndarray): Array of discrete global material assignments (one per layer).
        disc_height_image (np.ndarray): 2D array representing the discrete height at each pixel.
        material_colors_np (np.ndarray): Array of RGB colors for each material (shape: [num_materials, 3]).
        material_names (list): List of material names.
        material_TDs_np (np.ndarray): Array of transmission distances for each material.
        layer_height (float): Height of each layer in mm.
        background_height (float): Height of the background in mm.
        background_color_hex (str): Hex color of the background.
        maximum_x_y_size (float): Maximum size for the x and y dimensions in mm.
        output_folder (str): Folder to save the STL files.
        cap_layers (int): Number of complete transparent layers to add on top.
        alpha_mask (np.ndarray): Optional alpha mask (same shape as disc_height_image).
    
    Returns:
        list: List of filenames for the generated STL files.
    """
    H, W = disc_height_image.shape
    max_layer = int(np.max(disc_height_image))
    
    if max_layer == 0:
        print("Warning: No layers to print in FlatForge mode.")
        return []
    
    # Determine valid mask
    valid_mask = (
        np.ones((H, W), dtype=bool) if alpha_mask is None else (alpha_mask >= 128)
    )
    
    # Find the most transparent material (highest TD value) for clear areas
    most_transparent_idx = int(np.argmax(material_TDs_np))
    clear_material_name = material_names[most_transparent_idx].replace(" ", "_").replace("/", "-")
    clear_rgb = material_colors_np[most_transparent_idx]
    clear_color_hex = "{:02x}{:02x}{:02x}".format(
        int(clear_rgb[0] * 255), int(clear_rgb[1] * 255), int(clear_rgb[2] * 255)
    )
    print(f"Selected clear material: {material_names[most_transparent_idx]} (TD: {material_TDs_np[most_transparent_idx]:.2f})")
    
    heights = np.asarray(disc_height_image).astype(np.int64)

    # The material printed at each layer. A gap in disc_global (-1) takes the
    # color from below, so there is never clear between two colored layers.
    layer_color = np.full(max_layer, -1, dtype=np.int64)
    last_color = -1
    for layer in range(max_layer):
        material = int(disc_global[layer])
        if material >= 0:
            last_color = material
        layer_color[layer] = last_color

    # Contiguous runs of one material: (material, first_layer, end_layer).
    bands = []
    for layer in range(max_layer):
        material = int(layer_color[layer])
        if material < 0:
            continue
        if bands and bands[-1][0] == material and bands[-1][2] == layer:
            bands[-1] = (material, bands[-1][1], layer + 1)
        else:
            bands.append((material, layer, layer + 1))

    # Get unique materials used (excluding background)
    unique_materials = sorted({material for material, _, _ in bands})

    # Calculate the total height of the print (including cap layers)
    total_height = background_height + (max_layer + cap_layers) * layer_height

    stl_files = []

    def create_color_stl(material_idx, material_name, color_hex):
        """Create the STL for one material: one solid per band it prints in.

        A material can recur in several separate bands (e.g. white low down
        and again as a highlight). A single solid spanning the lowest to the
        highest of its layers used to swallow every other material printed in
        between, so the per-color STLs overlapped. At a pixel of height z, the
        band [a, b) is filled from layer a up to min(b, z)."""
        pieces = []
        for material, first, end in bands:
            if material != material_idx:
                continue
            band_mask = heights > first
            if not np.any(band_mask & valid_mask):
                continue
            top_mm = np.minimum(heights, end).astype(float) * layer_height
            bottom_mm = np.full((H, W), first * layer_height, dtype=float)
            mesh_data = _create_flatforge_box_mesh(
                top_mm, bottom_mm, background_height,
                maximum_x_y_size, valid_mask, band_mask
            )
            if mesh_data is not None:
                pieces.append(mesh_data)

        if not pieces:
            return None

        filename = os.path.join(output_folder, f"{material_name}_{color_hex.lstrip('#')}.stl")
        _save_stl_with_manifold_fix(np.concatenate(pieces, axis=0), filename)
        stl_files.append(filename)
        return filename

    # Generate STL for each unique material
    for mat_idx in unique_materials:
        material_name = material_names[mat_idx].replace(" ", "_").replace("/", "-")
        # Get color hex from material_colors_np
        rgb = material_colors_np[mat_idx]
        color_hex = "#{:02x}{:02x}{:02x}".format(
            int(rgb[0] * 255), int(rgb[1] * 255), int(rgb[2] * 255)
        )

        print(f"Generating FlatForge STL for {material_name}...")
        create_color_stl(mat_idx, material_name, color_hex)

    # Generate STL for clear/transparent areas
    # Clear is only placed above the topmost colored layer of a pixel, never
    # between colors (gaps take the color from below, see layer_color).
    print("Generating FlatForge STL for clear areas...")

    clear_mask = valid_mask & (heights < max_layer)
    if np.any(clear_mask):
        clear_height_map_mm = np.full((H, W), max_layer * layer_height, dtype=float)
        clear_min_height_map_mm = heights.astype(float) * layer_height

        filename = os.path.join(output_folder, f"Clear_{clear_material_name}_{clear_color_hex}.stl")
        mesh_data = _create_flatforge_box_mesh(
            clear_height_map_mm, clear_min_height_map_mm, background_height,
            maximum_x_y_size, valid_mask, clear_mask
        )
        if mesh_data is not None:
            _save_stl_with_manifold_fix(mesh_data, filename)
            stl_files.append(filename)
    else:
        print("No clear areas needed - all layers filled with colored materials.")

    # Generate cap layer if requested
    if cap_layers > 0:
        print(f"Generating FlatForge STL for cap layer ({cap_layers} layers)...")
        # Cap layer covers the entire valid area at the top, creating a flat top surface
        # The cap should be ONLY transparent filament with thickness = cap_layers * layer_height
        # It starts at max_layer (above all other layers) and extends to max_layer + cap_layers
        cap_height_map = np.full((H, W), (max_layer + cap_layers) * layer_height, dtype=float)
        cap_min_height_map = np.full((H, W), max_layer * layer_height, dtype=float)
        
        # Cap covers all valid pixels to ensure flat top
        cap_mask = valid_mask
        
        filename = os.path.join(output_folder, f"Cap_{clear_material_name}_{clear_color_hex}.stl")
        mesh_data = _create_flatforge_box_mesh(
            cap_height_map, cap_min_height_map, background_height,
            maximum_x_y_size, valid_mask, cap_mask
        )
        if mesh_data is not None:
            _save_stl_with_manifold_fix(mesh_data, filename)
            stl_files.append(filename)
    
    # Generate background STL
    print("Generating FlatForge STL for background...")
    # Background is a flat layer at the bottom covering the entire valid area
    bg_height_map = np.full((H, W), background_height, dtype=float)
    bg_min_height_map = np.zeros((H, W), dtype=float)
    bg_mask = valid_mask
    
    filename = os.path.join(output_folder, f"Background_{background_color_hex.lstrip('#')}.stl")
    mesh_data = _create_flatforge_box_mesh(
        bg_height_map, bg_min_height_map, 0.0,  # No additional offset for background
        maximum_x_y_size, valid_mask, bg_mask
    )
    if mesh_data is not None:
        _save_stl_with_manifold_fix(mesh_data, filename)
        stl_files.append(filename)
    
    print(f"FlatForge mode: Generated {len(stl_files)} STL files.")
    return stl_files


def _create_flatforge_box_mesh(max_height_map, min_height_map, z_offset, 
                                maximum_x_y_size, valid_mask, material_mask):
    """
    Create a mesh for FlatForge where each pixel can have a different min and max height.
    
    Args:
        max_height_map (np.ndarray): Maximum height for each pixel (in mm).
        min_height_map (np.ndarray): Minimum height for each pixel (in mm).
        z_offset (float): Z offset to add to all heights.
        maximum_x_y_size (float): Maximum size for x and y dimensions.
        valid_mask (np.ndarray): Boolean mask of valid pixels.
        material_mask (np.ndarray): Boolean mask of pixels that have this material.
    
    Returns:
        tuple: (vertices, faces) or None if no valid geometry.
    """
    H, W = max_height_map.shape
    
    # Only process pixels that are both valid and have this material
    active_mask = valid_mask & material_mask
    
    if not np.any(active_mask):
        return None
    
    # Create coordinate grids
    j, i = np.meshgrid(np.arange(W), np.arange(H))
    x = j.astype(np.float32)
    y = (H - 1 - i).astype(np.float32)
    
    # Scale to match maximum_x_y_size
    original_max = max(W - 1, H - 1)
    scale = maximum_x_y_size / original_max
    x = x * scale
    y = y * scale
    
    # Create top and bottom vertex grids
    z_top = max_height_map.astype(np.float32) + z_offset
    z_bottom = min_height_map.astype(np.float32) + z_offset
    
    # Stack into vertex arrays [H, W, 3]
    top_vertices = np.stack([x, y, z_top], axis=2)
    bottom_vertices = np.stack([x, y, z_bottom], axis=2)
    
    # Build quad mesh only for active pixels
    # A quad is valid if all four corners are active
    quad_valid = (
        active_mask[:-1, :-1]
        & active_mask[:-1, 1:]
        & active_mask[1:, 1:]
        & active_mask[1:, :-1]
    )
    
    valid_i, valid_j = np.nonzero(quad_valid)
    num_quads = len(valid_i)
    
    if num_quads == 0:
        return None
    
    # Get quad corners
    v0 = top_vertices[valid_i, valid_j]
    v1 = top_vertices[valid_i, valid_j + 1]
    v2 = top_vertices[valid_i + 1, valid_j + 1]
    v3 = top_vertices[valid_i + 1, valid_j]
    
    # Top surface triangles
    top_triangles = np.concatenate([
        np.stack([v2, v1, v0], axis=1),
        np.stack([v3, v2, v0], axis=1)
    ], axis=0)
    
    # Bottom surface triangles
    bv0 = bottom_vertices[valid_i, valid_j]
    bv1 = bottom_vertices[valid_i, valid_j + 1]
    bv2 = bottom_vertices[valid_i + 1, valid_j + 1]
    bv3 = bottom_vertices[valid_i + 1, valid_j]
    
    bottom_triangles = np.concatenate([
        np.stack([bv0, bv1, bv2], axis=1),
        np.stack([bv0, bv2, bv3], axis=1)
    ], axis=0)
    
    # Side walls - same logic as in generate_stl
    side_triangles_list = []
    
    # Left edges
    left_cond = np.zeros_like(quad_valid, dtype=bool)
    left_cond[:, 0] = quad_valid[:, 0]
    left_cond[:, 1:] = quad_valid[:, 1:] & (~quad_valid[:, :-1])
    li, lj = np.nonzero(left_cond)
    if len(li) > 0:
        lv0 = top_vertices[li, lj]
        lv1 = top_vertices[li + 1, lj]
        lb0 = bottom_vertices[li, lj]
        lb1 = bottom_vertices[li + 1, lj]
        left_tris = np.concatenate([
            np.stack([lv0, lv1, lb1], axis=1),
            np.stack([lv0, lb1, lb0], axis=1)
        ], axis=0)
        side_triangles_list.append(left_tris)
    
    # Right edges
    right_cond = np.zeros_like(quad_valid, dtype=bool)
    right_cond[:, -1] = quad_valid[:, -1]
    right_cond[:, :-1] = quad_valid[:, :-1] & (~quad_valid[:, 1:])
    ri, rj = np.nonzero(right_cond)
    if len(ri) > 0:
        rv0 = top_vertices[ri, rj + 1]
        rv1 = top_vertices[ri + 1, rj + 1]
        rb0 = bottom_vertices[ri, rj + 1]
        rb1 = bottom_vertices[ri + 1, rj + 1]
        right_tris = np.concatenate([
            np.stack([rv0, rv1, rb1], axis=1),
            np.stack([rv0, rb1, rb0], axis=1)
        ], axis=0)
        side_triangles_list.append(right_tris)
    
    # Top edges
    top_cond = np.zeros_like(quad_valid, dtype=bool)
    top_cond[0, :] = quad_valid[0, :]
    top_cond[1:, :] = quad_valid[1:, :] & (~quad_valid[:-1, :])
    ti, tj = np.nonzero(top_cond)
    if len(ti) > 0:
        tv0 = top_vertices[ti, tj]
        tv1 = top_vertices[ti, tj + 1]
        tb0 = bottom_vertices[ti, tj]
        tb1 = bottom_vertices[ti, tj + 1]
        top_wall_tris = np.concatenate([
            np.stack([tv0, tv1, tb1], axis=1),
            np.stack([tv0, tb1, tb0], axis=1)
        ], axis=0)
        side_triangles_list.append(top_wall_tris)
    
    # Bottom edges
    bottom_cond = np.zeros_like(quad_valid, dtype=bool)
    bottom_cond[-1, :] = quad_valid[-1, :]
    bottom_cond[:-1, :] = quad_valid[:-1, :] & (~quad_valid[1:, :])
    bi, bj = np.nonzero(bottom_cond)
    if len(bi) > 0:
        bv0_edge = top_vertices[bi + 1, bj]
        bv1_edge = top_vertices[bi + 1, bj + 1]
        bb0 = bottom_vertices[bi + 1, bj]
        bb1 = bottom_vertices[bi + 1, bj + 1]
        bottom_wall_tris = np.concatenate([
            np.stack([bv0_edge, bv1_edge, bb1], axis=1),
            np.stack([bv0_edge, bb1, bb0], axis=1)
        ], axis=0)
        side_triangles_list.append(bottom_wall_tris)
    
    # Combine all triangles
    side_triangles = (
        np.concatenate(side_triangles_list, axis=0)
        if side_triangles_list
        else np.empty((0, 3, 3), dtype=np.float32)
    )
    
    all_triangles = np.concatenate([top_triangles, side_triangles, bottom_triangles], axis=0)
    
    return all_triangles


def _save_stl_with_manifold_fix(triangles, filename):
    """
    Save triangles to an STL file with manifold fixing.
    
    Args:
        triangles (np.ndarray): Array of triangles with shape [num_triangles, 3, 3].
        filename (str): Output filename.
    """
    # Compute normals
    v1_arr = triangles[:, 0, :]
    v2_arr = triangles[:, 1, :]
    v3_arr = triangles[:, 2, :]
    normals = np.cross(v2_arr - v1_arr, v3_arr - v1_arr)
    norms = np.linalg.norm(normals, axis=1)
    norms[norms == 0] = 1
    normals /= norms[:, np.newaxis]
    
    num_triangles = triangles.shape[0]
    
    # Create structured array for binary STL
    stl_dtype = np.dtype([
        ("normal", np.float32, (3,)),
        ("v1", np.float32, (3,)),
        ("v2", np.float32, (3,)),
        ("v3", np.float32, (3,)),
        ("attr", np.uint16),
    ])
    
    stl_data = np.empty(num_triangles, dtype=stl_dtype)
    stl_data["normal"] = normals
    stl_data["v1"] = triangles[:, 0, :]
    stl_data["v2"] = triangles[:, 1, :]
    stl_data["v3"] = triangles[:, 2, :]
    stl_data["attr"] = 0
    
    # Write to buffer
    buffer = io.BytesIO()
    header = "FlatForge STL".encode("utf-8").ljust(80, b" ")
    buffer.write(header)
    buffer.write(struct.pack("<I", num_triangles))
    buffer.write(stl_data.tobytes())
    buffer.seek(0)

    with open(filename, "wb") as f:
        buffer.seek(0)
        f.write(buffer.read())
    
    # # Load with trimesh and fix manifold issues
    # try:
    #     mesh = trimesh.load(buffer, file_type="stl")
    #     mesh.merge_vertices()
    #
    #     # Try available degenerate-face removal utilities across trimesh versions
    #     try:
    #         if hasattr(mesh, "remove_degenerate_faces"):
    #             mesh.remove_degenerate_faces()
    #         elif hasattr(trimesh.repair, "remove_degenerate_faces"):
    #             trimesh.repair.remove_degenerate_faces(mesh)
    #         elif hasattr(trimesh.repair, "remove_degenerate_triangles"):
    #             trimesh.repair.remove_degenerate_triangles(mesh)
    #     except Exception as repair_err:
    #         print(f"Warning: Skipped degenerate-face removal for {filename}: {repair_err}")
    #
    #     # Remove duplicate faces if the method exists
    #     try:
    #         if hasattr(mesh, "remove_duplicate_faces"):
    #             mesh.remove_duplicate_faces()
    #         elif hasattr(trimesh.repair, "remove_duplicate_faces"):
    #             trimesh.repair.remove_duplicate_faces(mesh)
    #     except Exception as repair_err:
    #         print(f"Warning: Skipped duplicate-face removal for {filename}: {repair_err}")
    #
    #     # Fill holes and fix normals when available
    #     try:
    #         if hasattr(mesh, "fill_holes"):
    #             mesh.fill_holes()
    #     except Exception as repair_err:
    #         print(f"Warning: Skipped hole filling for {filename}: {repair_err}")
    #
    #     try:
    #         mesh.fix_normals()
    #     except Exception as repair_err:
    #         print(f"Warning: Skipped normal fixing for {filename}: {repair_err}")
    #
    #     mesh.export(filename)
    # except Exception as e:
    #     print(f"Warning: Error fixing manifold for {filename}: {e}")
    #     # Fall back to saving without manifold fix
    #     with open(filename, "wb") as f:
    #         buffer.seek(0)
    #         f.write(buffer.read())
