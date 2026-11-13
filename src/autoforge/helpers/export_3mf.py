"""Generate Bambu Studio / OrcaSlicer .3mf project archives."""

import json
import logging
import os
import re
import struct
import xml.etree.ElementTree as ET
import zipfile

logger = logging.getLogger(__name__)


def stl_to_mesh_arrays(stl_path_or_bytes):
    """Parse a binary STL file or bytes into unique vertices and triangle indices."""
    if isinstance(stl_path_or_bytes, (str, os.PathLike)):
        with open(stl_path_or_bytes, "rb") as f:
            data = f.read()
    else:
        data = bytes(stl_path_or_bytes)

    if len(data) < 84:
        raise ValueError("STL data is too short to be a valid binary STL")

    num_triangles = struct.unpack("<I", data[80:84])[0]
    expected_len = 84 + num_triangles * 50
    if len(data) < expected_len:
        raise ValueError(
            f"Truncated binary STL: expected {expected_len} bytes for {num_triangles} triangles, got {len(data)}"
        )

    vertices = []
    triangles = []
    vertex_map = {}

    offset = 84
    for _ in range(num_triangles):
        # 50-byte triangle record: 12B facet normal, 3x12B vertex coords, 2B attribute count.
        offset += 12
        tri_indices = []
        for _ in range(3):
            vx, vy, vz = struct.unpack("<fff", data[offset : offset + 12])
            offset += 12
            v_key = (round(vx, 4), round(vy, 4), round(vz, 4))
            idx = vertex_map.get(v_key)
            if idx is None:
                idx = len(vertices)
                vertex_map[v_key] = idx
                vertices.append(v_key)
            tri_indices.append(idx)
        offset += 2
        triangles.append(tri_indices)

    return vertices, triangles


def build_3dmodel_xml(vertices, triangles, model_name="AutoForge_Model"):
    """Generate ``3D/3dmodel.model`` XML centered on the BBL H3C build plate."""
    root = ET.Element("model", {
        "unit": "millimeter",
        "xml:lang": "en-US",
        "xmlns": "http://schemas.microsoft.com/3dmanufacturing/core/2015/02",
    })
    app_meta = ET.SubElement(root, "metadata", {"name": "Application"})
    app_meta.text = "BambuStudio-02.09.01.70"
    title_meta = ET.SubElement(root, "metadata", {"name": "Title"})
    title_meta.text = model_name
    resources = ET.SubElement(root, "resources")
    obj = ET.SubElement(resources, "object", {"id": "1", "type": "model", "name": model_name})
    mesh = ET.SubElement(obj, "mesh")

    if vertices:
        min_x, max_x = min(v[0] for v in vertices), max(v[0] for v in vertices)
        min_y, max_y = min(v[1] for v in vertices), max(v[1] for v in vertices)
        min_z = min(v[2] for v in vertices)
        center_x = (min_x + max_x) / 2.0
        center_y = (min_y + max_y) / 2.0
        # Center XY on the 330x320 mm BBL H3C build plate and rest base at Z=0.
        shifted_vertices = [
            (vx - center_x + 165.0, vy - center_y + 160.0, vz - min_z)
            for vx, vy, vz in vertices
        ]
    else:
        shifted_vertices = []

    verts_elem = ET.SubElement(mesh, "vertices")
    for vx, vy, vz in shifted_vertices:
        ET.SubElement(verts_elem, "vertex", {
            "x": f"{vx:.4f}",
            "y": f"{vy:.4f}",
            "z": f"{vz:.4f}",
        })

    tris_elem = ET.SubElement(mesh, "triangles")
    for v1, v2, v3 in triangles:
        ET.SubElement(tris_elem, "triangle", {
            "v1": str(v1),
            "v2": str(v2),
            "v3": str(v3),
        })

    build = ET.SubElement(root, "build")
    ET.SubElement(build, "item", {"objectid": "1"})

    return ET.tostring(root, encoding="utf-8", xml_declaration=True)


def build_custom_gcodes_xml(swaps, base_height, layer_height):
    """Build ``Metadata/custom_gcode_per_layer.xml`` for layer tool-change pauses."""
    root = ET.Element("custom_gcodes_per_layer")
    plate = ET.SubElement(root, "plate")
    ET.SubElement(plate, "plate_info", {"id": "1"})

    for swap in swaps:
        top_z = (
            float(swap["top_z"])
            if swap.get("top_z") is not None
            else base_height + (swap["layer"] - 1) * layer_height
        )
        ET.SubElement(plate, "layer", {
            "top_z": f"{top_z:.4f}",
            "type": "2",
            "extruder": str(swap["extruder"]),
            "color": swap["color"].upper(),
            "extra": "",
            "gcode": "tool_change",
        })

    ET.SubElement(plate, "mode", {"value": "MultiAsSingle"})
    return ET.tostring(root, encoding="utf-8", xml_declaration=True)


def build_model_settings_xml(model_name="AutoForge_Model"):
    """Build ``Metadata/model_settings.config`` XML."""
    config = ET.Element("config")
    obj = ET.SubElement(config, "object", {"id": "1"})
    ET.SubElement(obj, "metadata", {"key": "name", "value": model_name})
    ET.SubElement(obj, "metadata", {"key": "extruder", "value": "1"})

    part = ET.SubElement(obj, "part", {"id": "1", "subtype": "normal_part"})
    ET.SubElement(part, "metadata", {"key": "name", "value": model_name})
    ET.SubElement(part, "metadata", {"key": "matrix", "value": "1 0 0 0 0 1 0 0 0 0 1 0 0 0 0 1"})

    plate = ET.SubElement(config, "plate")
    ET.SubElement(plate, "metadata", {"key": "plater_id", "value": "1"})
    ET.SubElement(plate, "metadata", {"key": "filament_map_mode", "value": "Auto For Match"})

    model_instance = ET.SubElement(plate, "model_instance")
    ET.SubElement(model_instance, "metadata", {"key": "object_id", "value": "1"})
    ET.SubElement(model_instance, "metadata", {"key": "instance_id", "value": "0"})

    assemble = ET.SubElement(config, "assemble")
    ET.SubElement(assemble, "assemble_item", {
        "object_id": "1",
        "instance_id": "0",
        "transform": "1 0 0 0 1 0 0 0 1 0 0 0",
        "offset": "0 0 0",
    })

    return ET.tostring(config, encoding="utf-8", xml_declaration=True)


_STOCK_SINGLE_FILAMENT_DEFAULTS = {
    "activate_air_filtration": "0",
    "additional_cooling_fan_speed": "60",
    "additional_fan_full_speed_layer": "0",
    "chamber_temperatures": "0",
    "circle_compensation_speed": "200",
    "close_additional_fan_first_x_layers": "1",
    "close_fan_the_first_x_layers": "1",
    "close_optional_fan_first_x_layers": "1",
    "complete_print_exhaust_fan_speed": "70",
    "cool_plate_temp": "35",
    "cool_plate_temp_initial_layer": "35",
    "cooling_perimeter_transition_distance": "10",
    "cooling_slowdown_logic": "uniform_cooling",
    "counter_coef_1": "0",
    "counter_coef_2": "0.008",
    "counter_coef_3": "-0.041",
    "counter_limit_max": "0.033",
    "counter_limit_min": "-0.035",
    "default_filament_colour": "",
    "diameter_limit": "50",
    "during_print_exhaust_fan_speed": "70",
    "enable_overhang_bridge_fan": "1",
    "enable_pressure_advance": "0",
    "eng_plate_temp": "55",
    "eng_plate_temp_initial_layer": "55",
    "fan_cooling_layer_time": "100",
    "fan_max_speed": "100",
    "filament_adhesiveness_category": "100",
    "filament_change_length": "0",
    "filament_change_length_nc": "0",
    "filament_colour_type": "1",
    "filament_cost": "15.99",
    "filament_dev_chamber_drying_bed_temperature": "70",
    "filament_dev_chamber_drying_time": "12",
    "filament_dev_drying_cooling_temperature": "45",
    "filament_diameter": "1.75",
    "filament_diameter_map": "0.4",
    "filament_end_gcode": "; filament end gcode \n",
    "filament_extruder_compatibility": "0",
    "filament_idle_travel_cooling_enabled": "0",
    "filament_is_support": "0",
    "filament_metal_stickiness": "None",
    "filament_minimal_purge_on_wipe_tower": "15",
    "filament_pre_heating_enabled": "1",
    "filament_prime_tower_speed_factor_curve": "0.5,1",
    "filament_prime_volume": "30",
    "filament_prime_volume_nc": "60",
    "filament_printable": "3",
    "filament_scarf_gap": "0%",
    "filament_scarf_length": "10",
    "filament_scarf_seam_type": "none",
    "filament_shrink": "100%",
    "filament_soluble": "0",
    "filament_start_gcode": "; filament start gcode\n",
    "filament_tower_interface_pre_extrusion_dist": "10",
    "filament_tower_interface_pre_extrusion_length": "0",
    "filament_tower_interface_print_temp": "-1",
    "filament_tower_interface_purge_volume": "20",
    "filament_tower_ironing_area": "4",
    "filament_type": "PLA",
    "filament_velocity_adaptation_factor": "1",
    "filament_vendor": "Bambu Lab",
    "filament_volume_map": "0",
    "fins_extrude_safe_temp": "57",
    "first_x_layer_fan_speed": "0",
    "first_x_layer_optional_fan_speed": "0",
    "first_x_layer_part_fan_speed": "0",
    "full_fan_speed_layer": "0",
    "hole_coef_1": "0",
    "hole_coef_2": "-0.008",
    "hole_coef_3": "0.23415",
    "hole_limit_max": "0.22",
    "hole_limit_min": "0.088",
    "hot_plate_temp": "55",
    "hot_plate_temp_initial_layer": "55",
    "ironing_fan_speed": "-1",
    "min_temperature_when_ec": "0",
    "no_slow_down_for_cooling_on_outwalls": "0",
    "nozzle_temperature_range_high": "240",
    "nozzle_temperature_range_low": "190",
    "optional_cooling_fan_speed": "80",
    "optional_fan_full_speed_layer": "0",
    "overhang_fan_speed": "100",
    "overhang_fan_threshold": "50%",
    "overhang_threshold_participating_cooling": "95%",
    "pre_start_fan_time": "2",
    "pressure_advance": "0.02",
    "reduce_fan_stop_start_freq": "1",
    "required_nozzle_HRC": "3",
    "slow_down_for_layer_cooling": "1",
    "slow_down_layer_time": "4",
    "supertack_plate_temp": "45",
    "supertack_plate_temp_initial_layer": "45",
    "temperature_vitrification": "45",
    "textured_plate_temp": "55",
    "textured_plate_temp_initial_layer": "55",
}

# BBL H3C has two extruder variants per filament slot ("Direct Drive Standard"
# and "Direct Drive High Flow"). Bambu Studio's PresetBundle::load_config_file_config
# slices 2 elements per filament for these 45 s_filament_extruder_variant_keys
# when extruder_variant_list, nozzle_diameter, filament_self_index, and
# filament_extruder_variant are present.
_STOCK_DUAL_VARIANT_FILAMENT_DEFAULTS = {
    "filament_adaptive_volumetric_speed": ["0", "0"],
    "filament_bridge_speed": ["25", "25"],
    "filament_cooling_before_tower": ["0", "0"],
    "filament_deretraction_speed": ["nil", "nil"],
    "filament_enable_overhang_speed": ["1", "1"],
    "filament_extruder_variant": ["Direct Drive Standard", "Direct Drive High Flow"],
    "filament_flow_ratio": ["0.98", "0.98"],
    "filament_flush_temp": ["0", "0"],
    "filament_flush_temp_fast": ["0", "0"],
    "filament_flush_volumetric_speed": ["0", "0"],
    "filament_long_retractions_when_cut": ["nil", "nil"],
    "filament_max_volumetric_speed": ["25", "40"],
    "filament_overhang_1_4_speed": ["0", "0"],
    "filament_overhang_2_4_speed": ["50", "50"],
    "filament_overhang_3_4_speed": ["30", "30"],
    "filament_overhang_4_4_speed": ["10", "10"],
    "filament_overhang_totally_speed": ["10", "10"],
    "filament_pre_cooling_temperature": ["0", "0"],
    "filament_pre_cooling_temperature_nc": ["0", "0"],
    "filament_preheat_temperature_delta": ["10", "10"],
    "filament_ramming_travel_time": ["0", "0"],
    "filament_ramming_travel_time_nc": ["0", "0"],
    "filament_ramming_volumetric_speed": ["-1", "-1"],
    "filament_ramming_volumetric_speed_nc": ["-1", "-1"],
    "filament_retract_before_wipe": ["nil", "nil"],
    "filament_retract_length_nc": ["14", "14"],
    "filament_retract_length_toolchange": ["nil", "nil"],
    "filament_retract_lift_above": ["0", "0"],
    "filament_retract_lift_below": ["0", "0"],
    "filament_retract_restart_extra": ["nil", "nil"],
    "filament_retract_restart_extra_toolchange": ["nil", "nil"],
    "filament_retract_when_changing_layer": ["nil", "nil"],
    "filament_retraction_distances_when_cut": ["nil", "nil"],
    "filament_retraction_length": ["0.6", "0.6"],
    "filament_retraction_minimum_travel": ["nil", "nil"],
    "filament_retraction_speed": ["nil", "nil"],
    "filament_wipe": ["1", "1"],
    "filament_wipe_distance": ["1", "1"],
    "filament_z_hop": ["nil", "nil"],
    "filament_z_hop_types": ["Spiral Lift", "Spiral Lift"],
    "long_retractions_when_ec": ["0", "0"],
    "nozzle_temperature": ["220", "220"],
    "nozzle_temperature_initial_layer": ["220", "220"],
    "override_process_overhang_speed": ["0", "0"],
    "retraction_distances_when_ec": ["10", "10"],
    "slow_down_min_speed": ["20", "20"],
    "volumetric_speed_coefficients": ["0 0 0 0 0 0", "0 0 0 0 0 0"],
}


def _is_matte_filament(fil):
    """Return True when any filament metadata field indicates a Matte filament."""
    if not isinstance(fil, dict):
        return False
    combined = " ".join(
        str(fil.get(k) or "")
        for k in ("brand", "name", "short_name", "filament_type", "filamentType")
    ).lower()
    return "matte" in combined


def build_project_settings_config(unique_filaments, base_height, layer_height):
    """Build ``Metadata/project_settings.config`` JSON for Bambu Studio."""
    count = len(unique_filaments)
    is_matte = [_is_matte_filament(f) for f in unique_filaments]
    mapped_preset_ids = [
        "Bambu PLA Matte @BBL H3C" if m else "Bambu PLA Basic @BBL H3C"
        for m in is_matte
    ]

    settings = {
        "version": "02.09.01.70",
        "from": "project",
        "name": "project_settings",
        "printer_model": "Bambu Lab H3C",
        "printer_settings_id": "Bambu Lab H3C 0.4 nozzle",
        "print_compatible_printers": ["Bambu Lab H3C 0.4 nozzle"],
        "printer_variant": "0.4",
        "printer_structure": "corexy",
        "printer_technology": "FFF",
        "printable_area": ["0x0", "330x0", "330x320", "0x320"],
        "printable_height": "325",
        "extruder_variant_list": [
            "Direct Drive Standard,Direct Drive High Flow,Direct Drive TPU285"
        ],
        "nozzle_diameter": ["0.4"],
        "print_settings_id": "",
        "default_print_profile": "0.20mm Standard @BBL H3C",
        "default_filament_profile": ["Bambu PLA Basic @BBL H3C"],
        "layer_height": f"{layer_height:.4f}",
        "initial_layer_print_height": f"{base_height:.4f}",
        "sparse_infill_density": "100%",
        "skeleton_infill_density": "100%",
        "skin_infill_density": "100%",
        # Bambu Studio expects the "zig-zag" enum value for rectilinear infill.
        "sparse_infill_pattern": "zig-zag",
        "internal_solid_infill_pattern": "zig-zag",
        "different_settings_to_system": [
            "skeleton_infill_density;skin_infill_density;sparse_infill_pattern",
            "",
            "",
            "",
            "",
            "",
        ]
        if count <= 4
        else ["skeleton_infill_density;skin_infill_density;sparse_infill_pattern"] + [""] * (1 + count),
        "bottom_surface_pattern": "monotonic",
        "top_surface_pattern": "monotonicline",
        "single_extruder_multi_material": "1",
        "filament_colour": [f["color"].upper() for f in unique_filaments],
        "filament_multi_colour": [f["color"].upper() for f in unique_filaments],
        "filament_settings_id": mapped_preset_ids,
        "filament_ids": ["GFA01" if m else "GFA00" for m in is_matte],
        "filament_density": ["1.32" if m else "1.26" for m in is_matte],
        "fan_min_speed": ["100" if m else "80" for m in is_matte],
        "filament_scarf_height": ["5%" if m else "10%" for m in is_matte],
        "impact_strength_z": ["6.6" if m else "13.8" for m in is_matte],
        "filament_map": [str(i + 1) for i in range(count)],
        "filament_self_index": [str(i + 1) for i in range(count) for _ in range(2)],
        "filament_notes": "",
        "filament_dev_ams_drying_ams_limitations": ["1", "0", "2", "3"] * count,
        "filament_dev_ams_drying_heat_distortion_temperature": ["45", "45", "60", "45"] * count,
        "filament_dev_drying_softening_temperature": ["50", "50", "65", "50"] * count,
        "filament_dev_ams_drying_temperature": ["45", "45", "45", "45", "55", "55", "45", "45"] * count,
        "filament_dev_ams_drying_time": ["12", "12", "12", "12", "12", "12", "12", "12"] * count,
        "wipe": ["1"] * (2 * count),
    }

    for key, val in _STOCK_SINGLE_FILAMENT_DEFAULTS.items():
        settings[key] = [val] * count

    for key, pair in _STOCK_DUAL_VARIANT_FILAMENT_DEFAULTS.items():
        settings[key] = list(pair) * count

    return json.dumps(settings, indent=4).encode("utf-8")


def _normalize_filament(fil, default_color="#000000"):
    """Normalize a filament dict so Bambu Studio config fields are always populated."""
    fil = fil if isinstance(fil, dict) else {}
    color = str(fil.get("color") or default_color or "#000000").strip()
    if not color.startswith("#"):
        color = "#" + color
    return {
        "color": color,
        "filament_type": str(fil.get("filament_type") or fil.get("filamentType") or "PLA"),
        "brand": str(fil.get("brand") or "Generic"),
        "name": str(fil.get("name") or fil.get("short_name") or ""),
        "short_name": str(fil.get("short_name") or ""),
    }


def _fallback_single_filament(settings, active_filaments):
    """Pick a valid single filament when no color sliders are enabled."""
    base_uuid = str(settings.get("base_filament_uuid") or "")
    if base_uuid and base_uuid in active_filaments:
        return _normalize_filament(active_filaments[base_uuid])
    bg_color = str(settings.get("background_color") or "").strip().lower()
    if bg_color:
        for fil in active_filaments.values():
            if str(fil.get("color") or "").strip().lower() == bg_color:
                return _normalize_filament(fil)
    if active_filaments:
        first_fil = next(iter(active_filaments.values()))
        return _normalize_filament(first_fil, default_color=bg_color or "#000000")
    return _normalize_filament({"color": bg_color or "#000000"})


def build_project_dict_from_stack(
    disc_global,
    max_layer,
    layer_height,
    background_height,
    background_color,
    materials,
    base_filament_uuid="",
):
    """Build an AutoForge project dictionary from a discrete material stack."""
    active_filaments = []
    uuids = []
    for i, mat in enumerate(materials or []):
        m = dict(mat) if isinstance(mat, dict) else {}
        uid = str(m.get("uuid") or f"mat_{i}")
        m["uuid"] = uid
        m.setdefault("color", "#808080")
        m.setdefault("td", float(m.get("td") or 5.0))
        m.setdefault("brand", m.get("brand") or "Generic")
        m.setdefault("filament_type", m.get("filament_type") or "PLA")
        active_filaments.append(m)
        uuids.append(uid)

    def _slider_entry(layer_num, mat_idx):
        uid = uuids[mat_idx] if 0 <= mat_idx < len(uuids) else ""
        td = float(active_filaments[mat_idx]["td"]) if 0 <= mat_idx < len(active_filaments) else 5.0
        return {
            "td": td,
            "layer": layer_num,
            "depth_mm": round(layer_num * float(layer_height), 2),
            "filament_uuid": uid,
            "enabled": True,
        }

    sliders = []
    if max_layer > 0 and len(disc_global) > 0:
        n_stack = min(len(disc_global), int(max_layer))
        current_mat = None
        for layer in range(1, n_stack + 1):
            mat_idx = int(disc_global[layer - 1])
            if current_mat is not None and mat_idx != current_mat:
                sliders.append(_slider_entry(layer - 1, current_mat))
            current_mat = mat_idx
        if current_mat is not None:
            sliders.append(_slider_entry(n_stack, current_mat))

    return {
        "settings": {
            "layer_height": float(layer_height),
            "background_height": float(background_height),
            "background_color": str(background_color or "#000000"),
            "base_filament_uuid": str(base_filament_uuid or ""),
        },
        "colorSliders": sliders,
        "activeFilaments": active_filaments,
    }


def parse_swap_instructions(text):
    """Parse base filament, base layer height, and layer swaps from ``swap_instructions.txt``."""
    start_match = re.search(r"Start with (.*?),\s*with a layer height of ([\d\.]+)mm", text)
    base_filament = start_match.group(1).strip() if start_match else None
    base_height = float(start_match.group(2)) if start_match else 0.24

    swaps = []
    for match in re.finditer(r"At layer #(\d+)\s*\(([\d\.]+)mm\)\s*swap to (.*?)(?:\n|$)", text):
        swaps.append({
            "layer": int(match.group(1)),
            "top_z": float(match.group(2)),
            "filament_name": match.group(3).strip(),
        })
    return base_filament, base_height, swaps


def _read_swap_instructions_text(swap_instructions_text):
    """Accept raw swap instructions text, bytes, or a path to ``swap_instructions.txt``."""
    if swap_instructions_text is None:
        return None
    if isinstance(swap_instructions_text, os.PathLike):
        with open(swap_instructions_text, "r", encoding="utf-8") as f:
            return f.read()
    if isinstance(swap_instructions_text, bytes):
        return swap_instructions_text.decode("utf-8", errors="replace")
    text = str(swap_instructions_text)
    if "\n" not in text and "\r" not in text and len(text) < 4096 and os.path.isfile(text):
        with open(text, "r", encoding="utf-8") as f:
            return f.read()
    return text


def _norm_filament_label(label):
    collapsed = re.sub(r"\s+", " ", str(label or "")).strip().lower()
    return re.sub(r"\s*-\s*", " - ", collapsed)


def _match_filament_in_active(target_name, active_list):
    """Return ``(index, filament_dict)`` from ``active_list`` matching ``target_name``, or ``(None, None)``."""
    norm_target = _norm_filament_label(target_name)
    if not norm_target:
        return None, None

    for idx, fil in enumerate(active_list):
        if not isinstance(fil, dict):
            continue
        brand = str(fil.get("brand") or "").strip()
        name = str(fil.get("name") or "").strip()
        short_name = str(fil.get("short_name") or "").strip()
        candidates = []
        for part in (short_name, name):
            if part:
                if brand and not _norm_filament_label(part).startswith(f"{_norm_filament_label(brand)} - "):
                    candidates.append(f"{brand} - {part}")
                candidates.append(part)
        if any(_norm_filament_label(cand) == norm_target for cand in candidates):
            return idx, fil
    return None, None


def _preserve_matte_hint(fil, label):
    """Retain a Matte hint from a swap instruction label when normalized fields omit it."""
    if label and "matte" in label.lower() and not _is_matte_filament(fil):
        fil["name"] = label


def _build_swaps_from_instructions(swap_text, active_list, active_filaments, settings, fallback_base_height):
    """Resolve ordered filaments and Z-height swaps from ``swap_instructions.txt``."""
    base_filament, parsed_base_height, parsed_swaps = parse_swap_instructions(swap_text)
    if base_filament is None and not parsed_swaps:
        return None

    base_height = parsed_base_height if base_filament is not None else fallback_base_height
    unique_filaments = []
    seen_to_extruder = {}

    matched_idx, matched_fil = (None, None)
    if base_filament and _norm_filament_label(base_filament) != "your background color":
        matched_idx, matched_fil = _match_filament_in_active(base_filament, active_list)

    if matched_fil is not None:
        base_fil = _normalize_filament(matched_fil, default_color=settings.get("background_color") or "#000000")
        _preserve_matte_hint(base_fil, base_filament)
        unique_filaments.append(base_fil)
        seen_to_extruder[("obj", matched_idx)] = 1
    else:
        base_fil, fb_idx = _fallback_single_filament_with_index(settings, active_list, active_filaments)
        if base_filament and not base_fil.get("name"):
            base_fil["name"] = base_filament
        else:
            _preserve_matte_hint(base_fil, base_filament)
        unique_filaments.append(base_fil)
        if fb_idx is not None:
            seen_to_extruder[("obj", fb_idx)] = 1

    if base_filament:
        seen_to_extruder[("name", _norm_filament_label(base_filament))] = 1

    swaps = []
    for item in parsed_swaps:
        fname = item["filament_name"]
        m_idx, m_fil = _match_filament_in_active(fname, active_list)
        key = ("obj", m_idx) if m_idx is not None else ("name", _norm_filament_label(fname))
        if key not in seen_to_extruder:
            fil = _normalize_filament(
                m_fil if m_fil is not None else {"name": fname},
                default_color="#FFFFFF",
            )
            _preserve_matte_hint(fil, fname)
            unique_filaments.append(fil)
            seen_to_extruder[key] = len(unique_filaments)
        ext_id = seen_to_extruder[key]
        swaps.append({
            "layer": item["layer"],
            "top_z": item["top_z"],
            "extruder": ext_id,
            "color": unique_filaments[ext_id - 1]["color"],
        })

    return unique_filaments, swaps, base_height


def _fallback_single_filament_with_index(settings, active_list, active_filaments):
    """Pick the single base filament plus its index in ``active_list`` when available."""
    base_uuid = str(settings.get("base_filament_uuid") or "")
    if base_uuid:
        for idx, fil in enumerate(active_list):
            if isinstance(fil, dict) and str(fil.get("uuid") or "") == base_uuid:
                return _normalize_filament(fil), idx
        if base_uuid in active_filaments:
            return _normalize_filament(active_filaments[base_uuid]), None
    bg_color = str(settings.get("background_color") or "").strip().lower()
    if bg_color:
        for idx, fil in enumerate(active_list):
            if isinstance(fil, dict) and str(fil.get("color") or "").strip().lower() == bg_color:
                return _normalize_filament(fil), idx
    return _fallback_single_filament(settings, active_filaments), None


def export_bambu_3mf(stl_data, autoforge_project_json, output_path_or_buffer, swap_instructions_text=None):
    """Combine STL geometry and AutoForge metadata into a Bambu Studio .3mf archive."""
    if autoforge_project_json is None:
        project_data = {}
    elif isinstance(autoforge_project_json, (str, bytes)):
        project_data = json.loads(autoforge_project_json)
    else:
        project_data = autoforge_project_json

    settings = project_data.get("settings") or {}
    layer_height = float(settings.get("layer_height") or 0.04)
    bg_raw = settings.get("background_height")
    base_height = float(0.24 if bg_raw is None else bg_raw)

    active_list = [f for f in (project_data.get("activeFilaments") or []) if isinstance(f, dict)]
    active_filaments = {
        str(f["uuid"]): f
        for f in active_list
        if f.get("uuid")
    }

    resolved_swap_text = _read_swap_instructions_text(swap_instructions_text)
    parsed_result = (
        _build_swaps_from_instructions(
            resolved_swap_text,
            active_list,
            active_filaments,
            settings,
            base_height,
        )
        if resolved_swap_text
        else None
    )

    if parsed_result is not None:
        unique_filaments, swaps, base_height = parsed_result
    else:
        raw_sliders = project_data.get("colorSliders") or []
        enabled_sliders = [
            s for s in raw_sliders if isinstance(s, dict) and s.get("enabled", True)
        ]
        enabled_sliders.sort(key=lambda s: int(s.get("layer", 0)))

        unique_filaments = []
        uuid_to_extruder = {}
        swaps = []

        if enabled_sliders:
            first_slider = enabled_sliders[0]
            first_uuid = str(first_slider.get("filament_uuid") or "")
            first_fil = _normalize_filament(
                active_filaments.get(first_uuid),
                default_color=first_slider.get("color") or settings.get("background_color") or "#000000",
            )
            unique_filaments.append(first_fil)
            uuid_to_extruder[first_uuid] = 1
            current_extruder = 1

            for slider in enabled_sliders[1:]:
                fuuid = str(slider.get("filament_uuid") or "")
                if fuuid not in uuid_to_extruder:
                    fil = _normalize_filament(
                        active_filaments.get(fuuid),
                        default_color=slider.get("color") or "#FFFFFF",
                    )
                    unique_filaments.append(fil)
                    uuid_to_extruder[fuuid] = len(unique_filaments)

                ext_id = uuid_to_extruder[fuuid]
                if ext_id != current_extruder:
                    swaps.append({
                        "layer": int(slider.get("layer", 1)),
                        "extruder": ext_id,
                        "color": unique_filaments[ext_id - 1]["color"],
                    })
                    current_extruder = ext_id
        else:
            unique_filaments.append(_fallback_single_filament(settings, active_filaments))

    vertices, triangles = stl_to_mesh_arrays(stl_data)

    content_types_xml = (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">\n'
        '  <Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>\n'
        '  <Default Extension="model" ContentType="application/vnd.ms-package.3dmanufacturing-3dmodelxml"/>\n'
        '  <Default Extension="config" ContentType="text/plain"/>\n'
        '  <Default Extension="xml" ContentType="application/xml"/>\n'
        '  <Default Extension="json" ContentType="application/json"/>\n'
        '</Types>'
    ).encode("utf-8")

    rels_xml = (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">\n'
        '  <Relationship Target="/3D/3dmodel.model" Id="rel0" '
        'Type="http://schemas.microsoft.com/3dmanufacturing/2013/01/3dmodel"/>\n'
        '</Relationships>'
    ).encode("utf-8")

    close_buffer = False
    if isinstance(output_path_or_buffer, (str, os.PathLike)):
        buf = open(output_path_or_buffer, "wb")
        close_buffer = True
    else:
        buf = output_path_or_buffer

    try:
        with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            zf.writestr("[Content_Types].xml", content_types_xml)
            zf.writestr("_rels/.rels", rels_xml)
            zf.writestr("3D/3dmodel.model", build_3dmodel_xml(vertices, triangles))
            zf.writestr("Metadata/model_settings.config", build_model_settings_xml())
            zf.writestr("Metadata/project_settings.config", build_project_settings_config(unique_filaments, base_height, layer_height))
            zf.writestr("Metadata/custom_gcode_per_layer.xml", build_custom_gcodes_xml(swaps, base_height, layer_height))
            zf.writestr(
                "Metadata/filament_sequence.json",
                b'{"plate_1":{"nozzle_sequence":[],"optimal_assignment":[],"sequence":[]}}',
            )
            zf.writestr(
                "Metadata/slice_info.config",
                b'<?xml version="1.0" encoding="UTF-8"?>\n<config><header><header_item key="X-BBL-Client-Type" value="slicer"/><header_item key="X-BBL-Client-Version" value="02.09.01.70"/></header></config>',
            )
    finally:
        if close_buffer:
            buf.close()

    return output_path_or_buffer


def export_solution_3mf(
    stl_path,
    out_3mf_path,
    disc_global,
    disc_height_image,
    layer_height,
    background_height,
    background_color,
    materials,
    background_material_index=None,
    swap_instructions_text=None,
):
    """Best-effort export of ``final_model_bambu.3mf`` from an optimization solution.

    Logs a warning and removes any partial output if generation fails so callers
    never fail their primary STL/TXT export. Returns ``out_3mf_path`` on
    success or ``None`` on failure.
    """
    try:
        import numpy as np

        dg = np.asarray(disc_global).reshape(-1).astype(int)
        max_layer = int(np.max(np.asarray(disc_height_image)))
        base_uuid = ""
        if (
            background_material_index is not None
            and 0 <= int(background_material_index) < len(materials)
        ):
            mat = materials[int(background_material_index)]
            base_uuid = str((mat.get("uuid") if isinstance(mat, dict) else "") or f"mat_{int(background_material_index)}")

        project_data = build_project_dict_from_stack(
            dg,
            max_layer,
            layer_height,
            background_height,
            background_color,
            materials,
            base_filament_uuid=base_uuid,
        )
        export_bambu_3mf(
            stl_path,
            project_data,
            out_3mf_path,
            swap_instructions_text=swap_instructions_text,
        )
        return out_3mf_path
    except Exception as exc:
        logger.warning("Failed to export Bambu .3mf (%s): %s", out_3mf_path, exc)
        if isinstance(out_3mf_path, (str, os.PathLike)) and os.path.exists(out_3mf_path):
            try:
                os.remove(out_3mf_path)
            except OSError:
                pass
        return None
