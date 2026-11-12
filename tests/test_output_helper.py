"""Tests for autoforge.Helper.OutputHelper.

Consolidated from the former test_output_helper.py, test_output_helper_extended.py
and the output half of test_filament_and_output.py.
"""

import json
import os
import types

import numpy as np
import pytest

pandas = pytest.importorskip("pandas")
pytest.importorskip("trimesh")

from autoforge.Helper.OutputHelper import (
    extract_filament_swaps,
    generate_project_file,
    generate_stl,
    generate_swap_instructions,
)


def _args(tmp_path):
    csv_path = tmp_path / "materials.csv"
    pandas.DataFrame(
        {
            "Brand": ["B0", "B1", "B2"],
            "Name": ["N0", "N1", "N2"],
            "Transmissivity": [1.0, 2.0, 3.0],
            "Color": ["#000000", "#111111", "#222222"],
        }
    ).to_csv(csv_path, index=False)
    return types.SimpleNamespace(
        csv_file=str(csv_path),
        json_file="",
        layer_height=0.04,
        background_height=0.24,
        max_layers=10,
        background_color="#000000",
    )


# --------------------------------------------------------------------------
# extract_filament_swaps
# --------------------------------------------------------------------------


def test_extract_filament_swaps_2d_height_map():
    disc_global = np.array([0, 0, 1, 1, 2, 2])
    disc_height = np.array([[0, 1], [1, 2]])  # L = 2
    indices, sliders = extract_filament_swaps(disc_global, disc_height, background_layers=6)
    assert indices[0] == 0
    assert indices[-1] == indices[-2]
    assert sliders[0] == 1
    assert sliders == sorted(sliders)
    assert sliders[-1] == sliders[-2] + 1


def test_extract_filament_swaps_1d_height_map_records_each_change():
    disc_global = np.array([0, 0, 1, 1, 2, 2])
    disc_height = np.array([0, 1, 2, 3, 4, 5])  # L = 5
    indices, sliders = extract_filament_swaps(disc_global, disc_height, background_layers=2)
    # materials seen while walking layers 0..4: 0,0,1,1,2  -> 0,1,2 then trailing repeat
    assert indices[0] == 0
    assert 1 in indices and 2 in indices
    assert len(indices) == len(sliders)
    assert sliders == sorted(sliders)


def test_extract_filament_swaps_empty_when_no_height():
    indices, sliders = extract_filament_swaps(
        np.array([0, 1, 2]), np.zeros((3, 3), dtype=int), background_layers=1
    )
    assert indices == [] and sliders == []


# --------------------------------------------------------------------------
# generate_swap_instructions
# --------------------------------------------------------------------------


def test_generate_swap_instructions_mentions_swaps_and_closing_line():
    disc_global = np.array([0, 0, 1, 1, 2, 2])
    disc_height = np.array([[0, 1], [1, 2]])
    names = ["A - Mat1", "B - Mat2", "C - Mat3"]
    instr = generate_swap_instructions(
        disc_global, disc_height, 0.04, 6, 0.24, names
    )
    assert any("swap" in line.lower() for line in instr)
    assert instr[-1].startswith("For the rest")


# --------------------------------------------------------------------------
# generate_stl
# --------------------------------------------------------------------------


def test_generate_stl_writes_a_binary_mesh(tmp_path):
    height_map = np.random.rand(5, 7).astype(np.float32)
    out_path = tmp_path / "out.stl"
    generate_stl(height_map, str(out_path), background_height=0.2, maximum_x_y_size=50)
    assert out_path.exists()
    # binary STL: 80-byte header + uint32 triangle count + 50 bytes/triangle
    assert out_path.stat().st_size > 84


def test_generate_stl_flat_region_still_valid(tmp_path):
    hm = np.zeros((4, 4), dtype=float)
    hm[1:3, 1:3] = 1.0
    out_path = tmp_path / "flat.stl"
    generate_stl(hm, str(out_path), background_height=0.2, maximum_x_y_size=10.0)
    assert os.path.getsize(out_path) > 100


def test_generate_stl_alpha_mask_shrinks_the_mesh(tmp_path):
    hm = np.random.rand(16, 16).astype(np.float32)
    full = tmp_path / "full.stl"
    masked = tmp_path / "masked.stl"
    generate_stl(hm, str(full), background_height=0.2, maximum_x_y_size=40.0)

    alpha = np.zeros((16, 16), dtype=np.uint8)
    alpha[4:12, 4:12] = 255  # keep only an 8x8 core
    generate_stl(hm, str(masked), background_height=0.2, maximum_x_y_size=40.0,
                 alpha_mask=alpha)

    assert masked.exists()
    # fewer valid quads -> fewer triangles -> smaller binary STL
    assert masked.stat().st_size < full.stat().st_size


# --------------------------------------------------------------------------
# generate_project_file
# --------------------------------------------------------------------------


def test_generate_project_file_contents(tmp_path):
    args = _args(tmp_path)
    disc_global = np.array([0, 1, 1, 2, 2, 2])
    disc_height = np.array([[0, 1], [2, 3]])
    project_path = tmp_path / "project.hfp"
    stl_path = tmp_path / "model.stl"
    generate_stl(np.zeros((2, 2), np.float32), str(stl_path), args.background_height, 10)

    generate_project_file(
        str(project_path),
        args,
        disc_global,
        disc_height,
        50,
        40,
        str(stl_path),
        args.csv_file,
    )
    data = json.loads(project_path.read_text())
    assert "filament_set" in data and len(data["filament_set"]) >= 2
    assert data["layer_height"] == args.layer_height
    assert data["slider_values"]
    assert data["stl"] == os.path.basename(str(stl_path))


def _hueforge_uuid(text):
    """The uuid as HueForge reads it (QUuid::fromString): 32 hex digits in
    8-4-4-4-12 groups, optionally in braces. None if it would be rejected."""
    import re

    m = re.fullmatch(r"\{?([0-9a-fA-F]{8}(?:-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12})\}?", text)
    return m.group(1).lower() if m else None


def test_project_file_uuids_are_readable_by_hueforge(tmp_path):
    # Issue #66: filaments added from the catalog carry ids like
    # "filamentcolors-4803". HueForge drops every filament_set entry whose
    # uuid it can't parse, so the project loaded white / shifted.
    args = _args(tmp_path)
    native = "{1800e8df-9e84-7c9a-088d-ac7b98baf869}"
    pandas.DataFrame(
        {
            "Brand": ["B0", "B1", "B2"],
            "Name": ["N0", "N1", "N2"],
            "Transmissivity": [1.0, 2.0, 3.0],
            "Color": ["#000000", "#111111", "#222222"],
            "Uuid": ["filamentcolors-4803", native, "filamentcolors-3500"],
        }
    ).to_csv(args.csv_file, index=False)
    args.background_material_index = 0
    disc_global = np.array([0, 1, 2, 0, 2, 1])
    disc_height = np.array([[0, 2], [4, 6]])
    project_path = tmp_path / "project.hfp"

    generate_project_file(
        str(project_path), args, disc_global, disc_height, 50, 40, "model.stl", args.csv_file,
    )
    entries = json.loads(project_path.read_text())["filament_set"]
    parsed = [_hueforge_uuid(e["uuid"]) for e in entries]
    assert None not in parsed, [e["uuid"] for e in entries]

    by_name = {}
    for e, u in zip(entries, parsed):
        by_name.setdefault(e["Name"], set()).add(u)
    # A filament used in several slots keeps one id, distinct filaments differ,
    # and an id that already is a uuid is passed through unchanged.
    assert all(len(ids) == 1 for ids in by_name.values()), by_name
    assert len({next(iter(ids)) for ids in by_name.values()}) == len(by_name)
    assert by_name["N1"] == {native.strip("{}")}


def test_project_file_catalog_uuid_is_stable_across_exports(tmp_path):
    # Same filament -> same uuid in every export, so HueForge sees it as the
    # same filament each time instead of a new one.
    from autoforge.Helper.OutputHelper import _hfp_filament

    mat = {"Brand": "Elegoo", "Name": "Yellow", "Color": "#f4cf15",
           "Transmissivity": 3.4, "Uuid": "filamentcolors-4803"}
    first, second = _hfp_filament(dict(mat))["uuid"], _hfp_filament(dict(mat))["uuid"]
    assert first == second and _hueforge_uuid(first)
