"""OrcaSlicer project (.3mf) export: profile resolution, printer index,
swap plan, project settings and the archive itself."""

import io
import json
import os
import re
import urllib.error
import xml.etree.ElementTree as ET
import zipfile

import numpy as np
import pytest
import trimesh

from autoforge.Helper import Export3MF as E
from autoforge.Helper import OrcaProfiles as OP
from autoforge.Helper.OutputHelper import generate_swap_instructions
from orca_fixtures import make_profiles


@pytest.fixture
def profiles_dir(tmp_path):
    return make_profiles(str(tmp_path / "profiles"))


@pytest.fixture
def acme(profiles_dir, monkeypatch):
    """The fixture tree as the printer index and the profile source."""
    index = OP.build_printer_index(profiles_dir)
    monkeypatch.setattr(OP, "_index_cache", index)
    monkeypatch.setenv("AUTOFORGE_ORCA_PROFILES", profiles_dir)
    return OP.OrcaProfiles(OP.LocalProfileSource(profiles_dir))


def _stl_bytes(size=(20.0, 10.0, 1.0)):
    buf = io.BytesIO()
    trimesh.creation.box(extents=list(size)).export(buf, file_type="stl")
    return buf.getvalue()


# ---------------------------------------------------------------- packaging

def test_bundled_printer_index_is_usable():
    index = OP.load_printer_index()
    assert index["orca_tag"] == OP.ORCA_TAG
    names = {p["name"] for p in index["printers"]}
    assert len(names) == len(index["printers"]) > 1000
    x1c = OP.find_printer("Bambu Lab X1 Carbon 0.4 nozzle")
    assert x1c["bed"] == [0.0, 0.0, 256.0, 256.0]
    assert "PLA" in x1c["filaments"]
    assert all("PLA" in p["filaments"] and p["bed"][2] > p["bed"][0] for p in index["printers"])


# ---------------------------------------------------------- profile resolving

def test_resolve_follows_inherits_then_includes_then_own_keys(acme):
    cfg = acme.resolve("Acme", "filament", "Acme PLA @Acme One")
    # own key over the parent chain (library common -> vendor base)
    assert cfg["nozzle_temperature"] == ["210"]
    # the include's keys sit between the parent and the preset's own keys
    assert cfg["filament_extruder_variant"] == ["Direct Drive Standard", "Direct Drive High Flow"]
    assert cfg["filament_max_volumetric_speed"] == ["15", "25"]
    assert cfg["filament_id"] == "ACME1"
    assert "inherits" not in cfg and "name" not in cfg


def test_resolve_reads_files_with_a_bom(acme):
    assert acme.resolve("Acme", "machine", "Acme One 0.4 nozzle")["machine_pause_gcode"] == "M601"


def test_filaments_fall_back_to_orca_library(acme):
    assert acme.resolve("Acme", "filament", "Generic PETG @System")["nozzle_temperature"] == ["245"]
    with pytest.raises(OP.ProfileError):
        acme.resolve("Acme", "machine", "Generic PETG @System")


def test_inheritance_cycle_is_an_error(profiles_dir):
    path = os.path.join(profiles_dir, "Acme", "process", "fdm_process_acme.json")
    data = json.load(open(path))
    data["inherits"] = "0.20mm Standard @Acme One"
    json.dump(data, open(path, "w"))
    with pytest.raises(OP.ProfileError, match="itself"):
        OP.OrcaProfiles(OP.LocalProfileSource(profiles_dir)).resolve("Acme", "process", "0.20mm Standard @Acme One")


class _Response(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def test_github_source_caches_and_never_caches_an_error_page(tmp_path, monkeypatch):
    calls = []

    def urlopen(request, timeout):
        calls.append(request.full_url)
        if "Bad" in request.full_url:
            return _Response(b"<html>rate limited</html>")
        return _Response(b'{"name": "x"}')

    monkeypatch.setattr(OP.urllib.request, "urlopen", urlopen)
    source = OP.GitHubProfileSource(str(tmp_path), tag="v9")
    assert source.read("Acme/machine/One nozzle.json") == {"name": "x"}
    assert source.read("Acme/machine/One nozzle.json") == {"name": "x"}
    assert len(calls) == 1
    assert calls[0].endswith("/v9/resources/profiles/Acme/machine/One%20nozzle.json")
    with pytest.raises(OP.ProfileError):
        source.read("Bad.json")
    assert not os.path.exists(tmp_path / "v9" / "Bad.json")


def test_github_source_reports_missing_internet(tmp_path, monkeypatch):
    def urlopen(request, timeout):
        raise urllib.error.URLError("no route to host")

    monkeypatch.setattr(OP.urllib.request, "urlopen", urlopen)
    with pytest.raises(OP.ProfileDownloadError, match="internet"):
        OP.GitHubProfileSource(str(tmp_path)).read("Acme.json")


# --------------------------------------------------------------- printer index

def test_index_skips_belt_printers_and_fixes_incompatible_default_process(acme):
    printers = {p["name"]: p for p in OP.load_printer_index()["printers"]}
    assert "Acme Belt 0.4 nozzle" not in printers
    # the default names a process that doesn't list the printer: Orca refuses
    # such a project, so the closest compatible one is used
    assert printers["Acme Lost 0.4 nozzle"]["process"] == "0.20mm Standard @Acme Lost"
    # a ';' list of defaults: the first compatible one
    assert printers["Acme Duo 0.4 nozzle"]["process"] == "0.20mm Standard @Acme One"
    assert printers["Acme Duo 0.4 nozzle"]["bed"] == [0.0, 0.0, 300.0, 150.0]


def test_index_matches_filament_presets_to_the_printer(acme):
    printers = {p["name"]: p for p in OP.load_printer_index()["printers"]}
    one = printers["Acme One 0.4 nozzle"]["filaments"]
    assert one == {"PLA": "Acme PLA @Acme One", "PETG": "Acme PETG @Acme One", "ABS": "Generic ABS @System"}
    # a 2.85 mm printer never gets the 1.75 mm library presets
    assert printers["Acme Big 0.5 nozzle"]["filaments"] == {"PLA": "Acme PLA 2.85 @Acme Big"}


# ------------------------------------------------------------------ swap plan

def _instructions_swaps(lines):
    out = []
    for line in lines:
        m = re.match(r"At layer #(\d+) \(([\d.]+)mm\) swap to (.*)", line)
        if m:
            out.append((float(m.group(2)), m.group(3)))
    return out


@pytest.mark.parametrize("base_index", [None, 0, 2])
def test_swap_plan_matches_swap_instructions(base_index):
    rng = np.random.default_rng(3)
    names = ["Black", "Red", "White", "Blue"]
    materials = [{"color": c, "name": n} for c, n in zip(["#000000", "#FF0000", "#FFFFFF", "#0000FF"], names)]
    for _ in range(25):
        stack = rng.integers(0, 4, size=12)
        stack = np.repeat(stack, rng.integers(1, 3, size=12))[:12]
        heights = np.array([[12, 3], [0, 7]])
        lines = generate_swap_instructions(stack, heights, 0.08, 3, 0.24, names, None, base_index)
        plan = E.swap_plan(stack, 12, 0.08, 0.24, materials, base_index, "#123456")
        expected = _instructions_swaps(lines)
        got = [(round(z, 2), plan.slots[s].name) for z, s in plan.swaps]
        assert got == [(round(z, 2), n) for z, n in expected]
        if base_index is None:
            assert plan.slots[0].color == "#123456"
        else:
            assert plan.slots[0].name == names[base_index]


def test_swap_plan_reuses_a_slot_for_a_returning_filament():
    materials = [{"color": "#000000"}, {"color": "#ff0000"}, {"color": "#00ff00"}]
    plan = E.swap_plan([0, 1, 1, 2, 1], 5, 0.04, 0.2, materials, 0)
    assert [s.color for s in plan.slots] == ["#000000", "#FF0000", "#00FF00"]
    assert plan.swaps == [(0.28, 1), (0.36, 2), (0.4, 1)]


# ----------------------------------------------------------------------- mesh

def test_read_stl_indexes_binary_and_ascii_alike():
    mesh = trimesh.creation.icosphere(subdivisions=2)
    binary, ascii_ = io.BytesIO(), io.BytesIO()
    mesh.export(binary, file_type="stl")
    mesh.export(ascii_, file_type="stl_ascii")
    for data in (binary.getvalue(), ascii_.getvalue()):
        vertices, triangles = E.read_stl(data)
        assert len(vertices) == len(mesh.vertices)
        assert len(triangles) == len(mesh.faces)
        np.testing.assert_allclose(vertices[triangles], mesh.triangles, atol=1e-5)


def test_read_stl_merges_negative_zero():
    tri = np.array([[[0.0, 0.0, 0.0], [1, 0, 0], [0, 1, 0]], [[-0.0, 0.0, 0.0], [0, 1, 0], [0, 0, 1]]], dtype=np.float32)
    data = bytearray(80) + np.uint32(2).tobytes()
    for t in tri:
        data += np.zeros(3, np.float32).tobytes() + t.tobytes() + b"\0\0"
    vertices, triangles = E.read_stl(bytes(data))
    assert len(vertices) == 4
    assert triangles[0][0] == triangles[1][0]


def test_read_stl_rejects_garbage():
    with pytest.raises(ValueError):
        E.read_stl(b"not an stl at all")


# ----------------------------------------------------------- project settings

def _slots(*types):
    colors = ["#000000", "#FF0000", "#FFFFFF", "#0000FF"]
    return [E.Slot(colors[i], f"F{i}", t) for i, t in enumerate(types)]


def test_project_settings_carry_full_presets_and_heightmap_overrides(acme):
    printer = OP.find_printer("Acme One 0.4 nozzle")
    cfg = E.project_settings(printer, _slots("PLA", "PETG", "PLA Matte"), 0.04, 0.24, True, acme)
    assert cfg["printer_settings_id"] == "Acme One 0.4 nozzle"
    assert cfg["print_settings_id"] == "0.20mm Standard @Acme One"
    assert cfg["machine_start_gcode"] == "G28 ; acme"  # the printer's own G-code, not a blank default
    assert cfg["filament_settings_id"] == ["Acme PLA @Acme One", "Acme PETG @Acme One", "Acme PLA @Acme One"]
    assert cfg["filament_type"] == ["PLA", "PETG", "PLA"]
    # per-variant: PLA has two variants (its include), PETG one
    assert cfg["nozzle_temperature"] == ["210", "210", "240", "210", "210"]
    assert cfg["filament_self_index"] == ["1", "1", "2", "3", "3"]
    assert cfg["filament_colour"] == ["#000000", "#FF0000", "#FFFFFF"]
    assert cfg["layer_height"] == "0.04"
    assert cfg["initial_layer_print_height"] == "0.24"
    assert cfg["sparse_infill_density"] == "100%"
    assert cfg["enable_prime_tower"] == "0"
    assert cfg["min_layer_height"] == ["0.04"]  # the printer's 0.07 would clamp 0.04 layers
    assert cfg["manual_filament_change"] == "1"
    assert cfg["change_filament_gcode"].rstrip().endswith("M601")
    changed = cfg["different_settings_to_system"]
    assert len(changed) == 3 + 2  # print, one per filament, printer
    assert set(changed[0].split(";")) == set(E.PROCESS_OVERRIDES) | {"layer_height", "initial_layer_print_height"}
    assert changed[1:4] == ["", "", ""]
    assert {"min_layer_height", "manual_filament_change", "change_filament_gcode"} <= set(changed[-1].split(";"))
    assert "compatible_printers" not in cfg


def test_project_settings_per_variant_layout(acme):
    """Orca reads per-variant filament vectors with filament_self_index; a
    mismatch aborted OrcaSlicer on multi-extruder printers."""
    printer = OP.find_printer("Acme Duo 0.4 nozzle")
    cfg = E.project_settings(printer, _slots("PLA", "PETG"), 0.08, 0.24, False, acme)
    # PLA has two variants (its include), the library PETG one
    assert cfg["filament_self_index"] == ["1", "1", "2"]
    assert cfg["filament_extruder_variant"] == ["Direct Drive Standard", "Direct Drive High Flow", "Direct Drive Standard"]
    assert cfg["filament_max_volumetric_speed"] == ["15", "25", "15"]
    # printer per-variant vectors padded to the printer's two variants
    assert cfg["retraction_length"] == ["0.8", "0.8"]
    assert "manual_filament_change" not in cfg


def test_flush_matrix_is_one_block_with_orcas_volumes(acme):
    cfg = {
        "nozzle_volume": ["107", "107"], "enable_long_retraction_when_cut": "2",
        "long_retractions_when_cut": ["0", "0"], "filament_long_retractions_when_cut": ["1", "1", "1", "1"],
        "retraction_distances_when_cut": ["18", "18"], "filament_retraction_distances_when_cut": ["18"] * 4,
        "nozzle_diameter": ["0.4", "0.4"],
    }
    # Values OrcaSlicer 2.5 itself wrote for these two colours on an X1 Carbon.
    assert E.flush_matrix(cfg, ["#F2754E", "#FFFFFF"]) == ["0", "458", "229", "0"]
    printer = OP.find_printer("Acme One 0.4 nozzle")
    settings = E.project_settings(printer, _slots("PLA", "PLA", "PLA"), 0.08, 0.24, False, acme)
    assert len(settings["flush_volumes_matrix"]) == 9


def test_wipe_tower_spot_is_on_the_bed(acme):
    printer = OP.find_printer("Acme Duo 0.4 nozzle")
    cfg = E.project_settings(printer, _slots("PLA", "PLA"), 0.08, 0.24, False, acme)
    assert 0 <= float(cfg["wipe_tower_y"][0]) < 150 and 0 <= float(cfg["wipe_tower_x"][0]) < 300


def test_auto_swaps_by_printer():
    assert E.printer_changes_filament({"vendor": "BBL", "name": "Bambu Lab A1 mini 0.4 nozzle"})
    assert E.printer_changes_filament({"vendor": "Prusa", "name": "Prusa MK4 MMU3 0.4 nozzle", "model": "Prusa MK4 MMU3"})
    assert E.printer_changes_filament({"vendor": "Prusa", "name": "Prusa XL 5T 0.4 nozzle", "extruders": 5})
    assert not E.printer_changes_filament({"vendor": "Prusa", "name": "Prusa MK4 0.4 nozzle", "model": "Prusa MK4"})
    assert not E.printer_changes_filament({"vendor": "Sovol", "name": "Sovol SV02 0.4 nozzle", "extruders": 2})


# ------------------------------------------------------------------- archive

def _plan():
    return E.SwapPlan(slots=_slots("PLA", "PETG", "PLA"), swaps=[(0.32, 1), (0.40, 2), (0.48, 1)])


def test_export_3mf_archive(acme):
    buf = io.BytesIO()
    E.export_3mf(buf, _stl_bytes(), _plan(), "Acme One 0.4 nozzle", 0.08, 0.24, title="cat & dog")
    with zipfile.ZipFile(io.BytesIO(buf.getvalue())) as zf:
        names = set(zf.namelist())
        assert {"[Content_Types].xml", "_rels/.rels", "3D/3dmodel.model", "3D/_rels/3dmodel.model.rels",
                "3D/Objects/object_1.model", "Metadata/project_settings.config",
                "Metadata/model_settings.config", "Metadata/custom_gcode_per_layer.xml"} <= names
        root = ET.fromstring(zf.read("3D/3dmodel.model"))
        ns = {"m": "http://schemas.microsoft.com/3dmanufacturing/core/2015/02"}
        assert root.find("m:metadata[@name='Title']", ns).text == "cat & dog"
        item = root.find("m:build/m:item", ns)
        tx, ty, tz = map(float, item.attrib["transform"].split()[9:])
        obj = ET.fromstring(zf.read("3D/Objects/object_1.model"))
        verts = np.array([[float(v.attrib[a]) for a in "xyz"] for v in obj.iter("{%s}vertex" % ns["m"])])
        assert len(verts) == 8 and len(list(obj.iter("{%s}triangle" % ns["m"]))) == 12
        # centred on the 200 x 200 bed, resting on it
        assert (verts[:, 0].min() + verts[:, 0].max()) / 2 + tx == pytest.approx(100)
        assert (verts[:, 1].min() + verts[:, 1].max()) / 2 + ty == pytest.approx(100)
        assert verts[:, 2].min() + tz == pytest.approx(0)
        layers = ET.fromstring(zf.read("Metadata/custom_gcode_per_layer.xml")).findall(".//layer")
        assert [(l.attrib["top_z"], l.attrib["extruder"], l.attrib["color"]) for l in layers] == [
            ("0.3200", "2", "#FF0000"), ("0.4000", "3", "#FFFFFF"), ("0.4800", "2", "#FF0000")]
        cfg = json.loads(zf.read("Metadata/project_settings.config"))
        # Acme isn't a printer that changes filament itself: pauses by default
        assert cfg["manual_filament_change"] == "1"


def test_export_3mf_refuses_a_model_bigger_than_the_bed(acme):
    with pytest.raises(ValueError, match="bed"):
        E.export_3mf(io.BytesIO(), _stl_bytes((250.0, 10.0, 1.0)), _plan(), "Acme One 0.4 nozzle", 0.08, 0.24)


def test_export_3mf_unknown_printer(acme):
    with pytest.raises(OP.ProfileError):
        E.export_3mf(io.BytesIO(), _stl_bytes(), _plan(), "No Such Printer", 0.08, 0.24)


def test_export_3mf_is_fast_on_a_large_mesh(acme):
    import time

    mesh = trimesh.creation.icosphere(subdivisions=6)  # 82k triangles
    mesh.apply_scale(50)
    buf = io.BytesIO()
    mesh.export(buf, file_type="stl")
    start = time.perf_counter()
    E.export_3mf(io.BytesIO(), buf.getvalue(), _plan(), "Acme One 0.4 nozzle", 0.08, 0.24)
    # the pure-Python parser of the original PR needed ~1 s for this
    assert time.perf_counter() - start < 2.0


# ----------------------------------------------------------------------- CLI

def test_cli_writes_the_slicer_project_only_when_asked(acme, tmp_path, capsys):
    import argparse

    from autoforge.auto_forge import _export_slicer_project, cli_defaults
    from autoforge.Helper.OutputHelper import generate_stl

    heights = np.full((10, 10), 4)
    generate_stl(heights.astype(np.float32) * 0.08, str(tmp_path / "final_model.stl"), 0.24, maximum_x_y_size=20)
    args = argparse.Namespace(**cli_defaults())
    args.output_folder = str(tmp_path)
    args.layer_height, args.background_height = 0.08, 0.24
    args.csv_file, args.json_file = "", ""
    assert args.orca_printer == "" and args.orca_manual_swaps == "auto"
    args.orca_printer, args.orca_manual_swaps = "Acme One 0.4 nozzle", "no"
    colors = np.array([[0, 0, 0], [1, 0, 0]], dtype=np.float32)
    _export_slicer_project(args, np.array([0, 1, 1, 1]), heights, colors, ["Acme - Black", "Acme - Red"], 0)
    with zipfile.ZipFile(tmp_path / "final_model.3mf") as zf:
        cfg = json.loads(zf.read("Metadata/project_settings.config"))
        layers = ET.fromstring(zf.read("Metadata/custom_gcode_per_layer.xml")).findall(".//layer")
    assert cfg["filament_colour"] == ["#000000", "#FF0000"]
    assert [l.attrib["top_z"] for l in layers] == ["0.4000"]
    assert "manual_filament_change" not in cfg

    # A failure (here an unknown printer) is reported, not raised.
    args.orca_printer = "No Such Printer"
    os.remove(tmp_path / "final_model.3mf")
    _export_slicer_project(args, np.array([0, 1, 1, 1]), heights, colors, ["A", "B"], 0)
    assert "Could not write the slicer project" in capsys.readouterr().out
    assert not os.path.exists(tmp_path / "final_model.3mf")
