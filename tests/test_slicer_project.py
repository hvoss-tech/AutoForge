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
    assert cfg["machine_pause_gcode"].rstrip().endswith("M601")
    assert cfg["change_filament_gcode"] == ""
    # Every option listed in every section: OrcaSlicer takes the system preset
    # for anything unlisted (the GUI for everything when the lists are empty).
    listed = cfg["different_settings_to_system"]
    assert len(listed) == 3 + 2 and len(set(listed)) == 1
    keys = set(listed[0].split(";"))
    assert {"layer_height", "initial_layer_print_height", "sparse_infill_density", "nozzle_temperature",
            "machine_start_gcode", "min_layer_height"} <= keys
    assert "printer_settings_id" not in keys and "name" not in keys
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
        layers = ET.fromstring(zf.read("Metadata/custom_gcode_per_layer.xml")).findall(".//layer[@type='2']")
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
        layers = ET.fromstring(zf.read("Metadata/custom_gcode_per_layer.xml")).findall(".//layer[@type='2']")
    assert cfg["filament_colour"] == ["#000000", "#FF0000"]
    assert [l.attrib["top_z"] for l in layers] == ["0.4000"]
    assert "manual_filament_change" not in cfg

    # A failure (here an unknown printer) is reported, not raised.
    args.orca_printer = "No Such Printer"
    os.remove(tmp_path / "final_model.3mf")
    _export_slicer_project(args, np.array([0, 1, 1, 1]), heights, colors, ["A", "B"], 0)
    assert "Could not write the slicer project" in capsys.readouterr().out
    assert not os.path.exists(tmp_path / "final_model.3mf")


def test_fast_vertex_merge_equals_unique():
    rng = np.random.default_rng(0)
    pool = rng.normal(size=(500, 3)).astype(np.float32)
    corners = pool[rng.integers(0, 500, size=3000)]
    vertices, triangles = E._index_mesh(corners)
    expected = np.unique(corners, axis=0)
    assert len(vertices) == len(expected)
    np.testing.assert_array_equal(vertices[triangles].reshape(-1, 3), corners)


def test_read_stl_reuses_the_last_parse(tmp_path, monkeypatch):
    path = tmp_path / "m.stl"
    path.write_bytes(_stl_bytes())
    calls = []
    original = E._read_stl
    monkeypatch.setattr(E, "_read_stl", lambda src: calls.append(src) or original(src))
    E.read_stl(str(path))
    E.read_stl(str(path))
    assert len(calls) == 1
    path.write_bytes(_stl_bytes((5.0, 5.0, 1.0)))  # changed file: parsed again
    os.utime(path, ns=(1, 1))
    assert len(E.read_stl(str(path))[1]) == 12 and len(calls) == 2


def test_export_progress_is_monotonic_and_can_stop(acme):
    seen = []
    E.export_3mf(io.BytesIO(), _stl_bytes(), _plan(), "Acme One 0.4 nozzle", 0.08, 0.24,
                 progress=lambda f, stage: seen.append((f, stage)))
    fractions = [f for f, _ in seen]
    assert fractions == sorted(fractions) and fractions[-1] == 1.0
    assert "Writing the 3mf" in {s for _, s in seen}

    def stop(fraction, stage):
        if stage == "Writing the 3mf":
            raise E.Cancelled()

    with pytest.raises(E.Cancelled):
        E.export_3mf(io.BytesIO(), _stl_bytes(), _plan(), "Acme One 0.4 nozzle", 0.08, 0.24, progress=stop)


# --------------------------------------------------------------------- nozzle

def test_preset_follows_the_results_nozzle(acme):
    printer, note, warning = OP.printer_for_nozzle("Acme One 0.4 nozzle", 0.4)
    assert printer["name"] == "Acme One 0.4 nozzle" and note is None and warning is None
    # Same naming as the picked preset, not the HF variant.
    printer, note, warning = OP.printer_for_nozzle("Acme One 0.4 nozzle", 0.6)
    assert printer["name"] == "Acme One 0.6 nozzle" and warning is None
    assert "0.6 mm" in note
    printer, _, _ = OP.printer_for_nozzle("Acme One HF0.6 nozzle", 0.6)
    assert printer["name"] == "Acme One HF0.6 nozzle"
    assert OP.printer_for_nozzle("Acme One 0.6 nozzle", None)[0]["name"] == "Acme One 0.6 nozzle"


def test_missing_nozzle_keeps_the_preset_and_warns(acme):
    printer, note, warning = OP.printer_for_nozzle("Acme One 0.4 nozzle", 0.8)
    assert printer["name"] == "Acme One 0.4 nozzle" and note is None
    assert "no 0.8 mm nozzle preset for the Acme One (only 0.2, 0.4, 0.6 mm)" in warning
    assert "set the nozzle diameter to 0.8 mm" in warning


def test_export_uses_the_nozzle_preset_and_reports_it(acme):
    buf = io.BytesIO()
    info = E.export_3mf(buf, _stl_bytes(), _plan(), "Acme One 0.4 nozzle", 0.08, 0.24, nozzle=0.6)
    assert info["printer"] == "Acme One 0.6 nozzle" and info["warnings"] == [] and len(info["notes"]) == 1
    cfg = json.loads(zipfile.ZipFile(io.BytesIO(buf.getvalue())).read("Metadata/project_settings.config"))
    assert cfg["printer_settings_id"] == "Acme One 0.6 nozzle"
    assert cfg["nozzle_diameter"] == ["0.6"]
    info = E.export_3mf(io.BytesIO(), _stl_bytes(), _plan(), "Acme One 0.4 nozzle", 0.08, 0.24, nozzle=1.0)
    assert info["printer"] == "Acme One 0.4 nozzle" and len(info["warnings"]) == 1


def test_cli_warns_about_a_missing_nozzle_preset(acme, tmp_path, capsys):
    import argparse

    from autoforge.auto_forge import _export_slicer_project, cli_defaults
    from autoforge.Helper.OutputHelper import generate_stl

    heights = np.full((10, 10), 4)
    generate_stl(heights.astype(np.float32) * 0.08, str(tmp_path / "final_model.stl"), 0.24, maximum_x_y_size=20)
    args = argparse.Namespace(**cli_defaults())
    args.output_folder, args.layer_height, args.background_height = str(tmp_path), 0.08, 0.24
    args.csv_file, args.json_file = "", ""
    args.orca_printer, args.nozzle_diameter = "Acme One 0.4 nozzle", 0.8
    _export_slicer_project(args, np.array([0, 1, 1, 1]), heights, np.array([[0, 0, 0], [1, 0, 0]], np.float32), ["A", "B"], 0)
    out = capsys.readouterr().out
    assert "WARNING: OrcaSlicer has no 0.8 mm nozzle preset" in out
    assert os.path.exists(tmp_path / "final_model.3mf")


# ---------------------------------------------------------- AutoForge print settings

def test_print_settings_are_applied_only_when_asked(acme):
    printer = OP.find_printer("Acme One 0.4 nozzle")
    plain = E.project_settings(printer, _slots("PLA", "PLA"), 0.08, 0.24, False, acme)
    tuned = E.project_settings(printer, _slots("PLA", "PLA"), 0.08, 0.24, False, acme, autoforge_settings=True)
    assert "wall_loops" not in plain and plain["outer_wall_speed"] == ["200"]
    assert tuned["wall_loops"] == "1"
    assert tuned["bottom_shell_layers"] == "999" and tuned["top_shell_layers"] == "1"
    assert tuned["top_surface_pattern"] == "monotonic"
    assert tuned["sparse_infill_pattern"] == plain["sparse_infill_pattern"] == "rectilinear"
    assert tuned["wall_generator"] == "classic" and tuned["enable_arc_fitting"] == "disabled"
    # a cap keeps the profile's vector shape
    assert tuned["outer_wall_speed"] == ["100"]
    # not in the profile: written per print variant
    assert tuned["initial_layer_speed"] == ["30"]
    assert tuned["skirt_loops"] == "1"


def test_print_setting_caps_never_speed_a_printer_up(acme):
    assert E._apply_setting(["60", "300"], "max", 100) == ["60", "100"]
    assert E._apply_setting("2", "min", 1) == "2"
    assert E._apply_setting("0", "min", 1) == "1"
    assert E._apply_setting(["nil"], "max", 100) == ["nil"]  # Orca's "use the default" stays
    assert E._apply_setting("50%", "max", 100) == "50%"


def test_every_print_setting_is_a_known_orca_key():
    """A typo would be ignored by OrcaSlicer without a word; the bundled
    Orca project keys (from an Orca 2.5 export) list what exists."""
    known = set(json.load(open(os.path.join(os.path.dirname(__file__), "data", "orca25_project_keys.json"))))
    assert set(E.AUTOFORGE_PRINT_SETTINGS) <= known
    assert set(E.PROCESS_OVERRIDES) <= known


# ------------------------------------------------ what OrcaSlicer would refuse

def test_option_ranges_come_from_orcas_print_config(tmp_path):
    cpp = tmp_path / "PrintConfig.cpp"
    cpp.write_text("""
    const int max_temp = 1500;
    def = this->add("complete_print_exhaust_fan_speed", coInts);
    def->min = 0;
    def->max = 100;
    def = this->add_nullable("retraction_distances_when_cut", coFloats);
    def->min = 10;
    def->max = 18;
    def = this->add("nozzle_temperature", coInts);
    def->min = 0;
    def->max = max_temp;
    def = this->add("chamber_temperature", coInts);
    def->min = -max_temp;
    def = this->add("some_string", coString);
    def->min = SOMETHING_ELSE;
    """)
    assert OP.extract_option_ranges(str(cpp)) == {
        "complete_print_exhaust_fan_speed": [0.0, 100.0],
        "retraction_distances_when_cut": [10.0, 18.0],
        "nozzle_temperature": [0.0, 1500.0],
        "chamber_temperature": [-1500.0, None],
    }


def test_bundled_index_has_orcas_option_ranges():
    ranges = OP.load_printer_index()["option_ranges"]
    assert ranges["retraction_distances_when_cut"] == [10.0, 18.0]
    assert ranges["complete_print_exhaust_fan_speed"] == [0.0, 100.0]


def test_out_of_range_profile_values_are_clamped():
    """Creality K2 (30 mm cut retraction) and Cubicon (exhaust fan 255) had
    values OrcaSlicer refuses a project for."""
    cfg = {
        "retraction_distances_when_cut": ["30", "nil", "12"],
        "complete_print_exhaust_fan_speed": ["255", "60"],
        "wall_loops": "3",
        "top_surface_speed": "150%",
    }
    ranges = {"retraction_distances_when_cut": [10, 18], "complete_print_exhaust_fan_speed": [0, 100],
              "wall_loops": [0, 1000], "top_surface_speed": [0, 100]}
    changed = E.clamp_to_orca_ranges(cfg, ranges)
    assert cfg["retraction_distances_when_cut"] == ["18", "nil", "12"]
    assert cfg["complete_print_exhaust_fan_speed"] == ["100", "60"]
    assert cfg["wall_loops"] == "3" and cfg["top_surface_speed"] == "100%"
    assert sorted(changed) == ["complete_print_exhaust_fan_speed", "retraction_distances_when_cut", "top_surface_speed"]


@pytest.mark.parametrize("base, layer, nozzle, first", [
    (0.24, 0.08, ["0.4"], 0.24),
    (0.24, 0.08, ["0.2"], 0.16),
    (0.24, 0.04, ["0.2"], 0.2),
    (0.24, 0.08, ["0.8", "0.4"], 0.24),
    (0.24, 0.08, ["0.8", "0.2"], 0.16),  # the smallest nozzle counts
    (0.3, 0.1, "0.25", 0.2),
])
def test_first_layer_fits_the_nozzle_and_keeps_layer_tops(base, layer, nozzle, first):
    got = E.first_layer_height(base, layer, nozzle)
    assert got == pytest.approx(first)
    smallest = min(float(n) for n in (nozzle if isinstance(nozzle, list) else [nozzle]))
    assert got <= smallest + 1e-9
    steps = (base - got) / layer
    assert steps == pytest.approx(round(steps))  # the base top is a layer top


def test_layers_thicker_than_the_nozzle_are_refused():
    with pytest.raises(ValueError, match="nozzle"):
        E.first_layer_height(0.24, 0.25, ["0.2"])


def test_small_nozzle_export_splits_the_base_and_says_so(acme):
    buf = io.BytesIO()
    info = E.export_3mf(buf, _stl_bytes(), _plan(), "Acme One 0.4 nozzle", 0.08, 0.24, nozzle=0.2)
    assert info["printer"] == "Acme One 0.2 nozzle"
    assert any("0.16 mm first layer" in n for n in info["notes"])
    cfg = json.loads(zipfile.ZipFile(io.BytesIO(buf.getvalue())).read("Metadata/project_settings.config"))
    assert cfg["initial_layer_print_height"] == "0.16"
    layers = ET.fromstring(zipfile.ZipFile(io.BytesIO(buf.getvalue())).read("Metadata/custom_gcode_per_layer.xml")).findall(".//layer[@type='2']")
    assert [l.attrib["top_z"] for l in layers] == ["0.3200", "0.4000", "0.4800"]


def test_vendor_profile_contradictions_are_resolved_like_orca_asks():
    # relative E on Marlin needs a reset each layer
    cfg = {"use_relative_e_distances": "1", "gcode_flavor": "marlin", "layer_change_gcode": ";AFTER", "before_layer_change_gcode": ""}
    E.make_orca_consistent(cfg, is_bbl=False)
    assert cfg["layer_change_gcode"].splitlines()[-1].startswith("G92 E0")
    # ... on Klipper too (OrcaSlicer 2.5.0 refused a Klipper profile without it)
    cfg = {"use_relative_e_distances": "1", "gcode_flavor": "klipper", "layer_change_gcode": ";AFTER"}
    E.make_orca_consistent(cfg, is_bbl=False)
    assert "G92 E0" in cfg["layer_change_gcode"]
    # ... but not on Bambu printers, which don't need it
    cfg = {"use_relative_e_distances": "1", "gcode_flavor": "marlin", "layer_change_gcode": ";AFTER"}
    E.make_orca_consistent(cfg, is_bbl=True)
    assert cfg["layer_change_gcode"] == ";AFTER"
    # lower case is refused: made upper case
    cfg = {"use_relative_e_distances": "1", "gcode_flavor": "klipper", "before_layer_change_gcode": "g92 e0 ; reset"}
    E.make_orca_consistent(cfg, is_bbl=False)
    assert cfg["before_layer_change_gcode"] == "G92 E0 ; reset"
    # absolute E: the reset goes (Dremel 3D40)
    cfg = {"use_relative_e_distances": "0", "before_layer_change_gcode": ";BEFORE\nG92 E0\nM117", "layer_change_gcode": ""}
    E.make_orca_consistent(cfg, is_bbl=False)
    assert "G92" not in cfg["before_layer_change_gcode"] and "M117" in cfg["before_layer_change_gcode"]
    # firmware retraction and wipe (VIRTU E3, E3NG)
    cfg = {"use_firmware_retraction": "1", "wipe": ["1", "1"], "filament_wipe": ["nil", "1"]}
    E.make_orca_consistent(cfg, is_bbl=False)
    assert cfg["wipe"] == ["0", "0"] and cfg["filament_wipe"] == ["nil", "0"]
    # bridge line wider than the smaller nozzle (Snapmaker U1 0.4+0.6)
    cfg = {"nozzle_diameter": ["0.4", "0.6"], "bridge_line_width": "100%"}
    E.make_orca_consistent(cfg, is_bbl=False)
    assert cfg["bridge_line_width"] == "0.4"
    # a plate type all filaments have a bed temperature for (Creality K2 Pro)
    cfg = {"cool_plate_temp": ["0", "0"], "textured_plate_temp": ["55", "60"]}
    E.make_orca_consistent(cfg, is_bbl=False)
    assert cfg["curr_bed_type"] == "Textured PEI Plate"
    cfg = {"default_bed_type": "Cool Plate", "cool_plate_temp": ["35"], "textured_plate_temp": ["55"]}
    E.make_orca_consistent(cfg, is_bbl=False)
    assert cfg["curr_bed_type"] == "Cool Plate"
    # an empty script entry is no script (the CLI refuses any)
    cfg = {"post_process": [""]}
    E.make_orca_consistent(cfg, is_bbl=False)
    assert cfg["post_process"] == []
    cfg = {"post_process": ["/usr/bin/thumbs"]}
    E.make_orca_consistent(cfg, is_bbl=False)
    assert cfg["post_process"] == ["/usr/bin/thumbs"]


def test_filament_overrides_get_their_printer_options_range():
    cfg = {"filament_retraction_distances_when_cut": ["30", "nil"]}
    E.clamp_to_orca_ranges(cfg, {"retraction_distances_when_cut": [10, 18]})
    assert cfg["filament_retraction_distances_when_cut"] == ["18", "nil"]


def test_project_holds_every_option_and_lists_no_differences(acme, monkeypatch):
    """OrcaSlicer fills a missing option from whatever printer it has loaded
    and resets unlisted ones to its own system preset; a complete config with
    empty difference lists is used exactly as written."""
    index = dict(OP.load_printer_index())
    index["defaults"] = {"machine": {"bed_exclude_area": ["0x0"]}, "process": {"skirt_distance": "2"}, "filament": {"filament_notes": [""]}}
    monkeypatch.setattr(OP, "_index_cache", index)
    printer = OP.find_printer("Acme One 0.4 nozzle")
    cfg = E.project_settings(printer, _slots("PLA", "PLA"), 0.08, 0.24, False, acme)
    assert cfg["bed_exclude_area"] == ["0x0"] and cfg["skirt_distance"] == "2"
    assert cfg["filament_notes"] == ["", ""]
    assert "bed_exclude_area" in cfg["different_settings_to_system"][-1].split(";")


def test_nozzle_lists_written_as_one_string_are_split():
    assert OP.nozzle_list(["0.4,0.4"]) == ["0.4", "0.4"]
    assert OP.nozzle_list("0.4") == ["0.4"]
    assert E.first_layer_height(0.24, 0.08, ["0.6,0.2"]) == pytest.approx(0.16)


def test_bundled_index_has_no_pellet_nozzles():
    names = {p["name"] for p in OP.load_printer_index()["printers"]}
    assert not names & {"Ginger G1 5.0 nozzle", "Ginger G1 8.0 nozzle", "FELIX Food Single 4.00mm nozzle"}
    assert all(float(p["nozzle"]) <= OP.MAX_NOZZLE_MM for p in OP.load_printer_index()["printers"])


def test_gcode_written_as_a_list_stays_gcode():
    """Prusa MK4S / CORE One write layer_change_gcode as ["..."]; turning
    that into the text "['...']" made OrcaSlicer refuse to slice."""
    cfg = {"use_relative_e_distances": "1", "gcode_flavor": "marlin2",
           "layer_change_gcode": [";AFTER_LAYER_CHANGE\n;[layer_z]"], "before_layer_change_gcode": [";BEFORE"]}
    E.make_orca_consistent(cfg, is_bbl=False)
    assert isinstance(cfg["layer_change_gcode"], list)
    assert cfg["layer_change_gcode"][0].startswith(";AFTER_LAYER_CHANGE\n;[layer_z]\nG92 E0")
    assert cfg["before_layer_change_gcode"] == [";BEFORE"]


@pytest.mark.parametrize("order, sequence, infill_first", [
    ("infill/outer wall/inner wall", "outer wall/inner wall", "1"),
    ("inner wall/outer wall/infill", "inner wall/outer wall", "0"),
    ("inner-outer-inner wall/infill", "inner-outer-inner wall", "0"),
])
def test_legacy_wall_order_is_converted_like_orca_does(order, sequence, infill_first):
    """Left in, OrcaSlicer converts it on load and marks the project as
    changed - and then resets the whole project to the system preset."""
    cfg = {"wall_infill_order": order}
    E.convert_legacy_options(cfg)
    assert "wall_infill_order" not in cfg
    assert cfg["wall_sequence"] == sequence and cfg["is_infill_first"] == infill_first


def test_legacy_hybrid_support_is_converted():
    cfg = {"support_type": "hybrid(auto)"}
    E.convert_legacy_options(cfg)
    assert cfg == {"support_type": "tree(auto)", "support_style": "tree_hybrid"}


def test_nil_only_stays_where_orca_allows_it():
    cfg = {"filament_flush_temp": ["nil", "220"], "filament_retraction_length": ["nil", "0.8"], "name": "x"}
    changed = E.fill_nil_values(cfg, {"filament_retraction_length"}, {"filament": {"filament_flush_temp": ["0"]}})
    assert cfg["filament_flush_temp"] == ["0", "220"]
    assert cfg["filament_retraction_length"] == ["nil", "0.8"]
    assert changed == ["filament_flush_temp"]


def test_bundled_index_knows_nullable_and_integer_options():
    index = OP.load_printer_index()
    assert "filament_retraction_length" in index["nullable"]
    assert "filament_flush_temp" in index["nullable"] and "filament_flush_temp" in index["int_options"]


def test_nil_in_integer_options_takes_the_printers_value():
    """Kobra S1 Max: filament_flush_temp "nil" is nullable but an int, and
    OrcaSlicer's range check refuses an int nil."""
    cfg = {"filament_flush_temp": ["nil", "nil"], "flush_temp": ["230"],
           "filament_retraction_length": ["nil", "nil"], "retraction_length": ["0.8"]}
    E.fill_nil_values(cfg, {"filament_flush_temp", "filament_retraction_length"}, {}, {"filament_flush_temp"})
    assert cfg["filament_flush_temp"] == ["230", "230"]
    assert cfg["filament_retraction_length"] == ["nil", "nil"]  # a float nil is fine


def test_nullable_options_come_from_orcas_print_config(tmp_path):
    cpp = tmp_path / "PrintConfig.cpp"
    cpp.write_text(
        'const std::vector<std::string> filament_extruder_override_keys = {\n    "filament_z_hop", "filament_wipe"\n};\n'
        'def = this->add("filament_flush_temp", coInts);\n'
        'def = this->add("long_retractions_when_cut", coBools);\n    def->nullable = true;\n'
    )
    assert OP.extract_nullable_options(str(cpp)) == ["filament_wipe", "filament_z_hop", "long_retractions_when_cut"]



def test_hand_swaps_are_pause_entries_at_every_swap(acme):
    """OrcaSlicer skips the change-filament G-code on the first tool change
    of a print when filament changes are manual; on Bambu printers that was
    the first real colour change, printed without stopping. A pause entry
    (machine_pause_gcode) at each swap stops every time."""
    for manual in (True, False):
        buf = io.BytesIO()
        E.export_3mf(buf, _stl_bytes(), _plan(), "Acme One 0.4 nozzle", 0.08, 0.24, manual_swaps=manual)
        root = ET.fromstring(zipfile.ZipFile(io.BytesIO(buf.getvalue())).read("Metadata/custom_gcode_per_layer.xml"))
        tools = [l.attrib["top_z"] for l in root.findall(".//layer[@type='2']")]
        pauses = [l.attrib["top_z"] for l in root.findall(".//layer[@type='1']")]
        assert tools == ["0.3200", "0.4000", "0.4800"]
        assert pauses == (tools if manual else [])


@pytest.mark.parametrize("layer, base, min_expected, max_expected", [
    (0.04, 0.24, ["0.04"], ["0.28"]),   # thinner than the printer's minimum
    (0.32, 0.32, ["0.07"], ["0.32"]),   # thicker than its maximum
    (0.12, 0.24, ["0.07"], ["0.28"]),   # within limits: untouched
])
def test_layer_limits_follow_autoforges_layer_height(acme, layer, base, min_expected, max_expected):
    printer = OP.find_printer("Acme One 0.4 nozzle")
    cfg = E.project_settings(printer, _slots("PLA", "PLA"), layer, base, False, acme)
    assert cfg["layer_height"] == f"{layer:g}" and cfg["initial_layer_print_height"] == f"{base:g}"
    assert cfg["min_layer_height"] == min_expected and cfg["max_layer_height"] == max_expected


# ------------------------------------------------------- Bambu Studio values

def test_bambu_printers_get_values_bambu_studio_reads():
    """Bambu Studio names some values differently from OrcaSlicer and swaps
    a name it doesn't know for its default: sparse "rectilinear" became
    "cubic", which it refuses at 100 % infill, so it dropped the project's
    settings (0.04 mm layers opened as 0.2 mm). OrcaSlicer reads Bambu's
    names as legacy values, so one file works in both."""
    cfg = {
        "sparse_infill_pattern": "rectilinear",
        "bottom_surface_pattern": "rectilinear",
        "ironing_pattern": ["rectilinear"],
        "top_surface_pattern": "monotonic",
        "internal_solid_infill_pattern": "alignedrectilinear",
        "enable_arc_fitting": "disabled",
        "ensure_vertical_shell_thickness": "none",
    }
    changed = E.make_bambu_compatible(cfg)
    assert cfg == {
        "sparse_infill_pattern": "zig-zag",
        "bottom_surface_pattern": "zig-zag",
        "ironing_pattern": ["zig-zag"],
        "top_surface_pattern": "monotonic",
        "internal_solid_infill_pattern": "alignedrectilinear",
        "enable_arc_fitting": "0",
        "ensure_vertical_shell_thickness": "0",
    }
    assert set(changed) == {
        "sparse_infill_pattern", "bottom_surface_pattern", "ironing_pattern",
        "enable_arc_fitting", "ensure_vertical_shell_thickness",
    }
    assert E.make_bambu_compatible(cfg) == []  # already Bambu's


def test_ensure_vertical_shell_levels_map_to_bambus_old_switch():
    for orca, bambu in (("none", "0"), ("ensure_critical_only", "0"), ("ensure_moderate", "0"), ("ensure_all", "1")):
        cfg = {"ensure_vertical_shell_thickness": orca}
        E.make_bambu_compatible(cfg)
        assert cfg["ensure_vertical_shell_thickness"] == bambu
    cfg = {"enable_arc_fitting": "enabled"}
    E.make_bambu_compatible(cfg)
    assert cfg["enable_arc_fitting"] == "1"
