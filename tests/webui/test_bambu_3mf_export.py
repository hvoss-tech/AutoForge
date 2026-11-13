"""Tests for Bambu Studio / OrcaSlicer .3mf export integration."""

import io
import json
import os
import tempfile
import xml.etree.ElementTree as ET
import zipfile

import numpy as np
import pytest
import trimesh
from fastapi.testclient import TestClient

from autoforge.webui.server import create_app


def _sample_stl_bytes() -> bytes:
    mesh = trimesh.creation.box(extents=[10.0, 10.0, 1.0])
    buf = io.BytesIO()
    mesh.export(buf, file_type="stl")
    return buf.getvalue()


@pytest.fixture
def client():
    return TestClient(create_app())


def _completed_stl_job(stl_bytes: bytes | None = None, with_bambu: bool = False):
    from autoforge.webui.config import config
    from autoforge.webui.services.optimization_service import get_optimization_service

    svc = get_optimization_service()
    job = svc.create_job({"iterations": 10, "input_image": "a.png"})
    svc.update_status(job.job_id, "completed")
    job_dir = os.path.join(config.checkpoints_dir, job.job_id)
    os.makedirs(job_dir, exist_ok=True)

    raw_stl = stl_bytes if stl_bytes is not None else _sample_stl_bytes()
    with open(os.path.join(job_dir, "final_model.stl"), "wb") as f:
        f.write(raw_stl)
    with open(os.path.join(job_dir, "swap_instructions.txt"), "w", encoding="utf-8") as f:
        f.write("Print at 100% infill\n")
    if with_bambu:
        from autoforge.helpers.export_3mf import export_bambu_3mf

        export_bambu_3mf(
            raw_stl,
            {"settings": {"layer_height": 0.04, "background_height": 0.24}, "colorSliders": [], "activeFilaments": []},
            os.path.join(job_dir, "final_model_bambu.3mf"),
        )
    return job, job_dir


def test_export_bambu_3mf_file_and_buffer_roundtrip(tmp_path):
    from autoforge.helpers.export_3mf import export_bambu_3mf

    stl_bytes = _sample_stl_bytes()
    project = {
        "settings": {"layer_height": 0.04, "background_height": 0.24, "background_color": "#000000"},
        "activeFilaments": [
            {"uuid": "u1", "color": "#112233", "filament_type": "PLA", "brand": "Bambu", "name": "Basic Black"},
            {"uuid": "u2", "color": "#AABBCC", "filament_type": "", "brand": "", "name": "Matte Ice Blue"},
            {"uuid": "u3", "color": "#DDEEFF", "filament_type": "PLA", "brand": "Polymaker", "name": "PolyTerra"},
        ],
        "colorSliders": [
            {"layer": 3, "filament_uuid": "u1", "enabled": True},
            {"layer": 6, "filament_uuid": "u2", "enabled": True},
            {"layer": 8, "filament_uuid": "u3", "enabled": True},
        ],
    }

    out_file = tmp_path / "model.3mf"
    export_bambu_3mf(stl_bytes, project, str(out_file))

    mem_buf = io.BytesIO()
    export_bambu_3mf(stl_bytes, json.dumps(project), mem_buf)
    mem_buf.seek(0)

    for src in (out_file, mem_buf):
        with zipfile.ZipFile(src) as zf:
            names = set(zf.namelist())
            assert {
                "[Content_Types].xml",
                "_rels/.rels",
                "3D/3dmodel.model",
                "Metadata/model_settings.config",
                "Metadata/project_settings.config",
                "Metadata/custom_gcode_per_layer.xml",
                "Metadata/filament_sequence.json",
                "Metadata/slice_info.config",
            }.issubset(names)
            ns = {"m": "http://schemas.microsoft.com/3dmanufacturing/core/2015/02"}
            model_root = ET.fromstring(zf.read("3D/3dmodel.model"))
            metadata = {el.attrib["name"]: el.text for el in model_root.findall("m:metadata", ns)}
            assert metadata["Application"] == "BambuStudio-02.09.01.70"
            assert metadata["Title"] == "AutoForge_Model"
            assert json.loads(zf.read("Metadata/filament_sequence.json")) == {
                "plate_1": {"nozzle_sequence": [], "optimal_assignment": [], "sequence": []}
            }
            cfg = json.loads(zf.read("Metadata/project_settings.config"))
            assert cfg["printer_model"] == "Bambu Lab H3C"
            assert cfg["printer_settings_id"] == "Bambu Lab H3C 0.4 nozzle"
            assert cfg["print_compatible_printers"] == ["Bambu Lab H3C 0.4 nozzle"]
            assert cfg["printer_variant"] == "0.4"
            assert cfg["printer_structure"] == "corexy"
            assert cfg["printer_technology"] == "FFF"
            assert cfg["printable_area"] == ["0x0", "330x0", "330x320", "0x320"]
            assert cfg["printable_height"] == "325"
            assert cfg["print_settings_id"] == ""
            assert cfg["default_print_profile"] == "0.20mm Standard @BBL H3C"
            assert cfg["default_filament_profile"] == ["Bambu PLA Basic @BBL H3C"]
            assert cfg["sparse_infill_density"] == "100%"
            assert cfg["skeleton_infill_density"] == "100%"
            assert cfg["skin_infill_density"] == "100%"
            assert cfg["sparse_infill_pattern"] == "zig-zag"
            assert cfg["internal_solid_infill_pattern"] == "zig-zag"
            assert cfg["different_settings_to_system"] == [
                "skeleton_infill_density;skin_infill_density;sparse_infill_pattern",
                "",
                "",
                "",
                "",
                "",
            ]
            assert cfg["filament_colour"] == ["#112233", "#AABBCC", "#DDEEFF"]
            assert cfg["filament_type"] == ["PLA", "PLA", "PLA"]
            assert cfg["filament_vendor"] == ["Bambu Lab", "Bambu Lab", "Bambu Lab"]
            assert cfg["filament_settings_id"] == [
                "Bambu PLA Basic @BBL H3C",
                "Bambu PLA Matte @BBL H3C",
                "Bambu PLA Basic @BBL H3C",
            ]
            assert cfg["filament_ids"] == ["GFA00", "GFA01", "GFA00"]
            assert cfg["filament_density"] == ["1.26", "1.32", "1.26"]
            assert cfg["filament_cost"] == ["15.99", "15.99", "15.99"]
            assert cfg["temperature_vitrification"] == ["45", "45", "45"]
            assert cfg["fins_extrude_safe_temp"] == ["57", "57", "57"]
            assert cfg["impact_strength_z"] == ["13.8", "6.6", "13.8"]
            assert cfg["fan_min_speed"] == ["80", "100", "80"]
            assert cfg["filament_scarf_height"] == ["10%", "5%", "10%"]
            assert cfg["extruder_variant_list"] == [
                "Direct Drive Standard,Direct Drive High Flow,Direct Drive TPU285"
            ]
            assert cfg["nozzle_diameter"] == ["0.4"]
            assert cfg["filament_self_index"] == ["1", "1", "2", "2", "3", "3"]
            assert cfg["filament_extruder_variant"] == [
                "Direct Drive Standard",
                "Direct Drive High Flow",
            ] * 3
            for array_key in (
                "filament_colour",
                "filament_multi_colour",
                "filament_density",
                "filament_cost",
                "hot_plate_temp",
                "hot_plate_temp_initial_layer",
                "textured_plate_temp",
                "textured_plate_temp_initial_layer",
                "eng_plate_temp",
                "eng_plate_temp_initial_layer",
                "cool_plate_temp",
                "cool_plate_temp_initial_layer",
                "supertack_plate_temp",
                "supertack_plate_temp_initial_layer",
                "temperature_vitrification",
                "fins_extrude_safe_temp",
                "impact_strength_z",
                "fan_cooling_layer_time",
                "fan_max_speed",
                "fan_min_speed",
                "filament_scarf_height",
                "filament_ids",
                "filament_map",
            ):
                assert len(cfg[array_key]) == 3
            for array_key in (
                "nozzle_temperature",
                "nozzle_temperature_initial_layer",
                "filament_flow_ratio",
                "filament_wipe",
                "wipe",
                "filament_extruder_variant",
                "filament_self_index",
            ):
                assert len(cfg[array_key]) == 6
            verts = model_root.findall(".//m:vertex", ns)
            xs = [float(v.attrib["x"]) for v in verts]
            ys = [float(v.attrib["y"]) for v in verts]
            zs = [float(v.attrib["z"]) for v in verts]
            assert (min(xs) + max(xs)) / 2.0 == pytest.approx(165.0)
            assert (min(ys) + max(ys)) / 2.0 == pytest.approx(160.0)
            assert min(zs) == pytest.approx(0.0)
            gcodes = ET.fromstring(zf.read("Metadata/custom_gcode_per_layer.xml"))
            assert gcodes.tag == "custom_gcodes_per_layer"
            plate = gcodes.find("plate")
            assert plate is not None
            assert plate.find("plate_info").attrib["id"] == "1"
            assert plate.find("mode").attrib["value"] == "MultiAsSingle"
            layers = gcodes.findall(".//layer")
            assert len(layers) == 2
            assert layers[0].attrib["extruder"] == "2"
            assert layers[1].attrib["extruder"] == "3"


@pytest.mark.parametrize(
    "sliders",
    [
        [],
        [{"layer": 5, "filament_uuid": "u1", "enabled": False}],
        [{"layer": 5, "filament_uuid": "u1", "enabled": True}],
    ],
)
def test_export_bambu_3mf_handles_empty_or_single_slider(sliders):
    from autoforge.helpers.export_3mf import export_bambu_3mf

    project = {
        "settings": {"layer_height": 0.04, "background_height": 0.24, "background_color": "#123456"},
        "activeFilaments": [{"uuid": "u1", "color": "#123456", "filament_type": "PETG", "brand": "Prusa"}],
        "colorSliders": sliders,
    }
    buf = io.BytesIO()
    export_bambu_3mf(_sample_stl_bytes(), project, buf)
    buf.seek(0)
    with zipfile.ZipFile(buf) as zf:
        cfg = json.loads(zf.read("Metadata/project_settings.config"))
        assert len(cfg["filament_colour"]) == 1
        assert cfg["filament_colour"][0] == "#123456"
        gcodes = ET.fromstring(zf.read("Metadata/custom_gcode_per_layer.xml"))
        assert gcodes.findall(".//layer") == []


def test_export_bambu_3mf_closes_file_on_error(tmp_path, monkeypatch):
    from autoforge.helpers import export_3mf as export_3mf_mod
    from autoforge.helpers.export_3mf import export_bambu_3mf

    bad_out = tmp_path / "broken.3mf"
    with pytest.raises(ValueError):
        export_bambu_3mf(b"not an stl", {"settings": {}, "colorSliders": [], "activeFilaments": []}, str(bad_out))
    assert not bad_out.exists()

    # Even if an error occurs after opening the output file, the descriptor must be closed.
    monkeypatch.setattr(export_3mf_mod, "build_model_settings_xml", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom")))
    with pytest.raises(RuntimeError, match="boom"):
        export_bambu_3mf(_sample_stl_bytes(), {"settings": {}, "colorSliders": [], "activeFilaments": []}, str(bad_out))
    os.remove(bad_out)


def test_export_solution_3mf_logs_warning_without_raising(tmp_path, caplog):
    from autoforge.helpers.export_3mf import export_solution_3mf

    bad_stl = tmp_path / "bad.stl"
    bad_stl.write_bytes(b"corrupt")
    out_3mf = tmp_path / "final_model_bambu.3mf"

    res = export_solution_3mf(
        str(bad_stl),
        str(out_3mf),
        np.array([0, 1]),
        np.array([[2, 2], [2, 2]]),
        0.04,
        0.24,
        "#000000",
        [{"uuid": "a", "color": "#000000", "td": 1.0}],
    )
    assert res is None
    assert not out_3mf.exists()
    assert any("Failed to export Bambu .3mf" in r.message for r in caplog.records)


def test_outputs_3mf_endpoints_and_zip_bundle(client):
    job, job_dir = _completed_stl_job()

    # GET generates on-demand when final_model_bambu.3mf isn't pre-written
    r_get = client.get(f"/api/outputs/3mf/{job.job_id}")
    assert r_get.status_code == 200, r_get.text
    assert r_get.headers["content-type"].startswith("application/vnd.ms-package.3dmanufacturing-3dmodel+xml")
    with zipfile.ZipFile(io.BytesIO(r_get.content)) as zf:
        assert "3D/3dmodel.model" in zf.namelist()

    # POST applies project sliders & name
    project = {
        "settings": {"layer_height": 0.04, "background_height": 0.24, "background_color": "#000000"},
        "activeFilaments": [
            {"uuid": "f1", "color": "#000000", "filament_type": "PLA", "brand": "Bambu"},
            {"uuid": "f2", "color": "#FF0000", "filament_type": "PLA", "brand": "Bambu"},
        ],
        "colorSliders": [
            {"layer": 2, "filament_uuid": "f1", "enabled": True},
            {"layer": 6, "filament_uuid": "f2", "enabled": True},
        ],
    }
    r_post = client.post(f"/api/outputs/3mf/{job.job_id}", json={"name": "My Test", "project": project})
    assert r_post.status_code == 200, r_post.text
    assert 'filename="My-Test_model_bambu.3mf"' in r_post.headers["content-disposition"]
    assert r_post.headers["content-type"].startswith("application/vnd.ms-package.3dmanufacturing-3dmodel+xml")
    with zipfile.ZipFile(io.BytesIO(r_post.content)) as zf:
        cfg = json.loads(zf.read("Metadata/project_settings.config"))
        assert cfg["filament_colour"] == ["#000000", "#FF0000"]

    # Export ZIP includes the .3mf automatically
    r_zip = client.post(f"/api/outputs/export/{job.job_id}", json={"name": "My Test", "project": project})
    assert r_zip.status_code == 200, r_zip.text
    with zipfile.ZipFile(io.BytesIO(r_zip.content)) as zf:
        assert "My-Test_model_bambu.3mf" in zf.namelist()
        with zipfile.ZipFile(io.BytesIO(zf.read("My-Test_model_bambu.3mf"))) as inner:
            cfg = json.loads(inner.read("Metadata/project_settings.config"))
            assert cfg["filament_colour"] == ["#000000", "#FF0000"]


def test_outputs_3mf_rejects_traversal_and_missing_jobs(client):
    assert client.get("/api/outputs/3mf/nonexistent-job").status_code == 404
    assert client.post("/api/outputs/3mf/nonexistent-job", json={}).status_code == 404
    assert client.get("/api/outputs/3mf/%2e%2e").status_code in (400, 404)


def test_3mf_endpoint_leaves_no_temp_files(client, tmp_path, monkeypatch):
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))
    job, _ = _completed_stl_job()
    r = client.post(f"/api/outputs/3mf/{job.job_id}", json={"name": "car", "project": {"settings": {}}})
    assert r.status_code == 200
    assert os.listdir(tmp_path) == []


def _make_pipeline_result(out_dir):
    import argparse
    import torch
    from autoforge.Modules.Optimizer import FilamentOptimizer
    from autoforge.auto_forge import cli_defaults

    filaments = [
        {
            "color": "#ff0000", "td": 1.5, "name": "Acme - Red",
            "brand": "Acme", "short_name": "Red", "owned": True,
            "uuid": "u-red", "filament_type": "PLA",
        },
        {
            "color": "#00ff00", "td": 2.5, "name": "Acme - Green",
            "brand": "Acme", "short_name": "Green", "owned": False,
            "uuid": "u-green", "filament_type": "PETG",
        },
    ]
    defaults = cli_defaults()
    defaults.update(
        max_layers=8,
        layer_height=0.2,
        output_folder=str(out_dir),
        background_height=0.4,
        background_color="#000000",
        stl_output_size=20,
        flatforge=False,
        perform_pruning=False,
        csv_file="",
        json_file="",
        visualize=False,
    )
    args = argparse.Namespace(**defaults)
    torch.manual_seed(0)
    np.random.seed(0)
    device = torch.device("cpu")
    target = torch.rand(12, 12, 3) * 255
    pixel_height_logits_init = np.zeros((12, 12), dtype=np.float32)
    pixel_height_labels = np.zeros((12, 12), dtype=np.int32)
    global_logits_init = np.random.randn(8, 2).astype(np.float32)
    material_colors = torch.rand(2, 3)
    material_tds = torch.rand(2) * 4.0 + 1.0
    optimizer = FilamentOptimizer(
        args, target, pixel_height_logits_init, pixel_height_labels,
        global_logits_init, material_colors, material_tds, torch.zeros(3), device,
        perception_loss_module=None,
    )
    for _ in range(3):
        optimizer.step(record_best=True)
    return {
        "optimizer": optimizer,
        "args": args,
        "device": device,
        "material_colors_np": material_colors.numpy(),
        "material_TDs_np": material_tds.numpy(),
        "material_names": [f["name"] for f in filaments],
        "active_filaments": filaments,
        "alpha": None,
        "output_target": target,
        "focus_map_full": None,
        "focus_map_proc": None,
        "pixel_height_logits_init": pixel_height_logits_init,
        "pixel_height_labels": pixel_height_labels,
    }


def test_pipeline_export_results_writes_bambu_3mf_and_survives_3mf_failure(tmp_path, monkeypatch):
    from autoforge.webui.helpers.pipeline_runner import export_results

    out_ok = tmp_path / "ok"
    out_ok.mkdir()
    result = _make_pipeline_result(out_ok)
    outputs = export_results(result)
    assert outputs["bambu_3mf"] == os.path.join(str(out_ok), "final_model_bambu.3mf")
    assert os.path.exists(outputs["bambu_3mf"])
    with zipfile.ZipFile(outputs["bambu_3mf"]) as zf:
        assert "3D/3dmodel.model" in zf.namelist()
        assert "Metadata/project_settings.config" in zf.namelist()

    # Failure in 3MF generation must not break STL, swap_instructions, or project_file.hfp.
    import autoforge.helpers.export_3mf as export_3mf_mod

    monkeypatch.setattr(export_3mf_mod, "export_bambu_3mf", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("3mf boom")))
    out_fail = tmp_path / "fail"
    out_fail.mkdir()
    result_fail = _make_pipeline_result(out_fail)
    outputs_fail = export_results(result_fail)
    assert outputs_fail["bambu_3mf"] is None
    assert not (out_fail / "final_model_bambu.3mf").exists()
    assert os.path.exists(outputs_fail["stl"])
    assert os.path.exists(outputs_fail["swap_instructions"])
    assert os.path.exists(outputs_fail["project_file"])


def test_parse_swap_instructions_and_3mf_matches_exact_sequence(tmp_path):
    from autoforge.helpers.export_3mf import export_bambu_3mf, parse_swap_instructions

    swap_text = (
        "Print at 100% infill with a layer height of 0.04mm with a base layer of 0.24mm "
        "using background filament Bambu - Silver.\n\n"
        "Start with Bambu - Silver, with a layer height of 0.24mm for the first layer.\n"
        "At layer #2 (0.28mm) swap to Bambu - Gray\n"
        "At layer #3 (0.32mm) swap to Polymaker - Red\n"
        "At layer #4 (0.36mm) swap to Bambu - Gray\n"
        "At layer #5 (0.40mm) swap to Bambu - Green\n"
    )
    base_name, base_height, swaps = parse_swap_instructions(swap_text)
    assert base_name == "Bambu - Silver"
    assert base_height == pytest.approx(0.24)
    assert swaps == [
        {"layer": 2, "top_z": pytest.approx(0.28), "filament_name": "Bambu - Gray"},
        {"layer": 3, "top_z": pytest.approx(0.32), "filament_name": "Polymaker - Red"},
        {"layer": 4, "top_z": pytest.approx(0.36), "filament_name": "Bambu - Gray"},
        {"layer": 5, "top_z": pytest.approx(0.40), "filament_name": "Bambu - Green"},
    ]

    project = {
        "settings": {"layer_height": 0.04, "background_height": 0.24, "background_color": "#C0C0C0"},
        "activeFilaments": [
            {"uuid": "u-green", "brand": "Bambu", "name": "Green", "color": "#00ff00"},
            {"uuid": "u-silver", "brand": "Bambu", "name": "Silver", "color": "#c0c0c0"},
            {"uuid": "u-gray", "brand": "Bambu", "name": "Gray", "color": "#808080"},
            {"uuid": "u-red", "brand": "Polymaker", "name": "Red", "color": "#ff0000"},
        ],
        # Intentionally omitting Silver from sliders (base layer isn't in sliders).
        "colorSliders": [
            {"layer": 2, "filament_uuid": "u-gray", "enabled": True},
        ],
    }

    swap_file = tmp_path / "swap_instructions.txt"
    swap_file.write_text(swap_text, encoding="utf-8")

    for swap_arg in (swap_text, str(swap_file)):
        buf = io.BytesIO()
        export_bambu_3mf(_sample_stl_bytes(), project, buf, swap_instructions_text=swap_arg)
        buf.seek(0)
        with zipfile.ZipFile(buf) as zf:
            cfg = json.loads(zf.read("Metadata/project_settings.config"))
            assert cfg["print_settings_id"] == ""
            assert cfg["initial_layer_print_height"] == "0.2400"
            assert cfg["sparse_infill_density"] == "100%"
            assert cfg["sparse_infill_pattern"] == "zig-zag"
            assert cfg["filament_settings_id"] == ["Bambu PLA Basic @BBL H3C"] * 4
            assert cfg["filament_colour"] == ["#C0C0C0", "#808080", "#FF0000", "#00FF00"]

            gcodes = ET.fromstring(zf.read("Metadata/custom_gcode_per_layer.xml"))
            layers = gcodes.findall(".//layer")
            assert [(l.attrib["top_z"], l.attrib["extruder"], l.attrib["color"]) for l in layers] == [
                ("0.2800", "2", "#808080"),
                ("0.3200", "3", "#FF0000"),
                ("0.3600", "2", "#808080"),
                ("0.4000", "4", "#00FF00"),
            ]


def test_outputs_3mf_uses_swap_instructions_from_job_dir(client):
    job, job_dir = _completed_stl_job()
    with open(os.path.join(job_dir, "swap_instructions.txt"), "w", encoding="utf-8") as f:
        f.write(
            "Start with Bambu - Silver, with a layer height of 0.24mm for the first layer.\n"
            "At layer #2 (0.28mm) swap to Bambu - Gray\n"
            "At layer #3 (0.32mm) swap to Polymaker - Red\n"
        )
    project = {
        "settings": {"layer_height": 0.04, "background_height": 0.24, "background_color": "#C0C0C0"},
        "activeFilaments": [
            {"uuid": "f1", "brand": "Bambu", "name": "Silver", "color": "#c0c0c0"},
            {"uuid": "f2", "brand": "Bambu", "name": "Gray", "color": "#808080"},
            {"uuid": "f3", "brand": "Polymaker", "name": "Red", "color": "#ff0000"},
        ],
        "colorSliders": [{"layer": 2, "filament_uuid": "f2", "enabled": True}],
    }
    r_post = client.post(f"/api/outputs/3mf/{job.job_id}", json={"name": "SwapSync", "project": project})
    assert r_post.status_code == 200, r_post.text
    with zipfile.ZipFile(io.BytesIO(r_post.content)) as zf:
        cfg = json.loads(zf.read("Metadata/project_settings.config"))
        assert cfg["filament_colour"] == ["#C0C0C0", "#808080", "#FF0000"]
        layers = ET.fromstring(zf.read("Metadata/custom_gcode_per_layer.xml")).findall(".//layer")
        assert [(l.attrib["top_z"], l.attrib["extruder"], l.attrib["color"]) for l in layers] == [
            ("0.2800", "2", "#808080"),
            ("0.3200", "3", "#FF0000"),
        ]


def test_build_project_settings_scales_all_8_slots_with_matte_and_basic():
    from autoforge.helpers.export_3mf import export_bambu_3mf

    filaments = [
        {"uuid": f"u{i}", "brand": "Bambu Lab", "name": f"{'Matte' if i % 2 else 'Basic'} {i}", "color": f"#{i:02X}1122"}
        for i in range(8)
    ]
    swap_text = "\n".join(
        ["Start with Bambu Lab - Basic 0, with a layer height of 0.24mm for the first layer."]
        + [
            f"At layer #{i + 1} ({0.24 + i * 0.04:.2f}mm) swap to Bambu Lab - {'Matte' if i % 2 else 'Basic'} {i}"
            for i in range(1, 8)
        ]
    )
    buf = io.BytesIO()
    export_bambu_3mf(_sample_stl_bytes(), {"settings": {"layer_height": 0.04}, "activeFilaments": filaments, "colorSliders": []}, buf, swap_instructions_text=swap_text)
    buf.seek(0)
    with zipfile.ZipFile(buf) as zf:
        cfg = json.loads(zf.read("Metadata/project_settings.config"))
        assert cfg["filament_settings_id"] == [
            "Bambu PLA Matte @BBL H3C" if i % 2 else "Bambu PLA Basic @BBL H3C" for i in range(8)
        ]
        assert cfg["filament_vendor"] == ["Bambu Lab"] * 8
        assert cfg["filament_type"] == ["PLA"] * 8
        assert cfg["filament_ids"] == ["GFA01" if i % 2 else "GFA00" for i in range(8)]
        assert cfg["filament_density"] == ["1.32" if i % 2 else "1.26" for i in range(8)]
        assert cfg["fan_min_speed"] == ["100" if i % 2 else "80" for i in range(8)]
        assert cfg["filament_scarf_height"] == ["5%" if i % 2 else "10%" for i in range(8)]
        assert cfg["impact_strength_z"] == ["6.6" if i % 2 else "13.8" for i in range(8)]
        assert cfg["filament_map"] == [str(i + 1) for i in range(8)]
        for key in (
            "filament_colour", "filament_density", "filament_cost",
            "hot_plate_temp", "temperature_vitrification", "fins_extrude_safe_temp",
            "impact_strength_z", "fan_min_speed", "filament_scarf_height",
        ):
            assert len(cfg[key]) == 8
        for key in (
            "nozzle_temperature", "nozzle_temperature_initial_layer",
            "filament_flow_ratio", "filament_wipe", "wipe",
            "filament_extruder_variant", "filament_self_index",
        ):
            assert len(cfg[key]) == 16


def test_result_store_preserves_matte_brand_and_type_for_3mf_export(tmp_path):
    from autoforge.helpers.export_3mf import export_solution_3mf
    from autoforge.webui.helpers.result_store import solution_from_sliders

    heights = np.array([[1, 2], [2, 3]], dtype=np.int64)
    active_filaments = [
        {
            "uuid": "f-basic",
            "color": "#000000",
            "td": 1.0,
            "brand": "BambuLab Basic",
            "name": "Black",
            "short_name": "Black",
            "filament_type": "PLA",
        },
        {
            "uuid": "f-matte",
            "color": "#AABBCC",
            "td": 2.0,
            "brand": "BambuLab Matte",
            "name": "Ice Blue",
            "short_name": "Ice Blue",
            "filament_type": "PLA Matte",
        },
    ]
    sliders = [
        {"layer": 1, "filament_uuid": "f-basic", "enabled": True},
        {"layer": 3, "filament_uuid": "f-matte", "enabled": True},
    ]
    sol = solution_from_sliders(
        heights,
        sliders,
        active_filaments,
        {"layer_height": 0.04, "background_height": 0.24, "max_layers": 3},
        background_color="#000000",
    )
    materials = sol["meta"]["materials"]
    assert materials[0]["brand"] == "BambuLab Basic"
    assert materials[1]["brand"] == "BambuLab Matte"
    assert materials[1]["filament_type"] == "PLA Matte"

    stl_path = tmp_path / "model.stl"
    stl_path.write_bytes(_sample_stl_bytes())
    out_3mf = tmp_path / "out.3mf"
    res = export_solution_3mf(
        str(stl_path),
        str(out_3mf),
        sol["stack"],
        sol["heights"],
        0.04,
        0.24,
        "#000000",
        materials,
        background_material_index=sol["meta"]["background_material_index"],
    )
    assert res == str(out_3mf)
    with zipfile.ZipFile(out_3mf) as zf:
        cfg = json.loads(zf.read("Metadata/project_settings.config"))
        assert cfg["filament_settings_id"] == [
            "Bambu PLA Basic @BBL H3C",
            "Bambu PLA Matte @BBL H3C",
        ]
        assert cfg["filament_ids"] == ["GFA00", "GFA01"]
        assert cfg["filament_density"] == ["1.26", "1.32"]
        assert cfg["fan_min_speed"] == ["80", "100"]
