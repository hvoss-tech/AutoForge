"""Regression tests for the 2026-09-26 audit (core library + CLI).

Each test fails on the code before its fix."""

import argparse
import csv
import json
import math
import sys

import cv2
import numpy as np
import pytest
import torch
import trimesh

from autoforge.Helper.OutputHelper import (
    generate_flatforge_stls,
    generate_project_file,
    model_size_mm,
)
from autoforge.Helper.PruningHelper import (
    _eval_candidates_batch,
    remove_layer_from_solution,
)


# ---------------------------------------------------------------------------
# Layer pruning: removing layer i must only lower pixels that contain it
# ---------------------------------------------------------------------------


class _NoOffset:
    def _apply_height_offset(self, logits, offsets):
        return logits


def _params_for_heights(heights, max_layers):
    z = torch.tensor(heights, dtype=torch.float32)
    ratio = (z / max_layers).clamp(1e-6, 1 - 1e-6)
    return {
        "global_logits": torch.eye(max_layers),
        "pixel_height_logits": torch.log(ratio) - torch.log1p(-ratio),
        "height_offsets": torch.zeros(1),
    }


def _heights_after(params, max_layers):
    return torch.round(
        max_layers * torch.sigmoid(params["pixel_height_logits"])
    ).int().tolist()


@pytest.mark.parametrize("removed", [0, 2, 4])
def test_removing_a_layer_only_lowers_pixels_that_contain_it(removed):
    # A pixel of height z prints layers 0..z-1: it contains layer `removed`
    # only when z > removed.
    heights = [[0.0, 1.0, 2.0, 3.0, 4.0, 5.0]]
    params = _params_for_heights(heights, 5)
    new_params, new_layers = remove_layer_from_solution(
        _NoOffset(), params, removed, 0.04, 5
    )
    assert new_layers == 4
    expected = [[z - 1 if z > removed else z for z in map(int, heights[0])]]
    assert _heights_after(new_params, new_layers) == expected


def test_removing_an_unused_top_layer_leaves_every_pixel_alone():
    # Nothing reaches layer 5 (heights top out at 5 = layers 0..4 printed),
    # so removing it must change no pixel. The old `>=` shaved the tallest.
    heights = [[1.0, 3.0, 5.0], [0.0, 2.0, 4.0]]
    params = _params_for_heights(heights, 6)
    new_params, new_layers = remove_layer_from_solution(
        _NoOffset(), params, 5, 0.04, 6
    )
    assert _heights_after(new_params, new_layers) == [[1, 3, 5], [0, 2, 4]]


# ---------------------------------------------------------------------------
# .hfp project file
# ---------------------------------------------------------------------------


def _write_materials_csv(path, rows):
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Brand", "Name", "Color", "Transmissivity", "Type", "Owned", "Uuid"])
        writer.writerows(rows)


def _strict_load(path):
    def reject(constant):
        raise ValueError(f"not JSON: {constant}")

    with open(path) as f:
        return json.load(f, parse_constant=reject)


def _project_args(csv_path, **overrides):
    values = dict(
        csv_file=str(csv_path), json_file="", background_height=0.24,
        layer_height=0.04, background_color="#000000", max_layers=10,
    )
    values.update(overrides)
    return argparse.Namespace(**values)


def test_project_file_is_strict_json_when_csv_fields_are_blank(tmp_path):
    # The webui writes its own materials.csv with an empty Brand/Uuid for
    # filaments that have none; pandas reads those as NaN and json.dump wrote
    # a bare NaN, which HueForge cannot parse.
    csv_path = tmp_path / "m.csv"
    _write_materials_csv(csv_path, [
        ["", "Red", "#ff0000", 2.5, "", False, ""],
        ["Acme", "Blue", "#0000ff", 3, "PLA", True, "u2"],
    ])
    out = tmp_path / "p.hfp"
    generate_project_file(
        str(out), _project_args(csv_path, background_material_index=0),
        np.array([0, 0, 1, 1, 1, 0, 0, 0, 0, 0]), np.full((4, 4), 5),
        150.0, 100.0, "final_model.stl", str(csv_path),
    )
    data = _strict_load(out)  # raises on NaN
    red = [f for f in data["filament_set"] if f["Name"] == "Red"]
    assert red, data["filament_set"]
    for entry in red:
        assert isinstance(entry["Brand"], str)
        assert entry["Type"] == "PLA"
        assert isinstance(entry["uuid"], str) and entry["uuid"]
        assert entry["Transmissivity"] == 2.5
    blue = next(f for f in data["filament_set"] if f["Name"] == "Blue")
    assert blue["Brand"] == "Acme" and blue["uuid"] == "u2" and blue["Transmissivity"] == 3


def test_model_size_mm_matches_the_stl_scale():
    # generate_stl spans maximum_x_y_size over max(W, H) - 1 pixel steps.
    assert model_size_mm(751, 501, 150) == pytest.approx((150.0, 100.0))
    assert model_size_mm(501, 751, 150) == pytest.approx((100.0, 150.0))
    assert model_size_mm(1, 1, 150) == (0.0, 0.0)


def test_cli_project_file_records_millimetres_not_pixels(tmp_path, monkeypatch):
    width_px, height_px = 60, 40
    img = np.zeros((height_px, width_px, 3), np.uint8)
    img[:, : width_px // 2] = 255
    cv2.imwrite(str(tmp_path / "in.png"), img)
    csv_path = tmp_path / "m.csv"
    _write_materials_csv(csv_path, [
        ["T", "Black", "#000000", 0.6, "PLA", True, "b"],
        ["T", "White", "#ffffff", 5, "PLA", True, "w"],
    ])
    out = tmp_path / "out"
    _run_cli(monkeypatch, tmp_path / "in.png", csv_path, out, "--stl_output_size", "10")
    data = _strict_load(out / "project_file.hfp")
    # The longer side is the requested 10 mm (it was the pixel count, 50).
    assert max(data["width_in_mm"], data["height_in_mm"]) == pytest.approx(10.0)
    assert data["width_in_mm"] > data["height_in_mm"]


# ---------------------------------------------------------------------------
# FlatForge
# ---------------------------------------------------------------------------


def _flatforge(tmp_path, disc_global, heights, names=("A", "B"), tds=(1.0, 5.0)):
    colors = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]][: len(names)])
    return generate_flatforge_stls(
        np.asarray(disc_global), np.asarray(heights), colors, list(names),
        np.asarray(tds, dtype=float), 0.04, 0.24, "#000000", 10.0,
        str(tmp_path), cap_layers=0,
    )


def test_flatforge_material_in_two_bands_does_not_fill_the_band_between(tmp_path):
    # A at layers 0-1, B at 2-3, A again at 4-5. A's solid used to run from
    # layer 0 to 6, straight through B.
    files = _flatforge(tmp_path, [0, 0, 1, 1, 0, 0], np.full((6, 6), 6))
    a_file = next(f for f in files if "/A_" in f)
    b_file = next(f for f in files if "/B_" in f)
    lh, base = 0.04, 0.24
    b_lo, b_hi = base + 2 * lh, base + 4 * lh

    a = trimesh.load(a_file)
    z = a.vertices[:, 2]
    assert not np.any((z > b_lo + 1e-6) & (z < b_hi - 1e-6)), "A has geometry inside B's band"
    area = 10.0 * 10.0
    assert a.volume == pytest.approx(area * 4 * lh, rel=1e-4)

    b = trimesh.load(b_file)
    assert b.volume == pytest.approx(area * 2 * lh, rel=1e-4)
    assert b.bounds[0, 2] == pytest.approx(b_lo, abs=1e-5)
    assert b.bounds[1, 2] == pytest.approx(b_hi, abs=1e-5)


def test_flatforge_band_stops_at_each_pixels_own_height(tmp_path):
    # Left half is 3 layers tall, right half 6: A's upper band [4, 6) only
    # exists on the right, and B [2, 4) is only one layer thick on the left.
    heights = np.full((6, 12), 6)
    heights[:, :6] = 3
    files = _flatforge(tmp_path, [0, 0, 1, 1, 0, 0], heights)
    b = trimesh.load(next(f for f in files if "/B_" in f))
    assert b.bounds[1, 2] == pytest.approx(0.24 + 4 * 0.04, abs=1e-5)
    left = b.vertices[b.vertices[:, 0] < 10.0 * 4 / 11]
    assert left[:, 2].max() == pytest.approx(0.24 + 3 * 0.04, abs=1e-5)


# ---------------------------------------------------------------------------
# --pruning_batch_size is a real batch size, with identical results
# ---------------------------------------------------------------------------


class _BatchOptimizer:
    def __init__(self):
        g = torch.Generator().manual_seed(0)
        self.material_colors = torch.rand(4, 3, generator=g)
        self.material_TDs = torch.tensor([1.0, 2.0, 4.0, 8.0])
        self.background = torch.zeros(3)
        self.target = torch.rand(8, 8, 3, generator=g) * 255
        self.focus_map = None
        self.alpha = None
        self.best_seed = 7
        self.vis_tau = 0.01


def test_pruning_batch_size_chunks_without_changing_the_result(monkeypatch):
    import autoforge.Helper.PruningHelper as ph

    opt = _BatchOptimizer()
    eff = torch.rand(6, 8, 8) * 0.04
    candidates = [torch.randint(0, 4, (6,), generator=torch.Generator().manual_seed(i)) for i in range(7)]

    batch_sizes = []
    real_select = ph._material_select_batched

    def spy(gl, *args, **kwargs):
        batch_sizes.append(gl.shape[0])
        return real_select(gl, *args, **kwargs)

    monkeypatch.setattr(ph, "_material_select_batched", spy)
    loss_all, dg_all = _eval_candidates_batch(opt, candidates, eff_thick=eff)
    assert batch_sizes == [7]
    batch_sizes.clear()
    loss_3, dg_3 = _eval_candidates_batch(opt, candidates, eff_thick=eff, batch_size=3)
    assert batch_sizes == [3, 3, 1]
    assert loss_3 == pytest.approx(loss_all)
    assert torch.equal(dg_3, dg_all)


# ---------------------------------------------------------------------------
# CLI end to end: image formats, base height, --best_of, seed-search baseline
# ---------------------------------------------------------------------------


def _run_cli(monkeypatch, image, csv_path, out, *extra):
    from autoforge import auto_forge

    argv = [
        "autoforge", "--input_image", str(image), "--csv_file", str(csv_path),
        "--output_folder", str(out), "--iterations", "20", "--stl_output_size", "10",
        "--max_layers", "12", "--num_init_rounds", "1", "--num_init_threads", "1",
        "--no-visualize", "--random_seed", "3", "--device", "cpu", "--no-cuda_graph",
        *extra,
    ]
    monkeypatch.setattr(sys, "argv", argv)
    auto_forge.main()


@pytest.fixture
def cli_materials(tmp_path):
    csv_path = tmp_path / "mats.csv"
    _write_materials_csv(csv_path, [
        ["T", "Black", "#000000", 0.6, "PLA", True, "b"],
        ["T", "White", "#ffffff", 5, "PLA", True, "w"],
        ["T", "Grey", "#808080", 2, "PLA", True, "g"],
    ])
    return csv_path


def _pattern(dtype, maxval):
    img = np.zeros((40, 60), dtype)
    img[:, :30] = maxval
    img[10:30, 40:50] = maxval // 2
    return img


@pytest.mark.parametrize(
    "name,image",
    [
        ("gray8.png", _pattern(np.uint8, 255)),
        ("gray16.png", _pattern(np.uint16, 65535)),
    ],
)
def test_cli_accepts_grayscale_and_16bit_images(tmp_path, monkeypatch, cli_materials, name, image):
    path = tmp_path / name
    cv2.imwrite(str(path), image)
    out = tmp_path / "out"
    _run_cli(monkeypatch, path, cli_materials, out, "--no-perform_pruning")
    preview = cv2.imread(str(out / "final_model.png"))
    assert preview is not None
    # The 16-bit white half must come out light, not as an unmatched target.
    left = preview[:, : preview.shape[1] // 4].mean()
    assert left > 128


def test_cli_16bit_color_image_matches_its_8bit_version(tmp_path, monkeypatch, cli_materials):
    # A 16-bit RGB PNG didn't crash, but was optimized against values up to
    # 65535 - a target no filament stack can reach, so the loss exploded.
    rgb8 = np.dstack([_pattern(np.uint8, 255)] * 3)
    cv2.imwrite(str(tmp_path / "rgb8.png"), rgb8)
    cv2.imwrite(str(tmp_path / "rgb16.png"), rgb8.astype(np.uint16) * 257)
    losses = []
    for name in ("rgb8.png", "rgb16.png"):
        out = tmp_path / name.replace(".png", "")
        _run_cli(monkeypatch, tmp_path / name, cli_materials, out, "--no-perform_pruning")
        losses.append(float((out / "final_loss.txt").read_text()))
    assert losses[1] == pytest.approx(losses[0], rel=0.05)


def test_cli_16bit_priority_mask_is_scaled_to_0_1(tmp_path, monkeypatch):
    from autoforge.auto_forge import _load_priority_mask

    mask = np.zeros((20, 20), np.uint16)
    mask[:, 10:] = 65535
    cv2.imwrite(str(tmp_path / "mask.png"), mask)
    args = argparse.Namespace(
        priority_mask=str(tmp_path / "mask.png"), output_folder=str(tmp_path),
        priority_mask_strength=10.0,
    )
    focus = _load_priority_mask(args, np.zeros((20, 20, 3), np.uint8), torch.device("cpu"))
    assert float(focus.max()) == pytest.approx(1.0, abs=1e-3)
    assert float(focus.min()) == pytest.approx(0.0, abs=1e-3)


@pytest.mark.parametrize("background_height,layer_height", [(0.28, 0.04), (0.6, 0.2), (0.24, 0.04)])
def test_basic_check_accepts_whole_layer_multiples(tmp_path, background_height, layer_height):
    from autoforge.Helper.OtherHelper import perform_basic_check

    image = tmp_path / "in.png"
    image.write_bytes(b"x")
    args = argparse.Namespace(
        background_height=background_height, layer_height=layer_height,
        input_image=str(image), csv_file="", json_file="", priority_mask="",
        priority_mask_strength=10.0,
    )
    perform_basic_check(args)  # used to sys.exit(1) for 0.28/0.04 and 0.6/0.2


def test_basic_check_still_rejects_a_partial_layer(tmp_path):
    from autoforge.Helper.OtherHelper import perform_basic_check

    image = tmp_path / "in.png"
    image.write_bytes(b"x")
    args = argparse.Namespace(
        background_height=0.25, layer_height=0.04, input_image=str(image),
        csv_file="", json_file="", priority_mask="", priority_mask_strength=10.0,
    )
    with pytest.raises(SystemExit):
        perform_basic_check(args)


def test_best_of_runs_all_start_from_the_requested_settings(tmp_path, monkeypatch, cli_materials):
    from autoforge import auto_forge

    cv2.imwrite(str(tmp_path / "in.png"), _pattern(np.uint8, 255))
    seen = []
    real_start = auto_forge.start

    def spy(args):
        seen.append((args.max_layers, args.output_folder))
        return real_start(args)

    monkeypatch.setattr(auto_forge, "start", spy)
    _run_cli(
        monkeypatch, tmp_path / "in.png", cli_materials, tmp_path / "out",
        "--best_of", "2", "--pruning_max_layer", "6", "--background_height", "0.28",
    )
    # Pruning brings run 1 down to 6 layers; run 2 used to inherit that.
    assert [layers for layers, _ in seen] == [12, 12]
    assert (tmp_path / "out" / "final_model.stl").exists()


def test_best_of_reports_when_every_run_failed(tmp_path, monkeypatch, cli_materials):
    from autoforge import auto_forge

    cv2.imwrite(str(tmp_path / "in.png"), _pattern(np.uint8, 255))

    def boom(args):
        raise RuntimeError("no luck")

    monkeypatch.setattr(auto_forge, "start", boom)
    with pytest.raises(SystemExit) as exc:
        _run_cli(monkeypatch, tmp_path / "in.png", cli_materials, tmp_path / "out", "--best_of", "2")
    assert exc.value.code == 1


def test_no_pruning_seed_search_starts_from_the_full_resolution_loss(tmp_path, monkeypatch, cli_materials):
    from autoforge.Modules.Optimizer import FilamentOptimizer

    cv2.imwrite(str(tmp_path / "in.png"), _pattern(np.uint8, 255))
    calls = []
    real_search = FilamentOptimizer.rng_seed_search

    def spy(self, start_loss, num_seeds, *args, **kwargs):
        calls.append((start_loss, self.solution_loss(), self.best_discrete_loss))
        return real_search(self, start_loss, num_seeds, *args, **kwargs)

    monkeypatch.setattr(FilamentOptimizer, "rng_seed_search", spy)
    _run_cli(monkeypatch, tmp_path / "in.png", cli_materials, tmp_path / "out", "--no-perform_pruning")
    assert len(calls) == 1
    start_loss, full_res_loss, training_loss = calls[0]
    assert start_loss == pytest.approx(full_res_loss)
    # The two resolutions really do disagree here, so the check is meaningful.
    assert not math.isclose(training_loss, full_res_loss, rel_tol=1e-6)
