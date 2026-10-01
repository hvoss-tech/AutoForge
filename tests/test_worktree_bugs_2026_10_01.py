"""Regression tests for the bugs the agent worktrees found (2026-10-01
review; numbers refer to that list)."""

import inspect
import os
import random

import cv2
import numpy as np
import pytest
import torch


def test_3_imread_unchanged_applies_exif_orientation(tmp_path):
    """#3: a phone photo stored sideways with EXIF orientation 6 is read
    upright, like IMREAD_COLOR and every viewer shows it."""
    from PIL import Image

    from autoforge.Helper.ImageHelper import imread

    stored = np.zeros((16, 32, 3), np.uint8)  # 16 rows, 32 columns as stored
    stored[:8, :8] = (255, 0, 0)  # red block top-left (RGB)
    img = Image.fromarray(stored)
    exif = img.getexif()
    exif[0x0112] = 6  # rotate 90 degrees clockwise to display
    path = str(tmp_path / "phone.jpg")
    img.save(path, exif=exif, quality=100)

    out = imread(path, cv2.IMREAD_UNCHANGED)
    assert out.shape[:2] == (32, 16)
    # Displayed: the stored top-left block ends up top-right.
    assert out[3, 12, 2] > 200 and out[3, 3, 2] < 60  # BGR: red channel last
    assert imread(path, cv2.IMREAD_COLOR).shape[:2] == (32, 16)  # OpenCV's own rule agrees


def test_4_resize_keeps_at_least_one_pixel():
    """#4: a very long, thin image doesn't round its short side to 0."""
    from autoforge.Helper.ImageHelper import resize_image

    out = resize_image(np.zeros((1, 400, 3), np.uint8), 100)
    assert out.shape[:2] == (1, 100)


def test_5_set_seed_seeds_pythons_random():
    """#5: --random_seed also makes Python's random reproducible."""
    from argparse import Namespace

    from autoforge.Helper.OtherHelper import set_seed

    set_seed(Namespace(random_seed=7))
    a = [random.random() for _ in range(3)]
    set_seed(Namespace(random_seed=7))
    assert a == [random.random() for _ in range(3)]


def test_6_no_plain_cv2_imwrite_in_the_pipeline():
    """#6: every output image is written through the unicode-safe helper."""
    from autoforge import auto_forge
    from autoforge.webui.helpers import pipeline_runner, slider_render

    for module in (auto_forge, pipeline_runner, slider_render):
        source = inspect.getsource(module)
        assert "cv2.imwrite(" not in source.replace("# ImageHelper.imwrite: cv2.imwrite", ""), module.__name__


def test_8_mask_brush_outside_the_window_does_not_raise():
    """#8: dragging the brush past the image edge paints what is inside and
    never raises."""
    from autoforge.priority_mask_editor import PriorityMaskEditor

    editor = PriorityMaskEditor(np.zeros((20, 30, 3), np.uint8), brush_radius=5)
    editor._apply_brush(-50, -50)  # far outside: nothing to paint
    editor._apply_brush(-10, 10)  # left of the image: x1 = -4 sliced 26 columns
    editor._apply_brush(33, 10)  # partly outside on the right
    editor._apply_brush(32, -3)
    assert editor.mask.shape == (20, 30)
    assert editor.mask[10, 29] > 0
    assert editor.mask[:, :20].max() == 0


def test_9_mask_defaults_to_png():
    """#9: the default mask file is a lossless PNG, whatever the input is."""
    from autoforge.priority_mask_editor import _default_output_path

    assert _default_output_path(os.path.join("a", "photo.jpg")) == os.path.join("a", "photo_mask.png")
    assert _default_output_path("photo") == "photo_mask.png"


# --- #10-#14: STL mesh and FlatForge file names (agent-a55ede) ---
import struct

from autoforge.Helper.HeightfieldMesh import heightfield_mesh
from autoforge.Helper.OutputHelper import _file_safe, generate_flatforge_stls, generate_stl


def _triangle_count(path):
    with open(path, "rb") as f:
        data = f.read()
    n = struct.unpack("<I", data[80:84])[0]
    assert len(data) == 84 + 50 * n
    return n


@pytest.mark.parametrize("shape", [(1, 6), (6, 1), (1, 1)])
def test_10_generate_stl_one_pixel_wide_map_writes_an_empty_mesh(tmp_path, shape):
    # used to crash (IndexError / negative dimensions / division by zero)
    out = tmp_path / "m.stl"
    generate_stl(np.full(shape, 0.08, np.float32), str(out), 0.24, 50.0)
    assert _triangle_count(out) == 0


def test_11_generate_stl_fully_transparent_alpha_writes_an_empty_mesh(tmp_path):
    # used to crash with IndexError when no cell was valid
    out = tmp_path / "m.stl"
    hm = np.random.default_rng(0).integers(0, 5, (8, 9)).astype(np.float32) * 0.04
    generate_stl(hm, str(out), 0.24, 50.0, alpha_mask=np.zeros((8, 9), np.uint8))
    assert _triangle_count(out) == 0


def test_12_generate_stl_accepts_a_single_channel_alpha_mask():
    # (H, W, 1) - the shape the image loaders build - used to fail to broadcast
    hm = np.random.default_rng(1).integers(0, 5, (8, 9)).astype(np.float32) * 0.04
    alpha = np.full((8, 9), 255, np.uint8)
    alpha[2:5, 3:6] = 0
    v2, f2 = heightfield_mesh(hm, 0.24, 50.0, alpha)
    v3, f3 = heightfield_mesh(hm, 0.24, 50.0, alpha[..., None])
    assert np.array_equal(v2, v3) and np.array_equal(f2, f3)
    assert len(f2) > 0


def test_13_zero_background_height_leaves_no_zero_area_wall_triangles():
    # with background_height 0 a height-0 outline vertex coincides with its
    # bottom twin: the walls used to contain zero-area triangles there
    hm = np.random.default_rng(2).integers(0, 5, (8, 9)).astype(np.float32) * 0.04
    hm[0, :] = 0.0
    hm[:, 0] = 0.0
    v, f = heightfield_mesh(hm, 0.0, 50.0)
    t = v[f].astype(np.float64)
    area = 0.5 * np.linalg.norm(np.cross(t[:, 1] - t[:, 0], t[:, 2] - t[:, 0]), axis=1)
    assert (area > 0).all()


def test_14_flatforge_file_names_are_valid_on_windows(tmp_path):
    assert _file_safe('PLA+ Silk: Gold/Red\\x*?"<>|') == "PLA+_Silk-_Gold-Red-x------"
    disc_global = np.array([0, 1, 1])
    heights = np.array([[1, 2, 3], [3, 2, 1], [2, 2, 2]])
    files = generate_flatforge_stls(
        disc_global, heights, np.array([[1.0, 0, 0], [0, 0, 1.0], [1, 1, 1]]),
        ["Red: Matte", "Blue*", "Clear?"], np.array([1.0, 2.0, 9.0]),
        0.04, 0.24, "#000000", 30.0, str(tmp_path),
    )
    assert files
    for path in files:
        name = os.path.basename(path)
        assert not any(ch in name for ch in ':*?"<>|\\'), name
        assert os.path.exists(path)


# --- #19, #20: height-map initialisers (agent-acf179) ---


def test_19_lone_centroid_gives_finite_weights():
    """#19: a single over-cluster centroid no longer turns every k-means
    weight into NaN (inf / inf)."""
    from autoforge.Helper.Heightmaps.ChristofidesHeightMap import _compute_distinctiveness
    from autoforge.Helper.Heightmaps.FastTSPHeightMap import _refine_clusters

    assert _compute_distinctiveness(np.array([[10.0, 20.0, 30.0]])).tolist() == [0.0]
    pixels = np.tile(np.array([[50.0, 10.0, 10.0]]), (6, 1))
    centroids, labels = _refine_clusters(pixels, 2, 3, pixels[:1], np.zeros(6, np.int64), final_k=1)
    assert np.isfinite(centroids).all() and labels.shape == (2, 3)


def test_20_depth_init_on_an_image_with_fewer_pixels_than_layers(monkeypatch):
    """#20: the depth initializer no longer asks k-means for more clusters
    than the image has pixels."""
    import sys
    import types

    from autoforge.Helper.Heightmaps import DepthEstimateHeightMap as dem

    fake = types.ModuleType("transformers")
    fake.pipeline = lambda *a, **k: (lambda image: {"depth": np.arange(4, dtype=np.float32).reshape(2, 2)})
    monkeypatch.setitem(sys.modules, "transformers", fake)
    target = np.array([[[255, 0, 0], [0, 255, 0]], [[0, 0, 255], [255, 255, 255]]], np.float32)
    logits, labels = dem.init_height_map_depth_color_adjusted(target, max_layers=75, random_seed=0)
    assert logits.shape == (2, 2) and np.isfinite(logits).all()


# --- #23-#25: pixel height refine (agent-ae70e7) ---


def _refine_opt():
    from test_optimizer_training import _make_optimizer

    opt = _make_optimizer(H=12, W=12, M=3, max_layers=6)
    for _ in range(3):
        opt.step(record_best=True)
    return opt


def test_23_refine_keeps_transparent_pixels_unprinted():
    """#23: pixel and plateau refine never raise pixels whose alpha < 128
    (their colour error weighs nothing, so smoothness and bleed did)."""
    from autoforge.Helper.HeightAssign import heights_to_logits
    from autoforge.Helper.PixelHeightRefine import refine_pixel_heights, refine_plateaus

    opt = _refine_opt()
    alpha = torch.full((12, 12), 255.0)
    alpha[:, :6] = 0.0  # left half transparent
    opt.alpha = alpha
    z = torch.full((12, 12), 4.0)
    z[:, :6] = 0.0
    opt.best_params["pixel_height_logits"] = heights_to_logits(z, opt.max_layers)
    opt.best_params["height_offsets"] = torch.zeros_like(opt.best_params["height_offsets"])
    opt.pixel_height_logits = opt.best_params["pixel_height_logits"].clone()
    for step in (lambda: refine_pixel_heights(opt, sweeps=2, radius=-1), lambda: refine_plateaus(opt)):
        step()
        _, dh = opt.get_discretized_solution(best=True)
        assert int(dh[:, :6].max()) == 0


def test_24_refine_restores_heights_when_scoring_fails(monkeypatch):
    """#24: an error while measuring the new heights restores the old ones."""
    import pytest

    from autoforge.Helper import PruningHelper
    from autoforge.Helper.PixelHeightRefine import refine_pixel_heights

    opt = _refine_opt()
    before = opt.best_params["pixel_height_logits"].clone()
    live = opt.pixel_height_logits
    calls = []

    def flaky(o, dg, **kw):
        calls.append(1)
        if len(calls) > 1:
            raise RuntimeError("CUDA out of memory")
        return 1e9  # any new heights would win

    monkeypatch.setattr(PruningHelper, "_compute_loss_for_heightmap", flaky)
    with pytest.raises(RuntimeError):
        refine_pixel_heights(opt, sweeps=1, radius=-1)
    assert torch.equal(opt.best_params["pixel_height_logits"], before)
    assert opt.pixel_height_logits is live


def test_25_apply_stack_is_exception_safe_and_handles_a_missing_loss(monkeypatch):
    """#25: an error mid-swap leaves the old stack in place, and a None loss
    reverts instead of crashing in the log line."""
    import pytest

    from autoforge.Helper import PixelHeightRefine as phr

    opt = _refine_opt()
    stack = opt.best_params["global_logits"].clone()
    new_dg = torch.zeros(opt.max_layers, dtype=torch.long)

    def boom(*a, **k):
        raise RuntimeError("refine failed")

    monkeypatch.setattr(phr, "refine_pixel_heights", boom)
    with pytest.raises(RuntimeError):
        phr.apply_stack(opt, new_dg)
    assert torch.equal(opt.best_params["global_logits"], stack)

    monkeypatch.setattr(phr, "refine_pixel_heights", lambda *a, **k: None)
    monkeypatch.setattr(opt, "solution_loss", lambda: None)
    assert phr.apply_stack(opt, new_dg) is False
    assert torch.equal(opt.best_params["global_logits"], stack)


# --- #27, #30: FilamentOptimizer (agent-a09ed2) ---


def test_27_best_image_is_none_before_any_best_solution():
    """#27: the webui previews ask for the best image before the first
    discrete check; it returns None instead of raising."""
    from test_optimizer_training import _make_optimizer

    opt = _make_optimizer()
    assert opt.best_params is None
    assert opt.get_best_discretized_image() is None


def test_30_failed_init_does_not_raise_in_destructor():
    """#30: an __init__ that fails early doesn't add an AttributeError from
    __del__ on top of the real error."""
    import gc
    import sys

    from test_optimizer_training import _args

    from autoforge.Modules.Optimizer import FilamentOptimizer

    caught = []
    old = sys.unraisablehook
    sys.unraisablehook = lambda u: caught.append(u)
    try:
        with pytest.raises(Exception):
            FilamentOptimizer(
                _args(), None, np.zeros((4, 4), np.float32), np.zeros((4, 4), np.int32),
                np.zeros((8, 3), np.float32), torch.rand(3, 3), torch.ones(3), torch.zeros(3),
                torch.device("cpu"), perception_loss_module=None,
            )
        gc.collect()
    finally:
        sys.unraisablehook = old
    assert not caught, [str(u.exc_value) for u in caught]


# --- #33, #35-#38: settings, filament import, uploads (agent-a73b83) ---


def test_33_background_color_is_validated():
    """#33: an invalid base colour is rejected up front, short forms are
    normalised."""
    from pydantic import ValidationError

    from autoforge.webui.models import OptimizationSettings

    assert OptimizationSettings(background_color="#fff").background_color == "#ffffff"
    assert OptimizationSettings(background_color="").background_color == "#000000"
    with pytest.raises(ValidationError):
        OptimizationSettings(background_color="red")


def test_35_same_brand_and_name_in_different_types_stay_separate(tmp_path):
    """#35: "Acme Black" PLA and PETG are two filaments, on the first import
    and on a re-import; a file without types still updates its match."""
    from autoforge.webui.services.filament_service import FilamentService

    svc = FilamentService(library_path=str(tmp_path))
    svc.import_csv("Brand,Name,Color,TD,Type\nAcme,Black,#000000,1.0,PLA\nAcme,Black,#111111,2.0,PETG\n",
                   _mark_user_import=False)
    assert sorted((f.filament_type, f.td) for f in svc.list()) == [("PETG", 2.0), ("PLA", 1.0)]
    svc.import_csv("Brand,Name,Color,TD,Type\nAcme,Black,#000000,1.5,PLA\nAcme,Black,#111111,2.5,PETG\n",
                   _mark_user_import=False)
    assert sorted((f.filament_type, f.td) for f in svc.list()) == [("PETG", 2.5), ("PLA", 1.5)]

    other = FilamentService(library_path=str(tmp_path / "b"))
    other.import_csv("Brand,Name,Color,TD,Type\nAcme,Red,#ff0000,1.0,PLA\n", _mark_user_import=False)
    other.import_csv("Brand,Name,Color,TD\nAcme,Red,#ee0000,3.0\n", _mark_user_import=False)
    assert [(f.color, f.td) for f in other.list()] == [("#ee0000", 3.0)]


def test_36_multiline_csv_cell_keeps_its_line_break(tmp_path):
    """#36: a quoted cell spanning lines keeps the line break."""
    from autoforge.webui.services.filament_service import FilamentService

    svc = FilamentService(library_path=str(tmp_path))
    out = svc.import_csv('Brand,Name,Color,TD,Type\nAcme,"Multi\nline",#222222,1.0,PLA\n', _mark_user_import=False)
    assert [f.name for f in out] == ["Multi\nline"]


def test_37_csv_with_a_bom_keeps_the_brand(tmp_path):
    """#37: Excel's UTF-8 BOM before the header no longer hides "Brand"."""
    from autoforge.webui.services.filament_service import FilamentService

    svc = FilamentService(library_path=str(tmp_path))
    out = svc.import_csv("﻿Brand,Name,Color,TD\nAcme,Red,#ff0000,1\n", _mark_user_import=False)
    assert [(f.brand, f.name) for f in out] == [("Acme", "Red")]


def test_38_get_path_never_returns_the_uploads_folder(tmp_path, monkeypatch):
    """#38: "" / "." / a subfolder are not images."""
    from autoforge.webui.config import config
    from autoforge.webui.services.image_service import ImageService

    monkeypatch.setattr(config, "uploads_dir", str(tmp_path))
    (tmp_path / "sub").mkdir()
    (tmp_path / "a.png").write_bytes(b"x")
    svc = ImageService()
    for name in ("", ".", "sub", "../" + tmp_path.name):
        assert svc.get_path(name) is None, name
    assert svc.get_path("a.png") == os.path.realpath(tmp_path / "a.png")
