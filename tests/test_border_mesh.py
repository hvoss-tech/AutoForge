"""Borders on the base layer (Helper/BorderMesh): the plate around a model,
stitched into its STL as one solid, and the preview ring."""

import math
import os

import numpy as np
import pytest
import trimesh

import manifold3d as mf

from autoforge.Helper.BorderMesh import (
    HOLE_CLEARANCE_MM,
    MAX_CORNER_RADIUS_PER_WIDTH,
    BorderError,
    BorderSpec,
    add_border_to_stl,
    border_outline,
    border_ring_mesh,
    check_closed,
    load_mesh,
    mesh_footprint,
    outline_size_mm,
    stitch_border,
    write_plate_stl,
)
from autoforge.Helper.OutputHelper import generate_flatforge_stls, generate_stl
from autoforge.webui.helpers.colored_mesh import generate_colored_preview_mesh

BH = 0.24
SIZE = 50.0


def _heights(h=40, w=30, seed=0, zero_edge=False):
    rng = np.random.default_rng(seed)
    heights = (rng.integers(0, 6, size=(h, w)) * 0.08).astype(np.float32)
    if zero_edge:
        # No layers along the edge: the model's top meets the border's.
        heights[:3], heights[-3:], heights[:, :3], heights[:, -3:] = 0, 0, 0, 0
    return heights


def _disc_alpha(h=40, w=30):
    yy, xx = np.mgrid[0:h, 0:w]
    inside = (yy - h / 2) ** 2 + (xx - w / 2) ** 2 <= (min(h, w) / 2 - 2) ** 2
    return np.where(inside, 255, 0).astype(np.uint8)


def _two_islands(h=40, w=30):
    alpha = np.zeros((h, w), np.uint8)
    alpha[5:15, 3:12] = 255
    alpha[25:35, 18:27] = 255
    return alpha


def _ring_hole(h=40, w=30):
    alpha = np.full((h, w), 255, np.uint8)
    alpha[12:28, 8:22] = 0
    return alpha


def _diagonal_touch(h=20, w=20):
    """Two squares meeting at one corner: a pinch vertex in the outline."""
    alpha = np.zeros((h, w), np.uint8)
    alpha[2:10, 2:10] = 255
    alpha[10:18, 10:18] = 255
    return alpha


ALPHAS = {None: None, "disc": _disc_alpha, "islands": _two_islands, "hole": _ring_hole}


@pytest.fixture
def stl(tmp_path):
    def make(heights, alpha=None):
        path = str(tmp_path / f"model_{len(os.listdir(tmp_path))}.stl")
        generate_stl(heights, path, BH, SIZE, alpha_mask=alpha)
        return path

    return make


def _manifold(vertices, faces):
    return mf.Manifold(mf.Mesh(
        vert_properties=np.ascontiguousarray(vertices, dtype=np.float32),
        tri_verts=np.ascontiguousarray(faces, dtype=np.uint32),
    ))


def _read_back(path):
    """An STL as a slicer reads it: corners merged by position."""
    return trimesh.Trimesh(*load_mesh(path), process=False)


SPECS = [
    BorderSpec(shape="rectangle", width=4, corner_radius=0),
    BorderSpec(shape="rectangle", width=4, corner_radius=6),
    BorderSpec(shape="oval", width=3),
    BorderSpec(shape="circle", width=3),
    BorderSpec(shape="outline", width=2.5),
    BorderSpec(shape="rectangle", width=3, hanger="top", hanger_diameter=4),
    BorderSpec(shape="outline", width=2, hanger="corners", hanger_diameter=3),
    BorderSpec(shape="oval", width=1, hanger="top", hanger_diameter=5),
]


@pytest.mark.parametrize("zero_edge", [False, True], ids=["layers", "bare-edge"])
@pytest.mark.parametrize("alpha", list(ALPHAS))
@pytest.mark.parametrize("spec", SPECS, ids=lambda s: f"{s.shape}-{s.hanger}-{s.corner_radius:g}")
def test_bordered_stl_is_one_closed_solid_that_adds_exactly_the_plate(stl, tmp_path, spec, alpha, zero_edge):
    alpha_mask = ALPHAS[alpha]() if alpha else None
    src = stl(_heights(zero_edge=zero_edge), alpha_mask)
    dst = str(tmp_path / "bordered.stl")
    add_border_to_stl(src, dst, spec, BH)

    model = _manifold(*load_mesh(src))
    footprint = mesh_footprint(*load_mesh(src))
    plate = border_outline(footprint, spec)
    mesh = _read_back(dst)
    assert mesh.is_watertight and mesh.is_winding_consistent
    solid = _manifold(mesh.vertices, mesh.faces)
    assert solid.status() == mf.Error.NoError
    # Nothing is added inside the picture or above the base; the border is
    # the plate outside the footprint, one base layer thick.
    assert solid.volume() == pytest.approx(model.volume() + (plate - footprint).area() * BH, rel=1e-6)
    assert mesh.bounds[1, 2] == pytest.approx(model.bounding_box()[5])
    assert mesh.bounds[0, 2] == 0
    # The border never splits the print; a filled shape joins separate
    # islands into one piece (an outline only bridges gaps it covers).
    pieces = len(solid.decompose())
    assert pieces <= len(model.decompose())
    if spec.shape != "outline":
        assert pieces == 1
    if pieces == 1:
        # The only ways through are the plate's holes (hanging holes, and a
        # hole in the picture an outline border leaves open).
        assert solid.genus() == _holes(plate)


def _signed_area(contour):
    p = np.asarray(contour)
    return 0.5 * np.sum(p[:, 0] * np.roll(p[:, 1], -1) - np.roll(p[:, 0], -1) * p[:, 1])


def _holes(section):
    return sum(1 for p in section.to_polygons() if _signed_area(p) < 0)


def test_the_picture_surface_is_kept_exactly(stl, tmp_path):
    src = stl(_heights())
    dst = str(tmp_path / "bordered.stl")
    add_border_to_stl(src, dst, BorderSpec(shape="oval", width=3), BH)
    before, _ = load_mesh(src)
    after, _ = load_mesh(dst)
    above = before[before[:, 2] > BH]
    key = lambda v: set(map(bytes, np.ascontiguousarray(v).view(np.dtype((np.void, 12))).ravel()))
    assert key(above) <= key(after)


def test_a_pinched_outline_still_borders(stl, tmp_path):
    src = stl(_heights(20, 20), _diagonal_touch())
    dst = str(tmp_path / "bordered.stl")
    add_border_to_stl(src, dst, BorderSpec(shape="outline", width=2), BH)
    assert _read_back(dst).is_watertight


def test_footprint_is_the_union_of_the_printed_cells(stl):
    alpha = _disc_alpha()
    footprint = mesh_footprint(*load_mesh(stl(_heights(), alpha)))
    valid = alpha >= 128
    cells = valid[:-1, :-1] & valid[:-1, 1:] & valid[1:, :-1] & valid[1:, 1:]
    pitch = SIZE / (max(alpha.shape) - 1)
    assert footprint.area() == pytest.approx(cells.sum() * pitch**2, rel=1e-6)


def test_footprint_keeps_holes_in_the_picture(stl):
    alpha = _ring_hole()
    footprint = mesh_footprint(*load_mesh(stl(_heights(), alpha)))
    assert _holes(footprint) == 1


def test_rectangle_adds_the_width_on_every_side(stl):
    footprint = mesh_footprint(*load_mesh(stl(_heights())))
    w, h = outline_size_mm(footprint, BorderSpec())
    spec = BorderSpec(shape="rectangle", width=7, corner_radius=0)
    assert outline_size_mm(footprint, spec) == pytest.approx((w + 14, h + 14))
    assert border_outline(footprint, spec).area() == pytest.approx((w + 14) * (h + 14), rel=1e-6)


def test_corner_radius_rounds_the_corners_only(stl):
    footprint = mesh_footprint(*load_mesh(stl(_heights())))
    sharp = border_outline(footprint, BorderSpec(shape="rectangle", width=5, corner_radius=0))
    round_ = border_outline(footprint, BorderSpec(shape="rectangle", width=5, corner_radius=4))
    assert round_.bounds() == pytest.approx(sharp.bounds())
    # Four corners of (4 - pi) r^2 each are cut away.
    assert sharp.area() - round_.area() == pytest.approx((4 - math.pi) * 16, rel=2e-2)


def test_corner_radius_never_eats_into_the_picture(stl):
    footprint = mesh_footprint(*load_mesh(stl(_heights())))
    width = 2.0
    plate = border_outline(footprint, BorderSpec(shape="rectangle", width=width, corner_radius=100))
    assert len(plate.to_polygons()) == 1
    # At most MAX_CORNER_RADIUS_PER_WIDTH widths: half the width stays
    # around the picture's corners.
    grown = footprint.offset(width / 2 * 0.999, mf.JoinType.Round, circular_segments=256)
    assert (grown - plate).area() < 1e-6
    capped = border_outline(footprint, BorderSpec(shape="rectangle", width=width, corner_radius=MAX_CORNER_RADIUS_PER_WIDTH * width))
    assert (plate ^ capped).area() == pytest.approx(plate.area())


@pytest.mark.parametrize("shape", ["oval", "circle", "outline", "rectangle"])
@pytest.mark.parametrize("alpha", [None, "disc"])
def test_borders_keep_the_width_clear_around_the_picture(stl, shape, alpha):
    alpha_mask = _disc_alpha() if alpha else None
    footprint = mesh_footprint(*load_mesh(stl(_heights(), alpha_mask)))
    width = 3.0
    plate = border_outline(footprint, BorderSpec(shape=shape, width=width, corner_radius=0))
    grown = footprint.offset(width, mf.JoinType.Round, circular_segments=256)
    assert (grown - plate).area() < 1e-6


def test_circle_around_a_round_motif_fits_the_motif(stl):
    import cv2

    footprint = mesh_footprint(*load_mesh(stl(_heights(), _disc_alpha())))
    points = np.concatenate([np.asarray(p) for p in footprint.to_polygons()]).astype(np.float32)
    _, motif_radius = cv2.minEnclosingCircle(points)
    w, _ = outline_size_mm(footprint, BorderSpec(shape="circle", width=3))
    # The disc grown by the width, not a circle around the whole image.
    assert w == pytest.approx(2 * (motif_radius + 3), abs=0.05)
    assert w < math.hypot(29, 39) * SIZE / 39


@pytest.mark.parametrize(
    "spec, holes",
    [
        (BorderSpec(shape="rectangle", width=12, hanger="top", hanger_diameter=4), 1),
        (BorderSpec(shape="rectangle", width=1, hanger="top", hanger_diameter=4), 1),
        (BorderSpec(shape="oval", width=2, hanger="corners", hanger_diameter=3), 2),
        (BorderSpec(shape="outline", width=1, hanger="top", hanger_diameter=6), 1),
        (BorderSpec(shape="circle", width=1, hanger="corners", hanger_diameter=8), 2),
    ],
)
def test_hanging_holes_never_come_near_the_picture(stl, spec, holes):
    footprint = mesh_footprint(*load_mesh(stl(_heights())))
    plate = border_outline(footprint, spec)
    assert _holes(plate) == holes
    filled = mf.CrossSection([p for p in plate.to_polygons() if _signed_area(p) > 0])
    hole_area = filled - plate
    near = footprint.offset(HOLE_CLEARANCE_MM * 0.999, mf.JoinType.Round, circular_segments=128)
    assert (hole_area ^ near).area() < 1e-9
    # Each is the size asked for (the cut polygon sits inside the circle).
    r = spec.hanger_diameter / 2
    assert hole_area.area() / holes == pytest.approx(math.pi * r * r, rel=0.01)


def test_hanging_hole_needs_a_border():
    assert not BorderSpec(shape="none", hanger="top").active


def test_a_wide_border_holds_the_hole_without_a_tab(stl):
    footprint = mesh_footprint(*load_mesh(stl(_heights())))
    without = border_outline(footprint, BorderSpec(shape="rectangle", width=12, corner_radius=0))
    with_hole = border_outline(footprint, BorderSpec(shape="rectangle", width=12, corner_radius=0, hanger="top"))
    assert with_hole.bounds() == pytest.approx(without.bounds())
    # Centred in the band above the picture.
    _, hy0, _, hy1 = (without - with_hole).bounds()
    assert (hy0 + hy1) / 2 == pytest.approx(footprint.bounds()[3] + 6, abs=1e-3)


def test_a_narrow_border_grows_a_tab_for_the_hole(stl):
    footprint = mesh_footprint(*load_mesh(stl(_heights())))
    without = border_outline(footprint, BorderSpec(shape="rectangle", width=1))
    with_hole = border_outline(footprint, BorderSpec(shape="rectangle", width=1, hanger="top", hanger_diameter=4))
    assert with_hole.bounds()[3] > without.bounds()[3] + 4
    assert len(with_hole.decompose()) == 1


def test_ring_mesh_is_the_plate_outside_the_picture(stl):
    footprint = mesh_footprint(*load_mesh(stl(_heights(), _disc_alpha())))
    spec = BorderSpec(shape="rectangle", width=4, corner_radius=2, hanger="top")
    vertices, faces = border_ring_mesh(footprint, spec, BH)
    ring = trimesh.Trimesh(vertices, faces, process=False)
    assert ring.is_watertight
    plate = border_outline(footprint, spec)
    assert ring.volume == pytest.approx((plate - footprint).area() * BH, rel=1e-5)
    assert vertices[:, 2].min() == pytest.approx(0) and vertices[:, 2].max() == pytest.approx(BH)


def test_no_border_adds_nothing(stl, tmp_path):
    src = stl(_heights())
    footprint = mesh_footprint(*load_mesh(src))
    assert border_ring_mesh(footprint, BorderSpec(width=20), BH) is None
    assert border_outline(footprint, BorderSpec(width=20)) is None
    vertices, faces = load_mesh(src)
    out_v, out_f = stitch_border(vertices, faces, BorderSpec(), BH)
    assert out_v is vertices and out_f is faces


def test_preview_mesh_and_stl_share_the_footprint(stl, tmp_path):
    """The 3D view draws the border (built from the STL) next to the
    colored preview mesh; both must stand on the same footprint."""
    heights, alpha = _heights(), _disc_alpha()
    footprint = mesh_footprint(*load_mesh(stl(heights, alpha)))
    preview = generate_colored_preview_mesh(
        heights, np.zeros(heights.shape + (3,), np.uint8), BH, SIZE, alpha_mask=alpha, max_grid_dim=0
    )
    ply = str(tmp_path / "preview.ply")
    preview.export(ply, encoding="binary")
    from_ply = mesh_footprint(*load_mesh(ply))
    assert from_ply.bounds() == pytest.approx(footprint.bounds())
    assert (from_ply ^ footprint).area() == pytest.approx(footprint.area(), rel=1e-6)


def test_flatforge_base_with_border_is_the_whole_plate(tmp_path):
    heights = np.random.default_rng(1).integers(1, 5, size=(20, 16))
    files = generate_flatforge_stls(
        np.array([0, 1, 0, 1, 1]), heights, np.array([[255, 0, 0], [0, 0, 255]]), ["Red", "Blue"],
        np.array([2.0, 4.0]), 0.08, BH, "#000000", SIZE, str(tmp_path),
    )
    base = [f for f in files if os.path.basename(f).startswith("Background_")][0]
    base_mesh = trimesh.load(base)
    preview = generate_colored_preview_mesh(
        heights.astype(np.float32) * 0.08, np.zeros(heights.shape + (3,), np.uint8), BH, SIZE, max_grid_dim=0
    )
    ply = str(tmp_path / "preview.ply")
    preview.export(ply, encoding="binary")
    footprint = mesh_footprint(*load_mesh(ply))
    dst = str(tmp_path / "base_bordered.stl")
    spec = BorderSpec(shape="rectangle", width=3)
    write_plate_stl(footprint, dst, spec, BH)
    mesh = _read_back(dst)
    assert mesh.is_watertight
    expected = base_mesh.bounds[:, :2] + np.array([[-3, -3], [3, 3]])
    assert mesh.bounds[:, :2].ravel() == pytest.approx(expected.ravel(), abs=1e-4)
    assert mesh.volume == pytest.approx(border_outline(footprint, spec).area() * BH, rel=1e-6)


def test_open_mesh_is_refused():
    box = trimesh.creation.box(bounds=[[0, 0, 0], [1, 1, 1]])
    with pytest.raises(BorderError):
        check_closed(box.vertices, box.faces[:-1])


def test_base_height_zero_is_refused(stl):
    with pytest.raises(BorderError, match="base height"):
        stitch_border(*load_mesh(stl(_heights())), BorderSpec(shape="rectangle"), 0.0)


def test_spec_from_settings_clamps_and_drops_unknown_values():
    spec = BorderSpec.from_dict({"shape": "star", "width": -3, "corner_radius": "x", "hanger": "top", "hanger_diameter": 99})
    assert spec == BorderSpec(shape="none", width=1.0, corner_radius=3.0, hanger="top", hanger_diameter=20.0)
    assert BorderSpec.from_dict({"width": float("nan")}).width == 5.0
    assert BorderSpec.from_dict(None) == BorderSpec()
