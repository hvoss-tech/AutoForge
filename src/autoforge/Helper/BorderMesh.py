"""Borders around a finished model, on the base layer.

A border widens the base slab (z = 0 .. background_height, printed in the
background filament) past the edge of the picture: a rectangle with optional
rounded corners, an oval, a circle, or a margin that follows the picture's
outline. Hanging holes can be added on top, in the border or on a tab
("ear") that sticks out of it when the border is too narrow to hold them.

Everything is built in the model's own coordinates (mm, x right, y up, the
picture where ``generate_stl`` put it) from the model's footprint, read off
the mesh itself - so a border never needs the optimizer and can be changed
at any time.

Merging the border into the model is not a 3D boolean: the plate shares the
model's bottom plane and (where the picture has no layers) its base plane,
and a union of coplanar faces leaves coincident duplicate vertices that turn
into non-manifold edges once an STL is read back by position. Instead
``stitch_border`` takes the model apart where it stands on the bed - drops
its bottom face, cuts its outline walls at the base height - and closes it
with the plate: a top face between the plate's outline and the model's own
outline vertices, the plate's walls and a new bottom. The result is one
closed, 2-manifold solid with every original vertex of the model's surface
above the base kept exactly.
"""

from __future__ import annotations

import math
import struct
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Optional

import numpy as np

import manifold3d as mf

SHAPES = ("none", "rectangle", "oval", "circle", "outline")
HANGERS = ("none", "top", "corners")

# Accepted ranges (mm). The webui model clamps into the same ones.
WIDTH_RANGE = (1.0, 100.0)
CORNER_RADIUS_RANGE = (0.0, 100.0)
HANGER_DIAMETER_RANGE = (2.0, 20.0)

# A rounded corner wider than this many border widths would leave less than
# half the width around the picture's corner (at 3.4 it cuts into it).
MAX_CORNER_RADIUS_PER_WIDTH = 2.2

# Space between a hanging hole and the picture, and the least material left
# around a hole.
HOLE_CLEARANCE_MM = 1.5
MIN_HOLE_RING_MM = 2.0

# Longest chord of a polygonised arc. Well below a nozzle width, so arcs
# print as arcs.
_ARC_CHORD_MM = 0.5


class BorderError(ValueError):
    """The border can't be built for this model (not a closed solid, no base)."""


@dataclass(frozen=True)
class BorderSpec:
    shape: str = "none"
    width: float = 5.0
    corner_radius: float = 3.0
    # Hanging holes need a border to sit in (or a tab grown out of it).
    hanger: str = "none"
    hanger_diameter: float = 4.0

    @property
    def active(self) -> bool:
        return self.shape != "none"

    @classmethod
    def from_dict(cls, data: Optional[dict[str, Any]]) -> "BorderSpec":
        """Spec from a settings dict; unknown shapes fall back to none and
        sizes are clamped into range, so a hand-edited project file can't
        produce a degenerate plate."""
        if not data:
            return cls()
        default = cls()

        def number(key: str, lo_hi: tuple[float, float]) -> float:
            try:
                value = float(data.get(key, getattr(default, key)))
            except (TypeError, ValueError):
                value = getattr(default, key)
            if not math.isfinite(value):
                value = getattr(default, key)
            return min(max(value, lo_hi[0]), lo_hi[1])

        shape = data.get("shape", "none")
        hanger = data.get("hanger", "none")
        return cls(
            shape=shape if shape in SHAPES else "none",
            width=number("width", WIDTH_RANGE),
            corner_radius=number("corner_radius", CORNER_RADIUS_RANGE),
            hanger=hanger if hanger in HANGERS else "none",
            hanger_diameter=number("hanger_diameter", HANGER_DIAMETER_RANGE),
        )


def _segments(radius: float) -> int:
    """Segments per full turn for an arc of ``radius``."""
    return int(min(720, max(32, math.ceil(2 * math.pi * max(radius, 0.0) / _ARC_CHORD_MM))))


def _round_offset(section: mf.CrossSection, delta: float) -> mf.CrossSection:
    """``section`` grown by ``delta`` with round joins whose polygon stays
    outside the true arc (its corners are on a slightly larger circle)."""
    n = _segments(delta)
    return section.offset(delta / math.cos(math.pi / n), mf.JoinType.Round, circular_segments=n)


def _circle(radius: float) -> mf.CrossSection:
    """A polygon around the circle of ``radius`` (not inscribed in it)."""
    n = _segments(radius)
    return mf.CrossSection.circle(radius / math.cos(math.pi / n), n)


def _vertices(section: mf.CrossSection) -> np.ndarray:
    polygons = [np.asarray(p, dtype=np.float64) for p in section.to_polygons()]
    polygons = [p for p in polygons if len(p)]
    return np.concatenate(polygons) if polygons else np.zeros((0, 2))


def _strip(x0: float, x1: float, y0: float, y1: float) -> mf.CrossSection:
    return mf.CrossSection.square((x1 - x0, y1 - y0)).translate((x0, y0))


# ---------------------------------------------------------------------------
# The model
# ---------------------------------------------------------------------------

def load_mesh(path: str) -> tuple[np.ndarray, np.ndarray]:
    """(vertices float32 [N, 3], faces int64 [M, 3]) of a mesh file (STL or
    PLY), coincident corners merged: an STL stores every triangle's corners
    separately. Exact merge on the float32 bit patterns - the writers emit
    identical coordinates for a shared corner."""
    import trimesh

    # Not force="mesh": that alone costs seconds on a preview mesh.
    mesh = trimesh.load(path, process=False)
    if isinstance(mesh, trimesh.Scene):
        mesh = mesh.to_geometry()
    vertices = np.ascontiguousarray(mesh.vertices, dtype=np.float32)
    faces = np.asarray(mesh.faces, dtype=np.int64)
    if len(faces) == 0:
        raise BorderError("The model is empty.")
    # Sorted on the bit patterns (several times faster than np.unique on rows).
    bits = vertices.view(np.int32).reshape(-1, 3)
    order = np.lexsort(bits.T[::-1])
    sorted_bits = bits[order]
    starts = np.ones(len(order), dtype=bool)
    starts[1:] = (sorted_bits[1:] != sorted_bits[:-1]).any(axis=1)
    inverse = np.empty(len(order), dtype=np.int64)
    inverse[order] = np.cumsum(starts) - 1
    return vertices[order[starts]], inverse[faces]


def _bottom_loops(vertices: np.ndarray, faces: np.ndarray) -> list[np.ndarray]:
    """Vertex loops around the model's bottom face (the faces at z = 0),
    each in the direction of its bottom faces: seen from above, clockwise
    around the picture and counter-clockwise around a hole in it."""
    on_bed = vertices[:, 2] == 0
    bottom = faces[on_bed[faces].all(axis=1)]
    if len(bottom) == 0:
        raise BorderError("The model has no flat bottom to add a border to.")
    a = bottom.reshape(-1)
    b = bottom[:, [1, 2, 0]].reshape(-1)
    # An inner edge is shared by two bottom faces; the outline by one.
    undirected = np.minimum(a, b) * len(vertices) + np.maximum(a, b)
    order = np.argsort(undirected, kind="stable")
    keys = undirected[order]
    single = np.ones(len(keys), dtype=bool)
    same = keys[1:] == keys[:-1]
    single[1:] &= ~same
    single[:-1] &= ~same
    outline = order[single]
    starts, ends = a[outline], b[outline]
    successors: dict[int, list[int]] = defaultdict(list)
    for s, e in zip(starts.tolist(), ends.tolist()):
        successors[s].append(e)
    loops = []
    # A vertex where two parts of the outline touch (cells meeting at one
    # corner) has two successors; either pairing gives valid loops.
    for start in starts.tolist():
        if not successors[start]:
            continue
        loop = [start]
        current = successors[start].pop()
        while current != start:
            loop.append(current)
            if not successors[current]:
                raise BorderError("The model's bottom outline is not closed.")
            current = successors[current].pop()
        loops.append(np.array(loop, dtype=np.int64))
    return loops


def _signed_area(points: np.ndarray) -> float:
    x, y = points[:, 0], points[:, 1]
    return 0.5 * float(np.dot(x, np.roll(y, -1)) - np.dot(np.roll(x, -1), y))


def mesh_footprint(vertices: np.ndarray, faces: np.ndarray) -> mf.CrossSection:
    """Where the model stands on the bed: the region of its bottom face."""
    loops = _bottom_loops(vertices, faces)
    # Reversed: counter-clockwise around the picture, as CrossSection wants.
    contours = [vertices[loop[::-1], :2].astype(np.float64) for loop in loops]
    footprint = mf.CrossSection(contours, mf.FillRule.Positive)
    if footprint.is_empty():
        raise BorderError("The model has no base to add a border to.")
    return footprint


# ---------------------------------------------------------------------------
# The plate
# ---------------------------------------------------------------------------

def effective_corner_radius(spec: BorderSpec) -> float:
    return min(spec.corner_radius, MAX_CORNER_RADIUS_PER_WIDTH * spec.width)


def _shape_section(footprint: mf.CrossSection, spec: BorderSpec) -> mf.CrossSection:
    """The border's outer shape. Every shape holds the picture grown by the
    border width (corners of a rounded rectangle at least half of it)."""
    w = spec.width
    if spec.shape == "outline":
        return _round_offset(footprint, w)
    x0, y0, x1, y1 = footprint.bounds()
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    if spec.shape == "rectangle":
        width, height = x1 - x0 + 2 * w, y1 - y0 + 2 * w
        r = min(effective_corner_radius(spec), 0.5 * min(width, height) - 1e-3)
        if r <= 1e-3:
            return mf.CrossSection.square((width, height), center=True).translate((cx, cy))
        core = mf.CrossSection.square((width - 2 * r, height - 2 * r), center=True)
        # Inscribed corner arcs keep the sides exactly ``w`` out; the
        # corners' clearance is set by the radius cap, not the arcs.
        return core.offset(r, mf.JoinType.Round, circular_segments=_segments(r)).translate((cx, cy))
    # Oval and circle are fitted around the picture grown by the border
    # width, so the margin is at least ``w`` everywhere - and a round motif on
    # a transparent background gets a tight circle, not one around its
    # square image.
    grown = _vertices(_round_offset(footprint, w))
    if spec.shape == "circle":
        import cv2

        (ccx, ccy), radius = cv2.minEnclosingCircle(grown.astype(np.float32))
        # minEnclosingCircle works in float32; a hair of slack keeps every
        # point inside.
        radius = float(radius) * (1 + 1e-6) + 1e-4
        return _circle(radius).translate((float(ccx), float(ccy)))
    # Oval: the picture's aspect ratio, scaled until it holds every point.
    a0, b0 = (x1 - x0) / 2 + w, (y1 - y0) / 2 + w
    scale = float(np.sqrt(((grown[:, 0] - cx) / a0) ** 2 + ((grown[:, 1] - cy) / b0) ** 2).max())
    a, b = a0 * scale, b0 * scale
    n = _segments(max(a, b))
    grow = 1 / math.cos(math.pi / n)
    return mf.CrossSection.circle(1.0, n).scale((a * grow, b * grow)).translate((cx, cy))


def _hole_centers(footprint: mf.CrossSection, plate: mf.CrossSection, spec: BorderSpec) -> list[tuple[float, float]]:
    r = spec.hanger_diameter / 2
    big_r = r + max(MIN_HOLE_RING_MM, r)
    x0, _y0, x1, y1 = footprint.bounds()
    if spec.hanger == "top":
        if spec.shape == "outline":
            # Above the picture's highest point, where it hangs straight.
            pts = _vertices(footprint)
            top = pts[pts[:, 1] >= y1 - 1e-6]
            xs = [float(top[:, 0].min() + top[:, 0].max()) / 2]
        else:
            xs = [(x0 + x1) / 2]
    else:
        inset = max(big_r, 0.15 * (x1 - x0))
        xs = [x0 + inset, x1 - inset] if 2 * inset < x1 - x0 else [(x0 + x1) / 2]
    _, py0, _, py1 = plate.bounds()
    centers = []
    for x in xs:
        # Highest point of the picture under the hole (with its clearance),
        # and of the plate above it.
        local = footprint ^ _strip(x - r - HOLE_CLEARANCE_MM, x + r + HOLE_CLEARANCE_MM, py0 - 1, py1 + 1)
        top_picture = local.bounds()[3] if not local.is_empty() else y1
        column = plate ^ _strip(x - r, x + r, py0 - 1, py1 + 1)
        top_plate = column.bounds()[3] if not column.is_empty() else top_picture
        lowest = top_picture + HOLE_CLEARANCE_MM + r
        middle = (top_picture + top_plate) / 2
        # Centred in the border when it is wide enough to hold the hole with
        # its ring, otherwise as close to the picture as allowed (a tab is
        # added around it).
        y = middle if middle >= lowest and top_plate - (middle + r) >= big_r - r else lowest
        centers.append((x, y))
    return centers


def border_outline(footprint: mf.CrossSection, spec: BorderSpec) -> Optional[mf.CrossSection]:
    """The whole base plate with the border: picture footprint, border and
    hanging tabs, minus the holes. None when ``spec`` adds nothing."""
    if not spec.active:
        return None
    footprint = footprint.simplify(1e-4)
    plate = _shape_section(footprint, spec)
    if spec.hanger == "none":
        return plate
    r = spec.hanger_diameter / 2
    big_r = r + max(MIN_HOLE_RING_MM, r)
    holes = mf.CrossSection()
    for x, y in _hole_centers(footprint, plate, spec):
        # The hole is cut as the inscribed polygon: never wider than asked.
        holes = holes + mf.CrossSection.circle(r, _segments(r)).translate((x, y))
        ring = _circle(big_r).translate((x, y))
        if (ring - plate).area() <= 1e-9:
            continue
        # The hole doesn't fit in the border: a tab around it, joined to the
        # plate below by the hull with the plate's own edge there.
        below = plate ^ _strip(x - big_r, x + big_r, y - 2 * big_r, y)
        plate = plate + ((ring + below).hull() if not below.is_empty() else ring)
    return plate - holes


def outline_size_mm(footprint: mf.CrossSection, spec: BorderSpec) -> tuple[float, float]:
    """(width, height) in mm of the print with its border."""
    plate = border_outline(footprint, spec) or footprint
    x0, y0, x1, y1 = plate.bounds()
    return (x1 - x0, y1 - y0)


# ---------------------------------------------------------------------------
# Meshes
# ---------------------------------------------------------------------------

def _check_base(background_height: float) -> None:
    if not background_height > 0:
        raise BorderError("A border is part of the base layer; it needs a base height above 0.")


def _mesh_arrays(solid: mf.Manifold) -> tuple[np.ndarray, np.ndarray]:
    mesh = solid.to_mesh()
    vertices = np.asarray(mesh.vert_properties, dtype=np.float32)[:, :3]
    return np.ascontiguousarray(vertices), np.asarray(mesh.tri_verts, dtype=np.uint32)


def border_ring_mesh(
    footprint: mf.CrossSection, spec: BorderSpec, background_height: float
) -> Optional[tuple[np.ndarray, np.ndarray]]:
    """(vertices, faces) of just the border - the plate outside the picture,
    z = 0 .. background_height - for drawing next to the model's own mesh.
    None when the spec adds nothing."""
    plate = border_outline(footprint, spec)
    if plate is None:
        return None
    _check_base(background_height)
    ring = plate - footprint
    if ring.is_empty():
        return None
    return _mesh_arrays(mf.Manifold.extrude(ring, background_height))


def plate_mesh(footprint: mf.CrossSection, spec: BorderSpec, background_height: float) -> tuple[np.ndarray, np.ndarray]:
    """The whole plate, z = 0 .. background_height: a FlatForge base layer
    (which is the footprint at that height) with the border."""
    _check_base(background_height)
    plate = border_outline(footprint, spec)
    return _mesh_arrays(mf.Manifold.extrude(footprint if plate is None else plate, background_height))


def _wall_triangles(top: np.ndarray, bottom: np.ndarray) -> np.ndarray:
    """Outward-facing wall between a contour's top and bottom vertex ids
    (the contour counter-clockwise around solid, clockwise around a hole)."""
    t1, b1 = np.roll(top, -1), np.roll(bottom, -1)
    return np.concatenate([np.stack([bottom, b1, t1], axis=1), np.stack([bottom, t1, top], axis=1)])


def stitch_border(
    vertices: np.ndarray, faces: np.ndarray, spec: BorderSpec, background_height: float
) -> tuple[np.ndarray, np.ndarray]:
    """The model (an indexed mesh on z = 0, its outline walls vertical) with
    the border plate in place of its own base below ``background_height``,
    as one closed mesh. See the module docstring."""
    _check_base(background_height)
    vertices = np.asarray(vertices, dtype=np.float32)
    faces = np.asarray(faces, dtype=np.int64)
    loops = _bottom_loops(vertices, faces)
    footprint = mesh_footprint(vertices, faces)
    plate = border_outline(footprint, spec)
    if plate is None:
        return vertices, faces
    if (footprint - plate).area() > 1e-9:
        raise BorderError("The border does not hold the whole picture.")

    z = vertices[:, 2]
    on_bed = z == 0
    n_bed = on_bed[faces].sum(axis=1)
    top_faces = faces[n_bed == 0]
    walls = faces[(n_bed > 0) & (n_bed < 3)]
    if len(top_faces) and z[top_faces].min() < background_height - 1e-5:
        raise BorderError("The model dips below its base height.")

    # The top end of each bottom vertex's wall: the vertex above it.
    wall_ids = np.unique(walls)
    lows, highs = wall_ids[on_bed[wall_ids]], wall_ids[~on_bed[wall_ids]]
    xy = lambda ids: np.ascontiguousarray(vertices[ids, :2]).view(np.dtype((np.void, 8))).ravel()
    order = np.argsort(xy(highs))
    sorted_keys = xy(highs)[order]
    pos = np.minimum(np.searchsorted(sorted_keys, xy(lows)), len(highs) - 1)
    if len(highs) == 0 or not np.array_equal(sorted_keys[pos], xy(lows)):
        raise BorderError("The model's outline walls are not vertical.")
    above = highs[order][pos]

    # Each wall is cut at the base height. Where the picture has no layers
    # its top already is there and the wall below it simply goes away.
    out = [vertices]
    count = len(vertices)
    level = np.float32(background_height)
    cut = np.full(len(vertices), -1, dtype=np.int64)
    at_base = np.abs(z[above] - level) <= 1e-5
    cut[lows[at_base]] = above[at_base]
    new = lows[~at_base]
    cut[new] = count + np.arange(len(new))
    cut_vertices = vertices[new].copy()
    cut_vertices[:, 2] = level
    out.append(cut_vertices)
    count += len(new)

    walls = np.where(on_bed[walls], cut[walls], walls)
    walls = walls[(walls[:, 0] != walls[:, 1]) & (walls[:, 1] != walls[:, 2]) & (walls[:, 0] != walls[:, 2])]

    # The plate: its contours at the base height and on the bed.
    contours = [np.asarray(p, dtype=np.float64) for p in plate.to_polygons()]
    sizes = [len(c) for c in contours]
    flat = np.concatenate(contours)
    plate_top = count + np.arange(len(flat))
    plate_bottom = plate_top + len(flat)
    out.append(np.column_stack([flat, np.full(len(flat), level)]).astype(np.float32))
    out.append(np.column_stack([flat, np.zeros(len(flat))]).astype(np.float32))

    # Top of the border: between the plate's contours and the model's
    # outline (its bottom loops, which already run clockwise around the
    # picture, as holes must), through the cut vertices.
    inner = [cut[loop] for loop in loops]
    polygons = contours + [vertices[loop, :2].astype(np.float64) for loop in loops]
    ids = np.concatenate([plate_top] + inner)
    ring_top = ids[np.asarray(mf.triangulate(polygons), dtype=np.int64)]
    bottom = plate_bottom[np.asarray(mf.triangulate(contours), dtype=np.int64)][:, ::-1]

    side = []
    start = 0
    for size in sizes:
        idx = np.arange(start, start + size)
        side.append(_wall_triangles(plate_top[idx], plate_bottom[idx]))
        start += size

    all_vertices = np.concatenate(out)
    all_faces = np.concatenate([top_faces, walls, ring_top, bottom] + side)
    used = np.unique(all_faces)
    remap = np.full(len(all_vertices), -1, dtype=np.int64)
    remap[used] = np.arange(len(used))
    return all_vertices[used], remap[all_faces]


def check_closed(vertices: np.ndarray, faces: np.ndarray) -> None:
    """BorderError unless the mesh is a closed, 2-manifold solid *by
    position* - the way a slicer reads an STL."""
    key = np.ascontiguousarray(vertices, dtype=np.float32).view(np.dtype((np.void, 12))).ravel()
    if len(np.unique(key)) != len(vertices):
        raise BorderError("The bordered model has coincident vertices.")
    solid = mf.Manifold(mf.Mesh(
        vert_properties=np.ascontiguousarray(vertices, dtype=np.float32),
        tri_verts=np.ascontiguousarray(faces, dtype=np.uint32),
    ))
    if solid.status() != mf.Error.NoError or solid.volume() <= 0:
        raise BorderError(f"The bordered model is not a closed solid ({solid.status().name}).")


def write_binary_stl(vertices: np.ndarray, faces: np.ndarray, filename: str, header: str = "") -> None:
    """Binary STL of an indexed triangle mesh (normals from the winding)."""
    triangles = np.asarray(vertices, dtype=np.float64)[np.asarray(faces, dtype=np.int64)]
    normals = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    norms = np.linalg.norm(normals, axis=1)
    norms[norms == 0] = 1  # Prevent division by zero
    normals /= norms[:, np.newaxis]

    stl_dtype = np.dtype(
        [
            ("normal", np.float32, (3,)),
            ("v1", np.float32, (3,)),
            ("v2", np.float32, (3,)),
            ("v3", np.float32, (3,)),
            ("attr", np.uint16),
        ]
    )
    data = np.empty(len(triangles), dtype=stl_dtype)
    data["normal"] = normals
    data["v1"] = triangles[:, 0]
    data["v2"] = triangles[:, 1]
    data["v3"] = triangles[:, 2]
    data["attr"] = 0
    with open(filename, "wb") as f:
        f.write(header.encode("utf-8")[:80].ljust(80, b" "))
        f.write(struct.pack("<I", len(triangles)))
        f.write(data.tobytes())


def add_border_to_stl(src: str, dst: str, spec: BorderSpec, background_height: float) -> None:
    """Write ``src``'s model with the border merged in to ``dst``."""
    vertices, faces = stitch_border(*load_mesh(src), spec, background_height)
    check_closed(vertices, faces)
    write_binary_stl(vertices, faces, dst, header=f"Binary STL with a {spec.shape} border")


def write_plate_stl(footprint: mf.CrossSection, dst: str, spec: BorderSpec, background_height: float) -> None:
    """A FlatForge base-layer STL (the footprint, ``background_height``
    thick) with the border, at ``dst``."""
    vertices, faces = plate_mesh(footprint, spec, background_height)
    write_binary_stl(vertices, faces, dst, header=f"FlatForge base with a {spec.shape} border")
