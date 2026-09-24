"""Generate colored PLY mesh from height map + color image for WebUI 3D preview."""

import os
from typing import Optional, Tuple

import numpy as np
import trimesh
from trimesh import Trimesh

# Longest grid side the preview mesh is built at. 0 disables decimation.
#
# The mesh has one vertex per grid point (times two, top and bottom) and
# roughly four triangles per point, so it grows with the *area* of the height
# map. At the default settings (stl_output_size=150, nozzle 0.4) the solved
# grid is 750px on the long side: 845k vertices, 1.7M triangles, a 35MB PLY.
#
# Decimating to 384 was measured at ~70ms instead of ~325ms per slider
# re-render and 8.9MB instead of 35.4MB — but it is *visible*: 384 happens to
# land almost exactly on the processing resolution for default settings, so
# the preview lost half its relief detail in each axis. On the client side
# the full mesh costs only ~28ms to parse and ~50ms for normals (both now off
# the main thread), so the detail was being traded for time nobody was
# waiting on. Off by default; set AUTOFORGE_PREVIEW_MESH_MAX_DIM to trade
# detail for latency at very large stl_output_size values.
#
# It only ever affects the *preview* mesh: the printable STL is generated
# separately by OutputHelper.generate_stl at full resolution.
_DEFAULT_MAX_GRID_DIM = 0


def preview_mesh_max_dim() -> int:
    """Grid cap for preview meshes, overridable with
    ``AUTOFORGE_PREVIEW_MESH_MAX_DIM`` (0 or less disables decimation)."""
    raw = os.environ.get("AUTOFORGE_PREVIEW_MESH_MAX_DIM")
    if raw is None:
        return _DEFAULT_MAX_GRID_DIM
    try:
        return int(raw)
    except ValueError:
        return _DEFAULT_MAX_GRID_DIM


def downsample_grid_step(height: int, width: int, max_dim: int) -> int:
    """Stride that brings ``max(height, width)`` to at most ``max_dim``.

    Plain integer striding (rather than an interpolating resize) on purpose:
    it keeps every sampled height and color an exact value from the real
    solution, so the preview never shows a color that no layer actually has.
    """
    if max_dim <= 0:
        return 1
    return max(1, -(-max(height, width) // max_dim))


def _sample_grid(full_H: int, full_W: int, max_grid_dim: Optional[int]) -> tuple[np.ndarray, np.ndarray]:
    """The original row/column indices the preview grid samples."""
    step = downsample_grid_step(
        full_H,
        full_W,
        preview_mesh_max_dim() if max_grid_dim is None else max_grid_dim,
    )
    rows = np.arange(full_H)
    cols = np.arange(full_W)
    if step > 1:
        # The last row/column is kept as well as the strided ones, so the mesh
        # still spans the full footprint instead of stopping short of the edge.
        rows = np.unique(np.append(np.arange(0, full_H, step), full_H - 1))
        cols = np.unique(np.append(np.arange(0, full_W, step), full_W - 1))
    return rows, cols


def _valid_mask(alpha_mask: Optional[np.ndarray], H: int, W: int) -> np.ndarray:
    valid_mask: np.ndarray = (
        np.ones((H, W), dtype=bool)
        if alpha_mask is None
        else (alpha_mask >= 128).squeeze()
    )
    if valid_mask.ndim == 3 and valid_mask.shape[-1] >= 1:
        valid_mask = valid_mask[:, :, 0]
    return valid_mask.astype(bool)


def _quad_valid(valid_mask: np.ndarray) -> np.ndarray:
    return (
        valid_mask[:-1, :-1]
        & valid_mask[:-1, 1:]
        & valid_mask[1:, 1:]
        & valid_mask[1:, :-1]
    )


def top_vertex_pixel_indices(
    full_H: int,
    full_W: int,
    alpha_mask: Optional[np.ndarray] = None,
    max_grid_dim: Optional[int] = None,
) -> np.ndarray:
    """Flat (row-major) pixel index of every *top* vertex of the mesh that
    ``generate_colored_preview_mesh`` builds for this grid, in the mesh's
    own vertex order.

    The mesh keeps the top and bottom vertex of every corner of a valid quad
    (top ones first, row-major), so its first ``len(result)`` vertices are
    exactly these pixels and the rest are their bottom twins. That is what
    lets a slider edit — which changes colors, never heights — recolor the
    mesh the client already has by sending ``colors[result]`` instead of
    rebuilding, exporting and re-downloading the whole PLY.
    """
    rows, cols = _sample_grid(full_H, full_W, max_grid_dim)
    H, W = len(rows), len(cols)
    if alpha_mask is not None:
        alpha = np.asarray(alpha_mask)
        if alpha.ndim >= 2 and (len(rows) != full_H or len(cols) != full_W):
            alpha = alpha[np.ix_(rows, cols)]
        alpha_mask = alpha
    quad_valid = _quad_valid(_valid_mask(alpha_mask, H, W))
    corners = np.zeros((H, W), dtype=bool)
    corners[:-1, :-1] |= quad_valid
    corners[:-1, 1:] |= quad_valid
    corners[1:, 1:] |= quad_valid
    corners[1:, :-1] |= quad_valid
    ii, jj = np.nonzero(corners)  # row-major, like the mesh's vertex order
    return (rows[ii] * full_W + cols[jj]).astype(np.int64)


def generate_colored_preview_mesh(
    height_map: np.ndarray,
    color_image: np.ndarray,
    background_height: float,
    maximum_x_y_size: float,
    alpha_mask: Optional[np.ndarray] = None,
    background_color: Tuple[int, int, int] = (0, 0, 0),
    max_grid_dim: Optional[int] = None,
) -> Trimesh:
    """Build a colored mesh from a height map + per-pixel RGB color image.

    Top and bottom surfaces (and the side walls at the silhouette boundary)
    all share one vertex per grid point instead of giving every triangle its
    own unique vertices. This lets a renderer interpolate color smoothly
    between adjacent pixels (avoiding a blocky "flat shaded" look) and keeps
    the mesh roughly 6x smaller, which matters for preview load time.

    Args:
        height_map: (H,W) float32, per-pixel heights in mm.
        color_image: (H,W,3) uint8 RGB — color for each top vertex.
        background_height: Height of the base/background slab in mm.
        maximum_x_y_size: Max X/Y dimension of the output mesh in mm.
        alpha_mask: Optional (H,W) bool/uint8 — True = valid. Pixels with
            alpha<128 are omitted.
        background_color: RGB tuple for bottom/side faces (0-255).
        max_grid_dim: Longest grid side to build at; the inputs are strided
            down to fit. Defaults to ``preview_mesh_max_dim()``; pass 0 to
            build at full resolution.

    Returns:
        Trimesh with vertex_colors set.
    """
    full_H, full_W = height_map.shape
    rows, cols = _sample_grid(full_H, full_W, max_grid_dim)
    if len(rows) != full_H or len(cols) != full_W:
        height_map = height_map[np.ix_(rows, cols)]
        color_image = color_image[np.ix_(rows, cols)]
        if alpha_mask is not None:
            alpha = np.asarray(alpha_mask)
            alpha_mask = alpha[np.ix_(rows, cols)] if alpha.ndim >= 2 else alpha

    H, W = height_map.shape

    valid_mask = _valid_mask(alpha_mask, H, W)
    quad_valid = _quad_valid(valid_mask)
    vi, vj = np.nonzero(quad_valid)
    if len(vi) == 0:
        return trimesh.Trimesh()

    # One shared vertex per grid point (row-major index = i * W + j), for
    # both the top surface and the bottom surface. Side walls at the
    # silhouette boundary reuse these same indices — a boundary point's
    # "top" and "bottom" vertices are exactly the wall's top/bottom corners.
    # Positions come from the *original* pixel indices, not the strided ones:
    # an evenly-strided grid with the final row/column appended has one
    # narrower cell at each far edge, and numbering the samples 0..W-1 would
    # stretch that cell to full width, skewing the footprint.
    j, i = np.meshgrid(cols.astype(np.float32), rows.astype(np.float32))
    x = j
    y = (full_H - 1) - i
    scale = maximum_x_y_size / max(full_W - 1, full_H - 1, 1)
    x = x * scale
    y = y * scale

    top_z = height_map.astype(np.float32) + background_height
    bottom_z = np.zeros_like(top_z)

    top_vertices = np.stack([x, y, top_z], axis=2).reshape(-1, 3)
    bottom_vertices = np.stack([x, y, bottom_z], axis=2).reshape(-1, 3)

    top_colors = color_image.reshape(-1, 3).astype(np.uint8)
    bg = np.array(background_color, dtype=np.uint8)
    bottom_colors = np.broadcast_to(bg, top_colors.shape).astype(np.uint8)

    n_top = H * W
    all_vertices = np.concatenate([top_vertices, bottom_vertices], axis=0)
    all_colors = np.concatenate([top_colors, bottom_colors], axis=0)

    def top_idx(ii, jj):
        return ii * W + jj

    def bottom_idx(ii, jj):
        return n_top + ii * W + jj

    faces_list: list[np.ndarray] = []

    # --- Top surface (two triangles per valid quad) ---
    t00 = top_idx(vi, vj)
    t01 = top_idx(vi, vj + 1)
    t11 = top_idx(vi + 1, vj + 1)
    t10 = top_idx(vi + 1, vj)
    faces_list.append(np.stack([t11, t01, t00], axis=1))
    faces_list.append(np.stack([t10, t11, t00], axis=1))

    # --- Bottom surface (reversed winding so the normal points down) ---
    b00 = bottom_idx(vi, vj)
    b01 = bottom_idx(vi, vj + 1)
    b11 = bottom_idx(vi + 1, vj + 1)
    b10 = bottom_idx(vi + 1, vj)
    faces_list.append(np.stack([b00, b01, b11], axis=1))
    faces_list.append(np.stack([b00, b11, b10], axis=1))

    # --- Side walls: one quad per boundary edge, reusing top/bottom indices ---
    def add_walls(cond: np.ndarray, ia, ja, ib, jb, flip: bool):
        """Add a vertical quad between grid points (ia,ja)->(ib,jb) wherever
        `cond` is True. (ia,ja)-(ib,jb) is one edge of a valid quad that
        borders an invalid (or out-of-grid) neighbor."""
        si, sj = np.nonzero(cond)
        if len(si) == 0:
            return
        a_i, a_j = ia(si, sj), ja(si, sj)
        b_i, b_j = ib(si, sj), jb(si, sj)
        tA, tB = top_idx(a_i, a_j), top_idx(b_i, b_j)
        bA, bB = bottom_idx(a_i, a_j), bottom_idx(b_i, b_j)
        if flip:
            faces_list.append(np.stack([tA, bB, tB], axis=1))
            faces_list.append(np.stack([tA, bA, bB], axis=1))
        else:
            faces_list.append(np.stack([tA, tB, bB], axis=1))
            faces_list.append(np.stack([tA, bB, bA], axis=1))

    # Each direction's correct winding (outward normal) was derived by hand
    # and verified against mesh.volume/is_winding_consistent — it is NOT
    # uniform across directions, so each call states its own `flip`.

    # Left edges: quad (vi,vj) has no valid neighbor to its left. Outward = -X.
    left_cond = np.zeros_like(quad_valid, dtype=bool)
    left_cond[:, 0] = quad_valid[:, 0]
    left_cond[:, 1:] = quad_valid[:, 1:] & (~quad_valid[:, :-1])
    add_walls(left_cond, lambda i_, j_: i_, lambda i_, j_: j_, lambda i_, j_: i_ + 1, lambda i_, j_: j_, flip=True)

    # Right edges: quad (vi,vj) has no valid neighbor to its right. Outward = +X.
    right_cond = np.zeros_like(quad_valid, dtype=bool)
    right_cond[:, -1] = quad_valid[:, -1]
    right_cond[:, :-1] = quad_valid[:, :-1] & (~quad_valid[:, 1:])
    add_walls(right_cond, lambda i_, j_: i_ + 1, lambda i_, j_: j_ + 1, lambda i_, j_: i_, lambda i_, j_: j_ + 1, flip=True)

    # Top edges (row 0 side, i.e. smallest i): quad (vi,vj) has no valid neighbor above. Outward = +Y.
    top_cond = np.zeros_like(quad_valid, dtype=bool)
    top_cond[0, :] = quad_valid[0, :]
    top_cond[1:, :] = quad_valid[1:, :] & (~quad_valid[:-1, :])
    add_walls(top_cond, lambda i_, j_: i_, lambda i_, j_: j_ + 1, lambda i_, j_: i_, lambda i_, j_: j_, flip=True)

    # Bottom edges (last row side): quad (vi,vj) has no valid neighbor below. Outward = -Y.
    bottom_cond = np.zeros_like(quad_valid, dtype=bool)
    bottom_cond[-1, :] = quad_valid[-1, :]
    bottom_cond[:-1, :] = quad_valid[:-1, :] & (~quad_valid[1:, :])
    add_walls(bottom_cond, lambda i_, j_: i_ + 1, lambda i_, j_: j_ + 1, lambda i_, j_: i_ + 1, lambda i_, j_: j_, flip=False)

    faces = np.concatenate(faces_list, axis=0).astype(np.int64)

    # Vertices not referenced by any face (fully-invalid rows/columns) are
    # dropped so the exported mesh doesn't carry dead weight.
    used = np.unique(faces)
    remap = np.full(all_vertices.shape[0], -1, dtype=np.int64)
    remap[used] = np.arange(len(used))
    faces = remap[faces]

    mesh = trimesh.Trimesh(
        vertices=all_vertices[used],
        faces=faces,
        vertex_colors=all_colors[used],
        process=False,
    )
    return mesh


# ---------------------------------------------------------------------------
# Quick smoke test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    H, W = 4, 4
    hm = np.random.uniform(0.0, 1.0, (H, W)).astype(np.float32)
    ci = np.random.randint(0, 256, (H, W, 3), dtype=np.uint8)

    mesh = generate_colored_preview_mesh(
        height_map=hm,
        color_image=ci,
        background_height=0.24,
        maximum_x_y_size=50.0,
    )

    assert mesh.vertices.shape[0] > 0, "Mesh has no vertices"
    assert mesh.faces.shape[0] > 0, "Mesh has no faces"
    assert mesh.visual.vertex_colors is not None, "Mesh has no vertex colors"
    vc = mesh.visual.vertex_colors
    assert vc.shape[0] == mesh.vertices.shape[0]
    # Fully-valid HxW grid: exactly 2*H*W shared vertices (top + bottom).
    assert mesh.vertices.shape[0] == 2 * H * W, mesh.vertices.shape[0]

    print(f"Vertices: {mesh.vertices.shape[0]}")
    print(f"Faces: {mesh.faces.shape[0]}")
    print(f"Vertex colors shape: {vc.shape}")
    print(f"First 6 vertex colors:\n{vc[:6]}")
    print("Smoke test PASSED")
