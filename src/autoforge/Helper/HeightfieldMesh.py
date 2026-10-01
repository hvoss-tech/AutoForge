"""Exact, reduced triangle mesh of the solid a height map describes.

The reference solid is the per-pixel mesh: a vertex at every pixel centre at
its height, every grid cell split into two triangles along its
(i, j)-(i+1, j+1) diagonal, vertical walls along the outline and a flat
bottom at z=0. The layer quantisation leaves that surface made of large
planar pieces; this module meshes the same surface with as few vertices as
it can find a valid triangulation for. Any triangulation of a closed surface
has 2V - 4 triangles, so every vertex that is not a corner of the surface
and can be left out saves two triangles.

Every output vertex is a vertex of the per-pixel mesh (same float32
coordinates) and every triangle lies in one of its planes. The mesh is
closed, 2-manifold along every edge and consistently oriented (outward).

Top surface
-----------
Row i of cells is a sequence of half-cells t = 2j (triangle (i,j), (i+1,j),
(i+1,j+1)) and t = 2j+1 (triangle (i,j), (i,j+1), (i+1,j+1)); neighbours share
an edge. A run of coplanar half-cells is a trapezoid between rows i and i+1
(sides vertical or diagonal). Trapezoids of one plane in consecutive rows
with the same span on the shared row and collinear sides are stacked into
one convex quadrilateral (a "stack"). A vertex is kept where a stack has a
corner; each stack is zipped between the kept vertices on its boundary.
Kept vertices on a straight vertical/diagonal crease between exactly two
planes are dropped when each side is either the inside of a stack or a
one-row trapezoid; the gaps next to one-row trapezoids form "monotone
mountains" along the crease, triangulated separately.

Walls and bottom reuse the kept outline vertices.
"""

import numpy as np


def _row_runs(brk):
    """Maximal runs of a row sequence split where ``brk`` (between t and t+1)
    is set: (row, first t, last t)."""
    rows = brk.shape[0]
    edge = np.ones((rows, 1), dtype=bool)
    r, t0 = np.nonzero(np.concatenate([edge, brk], axis=1))
    _, t1 = np.nonzero(np.concatenate([brk, edge], axis=1))
    return r, t0, t1


def _segments(counts):
    """For groups of the given sizes: group id and position of every element."""
    gid = np.repeat(np.arange(len(counts)), counts)
    pos = np.arange(len(gid)) - np.repeat(np.cumsum(counts) - counts, counts)
    return gid, pos


def _zipper_strips(row, xlt, xrt, xlb, xrb, key, W):
    """Triangulate trapezoids between grid rows ``row`` and ``row+1``: top
    edge from column ``xlt`` to ``xrt``, bottom edge from ``xlb`` to ``xrb``,
    through the vertices in ``key`` (sorted flat indices i * W + j) on them.

    The two chains lie on distinct parallel lines, so zipping them left to
    right never makes a degenerate triangle; n_top + n_bottom vertices become
    n_top + n_bottom - 2 triangles. Returns [N, 3] indices into ``key``,
    counter-clockwise seen from +z with row 0 on top.
    """
    st = np.searchsorted(key, row * W + xlt)
    nt = np.searchsorted(key, row * W + xrt, side="right") - st
    sb = np.searchsorted(key, (row + 1) * W + xlb)
    nb = np.searchsorted(key, (row + 1) * W + xrb, side="right") - sb

    p, k = _segments(nt - 1)
    q = st[p] + k + 1
    x = key[q] - row[p] * W
    m = np.searchsorted(key, (row[p] + 1) * W + x) - sb[p] - 1
    m = np.clip(m, 0, nb[p] - 1)
    tri_top = np.stack([q - 1, sb[p] + m, q], axis=1)

    p, k = _segments(nb - 1)
    q = sb[p] + k + 1
    x = key[q] - (row[p] + 1) * W
    t = np.searchsorted(key, row[p] * W + x, side="right") - st[p] - 1
    t = np.clip(t, 0, nt[p] - 1)
    tri_bottom = np.stack([st[p] + t, q - 1, q], axis=1)
    return np.concatenate([tri_top, tri_bottom])


def _lookup(key, pos):
    """Index of ``pos`` in the sorted ``key`` and whether it is there."""
    i = np.minimum(np.searchsorted(key, pos), len(key) - 1)
    return i, key[i] == pos


def _convex_zipper(ring, poly, i, j):
    """Triangulate convex polygons given as boundary rings: ``ring`` lists the
    vertex ids of all polygons back to back (each polygon's in cyclic order,
    ``poly`` its polygon id, sorted), ``i``/``j`` the vertices' grid row and
    column. Vertices may be collinear (points on the sides); rings whose
    vertices are all collinear (no area) are skipped.

    A polygon with k >= 3 corners (vertices where the boundary turns) is
    split into two chains P and Q from corner X to corner Y: X = the first
    corner, Y = corner k // 2 (k >= 4), or for a triangle the ends of the
    side with the most vertices. The chains are zipped; a triangle can only
    be degenerate if a chain steps along the side through X while the other
    still is at X, or along the side through Y after the other reached Y.
    So P's first step (a duplicate-vertex triangle, dropped) comes first,
    Q's first step second, P's and Q's last steps last (Q's is a duplicate);
    in between any order is valid (convex). V vertices -> V - 2 triangles.
    Returns [N, 3] vertex ids (unoriented).
    """
    L = len(ring)
    start = np.ones(L, dtype=bool)
    start[1:] = poly[1:] != poly[:-1]
    first = np.flatnonzero(start)
    size = np.diff(np.append(first, L))
    pid = np.repeat(np.arange(len(first)), size)
    base = first[pid]
    pos = np.arange(L) - base
    nxt = base + (pos + 1) % size[pid]
    prv = base + (pos - 1) % size[pid]
    ri, rj = i[ring], j[ring]
    # turn at every vertex (x = column, y = -row)
    ax, ay = rj - rj[prv], ri[prv] - ri
    bx, by = rj[nxt] - rj, ri - ri[nxt]
    corner = ax * by - ay * bx != 0
    k = np.bincount(pid, weights=corner, minlength=len(first)).astype(np.int64)
    rank_base = np.cumsum(k) - k
    cidx = np.append(np.flatnonzero(corner), 0)  # ring positions of the corners
    live = k >= 3
    X = cidx[np.where(live, rank_base, len(cidx) - 1)]
    Y = cidx[np.where(live, rank_base + k // 2, len(cidx) - 1)]
    t = np.flatnonzero(live & (k == 3))
    if len(t):
        c = cidx[rank_base[t][:, None] + np.arange(3)]
        # vertices on the sides c0 -> c1, c1 -> c2, c2 -> c0 (ring order)
        side = np.stack(
            [c[:, 1] - c[:, 0], c[:, 2] - c[:, 1], size[t] - (c[:, 2] - c[:, 0])], axis=1
        )
        best = np.argmax(side, axis=1)
        X[t] = c[np.arange(len(t)), best]
        Y[t] = c[np.arange(len(t)), (best + 1) % 3]
    lp = np.flatnonzero(live)
    n = (Y[lp] - X[lp]) % size[lp]  # P: X -> Y forwards, Q: X -> Y backwards
    m = size[lp] - n
    ep, ek = _segments(n)
    eq, el = _segments(m)
    np_, mq = n[ep], m[eq]
    ep, eq = lp[ep], lp[eq]
    pv = ring[first[ep] + (X[ep] - first[ep] + ek + 1) % size[ep]]
    qv = ring[first[eq] + (X[eq] - first[eq] - el - 1) % size[eq]]
    pprog = np.where(ek == 0, -2.0, np.where(ek + 1 == np_, 3.0, (ek + 1) / np_))
    qprog = np.where(el == 0, -1.0, np.where(el + 1 == mq, 4.0, (el + 1) / mq))
    unit = np.concatenate([ep, eq])
    isQ = np.concatenate([np.zeros(len(ep), bool), np.ones(len(eq), bool)])
    prog = np.concatenate([pprog, qprog])
    vert = np.concatenate([pv, qv])
    o = np.lexsort((isQ, prog, unit))
    unit, isQ, vert = unit[o], isQ[o], vert[o]
    M = len(o)
    idx = np.arange(M)
    st = np.ones(M, dtype=bool)
    st[1:] = unit[1:] != unit[:-1]
    start_idx = np.maximum.accumulate(np.where(st, idx, 0))
    lastP = np.maximum.accumulate(np.where(~isQ, idx, -1))
    lastQ = np.maximum.accumulate(np.where(isQ, idx, -1))
    prevP = np.concatenate([[-1], lastP[:-1]])
    prevQ = np.concatenate([[-1], lastQ[:-1]])
    xv = ring[X[unit]]
    curP = np.where(prevP >= start_idx, vert[np.maximum(prevP, 0)], xv)
    curQ = np.where(prevQ >= start_idx, vert[np.maximum(prevQ, 0)], xv)
    tri = np.where(
        isQ[:, None],
        np.stack([curP, curQ, vert], axis=1),
        np.stack([curP, vert, curQ], axis=1),
    )
    ok = (tri[:, 0] != tri[:, 1]) & (tri[:, 1] != tri[:, 2]) & (tri[:, 0] != tri[:, 2])
    return tri[ok]


def _stack_rings(a, h, xla, kl, xra, kr, key, W):
    """Boundary rings (see _convex_zipper) of stacks: convex polygons from row
    ``a`` to row ``a + h`` between the left side x = xla + kl * (row - a) and
    the right side x = xra + kr * (row - a), through the vertices of ``key``
    on their boundary. Returns (ring of indices into ``key``, polygon id)."""
    b = a + h
    xlb = xla + kl * h
    xrb = xra + kr * h
    st = np.searchsorted(key, a * W + xla)
    nt = np.searchsorted(key, a * W + xra, side="right") - st
    sb = np.searchsorted(key, b * W + xlb)
    nb = np.searchsorted(key, b * W + xrb, side="right") - sb
    sid, k = _segments(h - 1)
    srow = a[sid] + 1 + k
    li, lin = _lookup(key, srow * W + xla[sid] + kl[sid] * (k + 1))
    ri, rin = _lookup(key, srow * W + xra[sid] + kr[sid] * (k + 1))
    nl = np.bincount(sid[lin], minlength=len(a))
    nr = np.bincount(sid[rin], minlength=len(a))
    size = nl + nb + nr + nt
    off = np.cumsum(size) - size
    ring = np.empty(int(size.sum()), dtype=np.int64)
    # left side top -> bottom, bottom row left -> right, right side bottom ->
    # top, top row right -> left: counter-clockwise
    g = sid[lin]
    ring[off[g] + _segments(nl)[1]] = li[lin]
    g, q = _segments(nb)
    ring[off[g] + nl[g] + q] = sb[g] + q
    g = sid[rin]
    ring[off[g] + nl[g] + nb[g] + nr[g] - 1 - _segments(nr)[1]] = ri[rin]
    g, q = _segments(nt)
    ring[off[g] + nl[g] + nb[g] + nr[g] + q] = st[g] + nt[g] - 1 - q
    return ring, np.repeat(np.arange(len(a)), size)


def _nearest_larger(keys, first, last):
    """For every element: the nearest element before it with key >= its key
    and the nearest after it with key > its key (pointer jumping; ``first``
    and ``last`` mark group bounds, whose keys no other element reaches)."""
    n = len(keys)
    prev = np.arange(n) - 1
    nxt = np.arange(n) + 1
    prev[first] = np.flatnonzero(first)
    nxt[last] = np.flatnonzero(last)
    todo = np.flatnonzero(~(first | last))
    p, q = prev[todo], nxt[todo]
    kt = keys[todo]
    while True:
        mp = keys[p] < kt
        mq = keys[q] <= kt
        if not (mp.any() or mq.any()):
            break
        p = np.where(mp, prev[p], p)
        q = np.where(mq, nxt[q], q)
        prev[todo], nxt[todo] = p, q
    return todo, p, q


def _mountains(di, dj, diag, single_l, single_r, key, W):
    """Triangles filling the gaps along chains of dropped crease vertices
    (rows ``di``, columns ``dj``) on their one-row-trapezoid sides.

    Along a maximal chain of dropped vertices between kept crease vertices
    e0 (above) and e1 (below), the gaps on one side join into a polygon: the
    straight base e0-e1 plus the nearest kept vertex of every row on that
    side, monotone in the rows (after shearing a diagonal base upright: a
    "monotone mountain"). Joining every chain vertex to its nearest vertices
    on either side that are at least as close to the base (the Cartesian
    tree of the distances) triangulates it: n chain vertices -> n triangles.
    Returns [N, 3] flat grid indices (unoriented).
    """
    line = np.where(diag, dj - di, dj)
    o = np.lexsort((di, line, diag))
    di, dj, line, diag = di[o], dj[o], line[o], diag[o]
    single_l, single_r = single_l[o], single_r[o]
    new = np.ones(len(di), dtype=bool)
    new[1:] = (line[1:] != line[:-1]) | (diag[1:] != diag[:-1]) | (di[1:] != di[:-1] + 1)
    starts = np.flatnonzero(new)
    ends = np.append(starts[1:], len(di)) - 1
    step = diag.astype(np.int64)  # column shift per row along the crease
    e0 = (di[starts] - 1) * W + dj[starts] - step[starts]
    e1 = (di[ends] + 1) * W + dj[ends] + step[ends]
    here = di * W + dj
    at = np.searchsorted(key, here)
    chain = np.repeat(np.arange(len(starts)), ends - starts + 1)
    tris = []
    for side, gap in ((-1, single_l), (1, single_r)):
        # chain vertices whose row leaves a gap on this side (rows inside a
        # stack along the crease leave none)
        g = np.flatnonzero(gap)
        cnt = np.bincount(chain[g], minlength=len(starts))
        ch = np.flatnonzero(cnt)
        length = cnt[ch]
        seg, pos = _segments(length + 2)
        is_first = pos == 0
        is_last = pos == np.repeat(length + 1, length + 2)
        inner = ~(is_first | is_last)
        src = g[np.isin(chain[g], ch)]  # g is sorted by chain, rows ascending
        nb = key[at[src] - 1] if side < 0 else key[at[src]]
        vert = np.empty(len(seg), dtype=np.int64)
        vert[is_first] = e0[ch]
        vert[is_last] = e1[ch]
        vert[inner] = nb
        keys = np.full(len(seg), np.iinfo(np.int64).max)
        keys[inner] = -np.abs(nb % W - dj[src])  # closer to the crease = larger
        t, p, q = _nearest_larger(keys, is_first, is_last)
        tris.append(np.stack([vert[p], vert[t], vert[q]], axis=1))
    return np.concatenate(tris)


def _n_valid_quads(quad_valid):
    """Number of valid cells around every grid vertex."""
    H, W = quad_valid.shape[0] + 1, quad_valid.shape[1] + 1
    qpad = np.zeros((H + 1, W + 1), dtype=np.int8)
    qpad[1:-1, 1:-1] = quad_valid
    return qpad[:-1, :-1] + qpad[:-1, 1:] + qpad[1:, :-1] + qpad[1:, 1:]


def _ccw(tris, ki, kj):
    """Orient [N, 3] vertex-index triangles counter-clockwise seen from +z
    (x = column, y = -row), in place."""
    ti, tj = ki[tris], kj[tris]
    cross = (tj[:, 1] - tj[:, 0]) * (ti[:, 0] - ti[:, 2]) - (ti[:, 0] - ti[:, 1]) * (
        tj[:, 2] - tj[:, 0]
    )
    flip = cross < 0
    tris[flip] = tris[flip][:, [0, 2, 1]]
    return tris


def heightfield_mesh(height_map, background_height, maximum_x_y_size, alpha_mask=None):
    """Vertices [V, 3] float32 and outward-oriented triangles [F, 3] of the
    solid ``generate_stl`` writes (see the module docstring)."""
    H, W = height_map.shape
    z = height_map.astype(np.float32) + background_height
    # max(..., 1): a single pixel has no extent to scale.
    scale = maximum_x_y_size / max(W - 1, H - 1, 1)
    xs = np.arange(W, dtype=np.float32) * scale
    ys = (H - 1 - np.arange(H)).astype(np.float32) * scale

    if alpha_mask is None:
        quad_valid = np.ones((H - 1, W - 1), dtype=bool)
    else:
        alpha_mask = np.asarray(alpha_mask)
        if alpha_mask.ndim == 3:  # (H, W, 1), as the image loaders build it
            alpha_mask = alpha_mask[..., 0]
        valid_mask = alpha_mask >= 128
        quad_valid = (
            valid_mask[:-1, :-1]
            & valid_mask[:-1, 1:]
            & valid_mask[1:, 1:]
            & valid_mask[1:, :-1]
        )
    if quad_valid.size == 0 or not quad_valid.any():
        # A one-pixel-wide map, or everything transparent: no cell to mesh.
        return np.empty((0, 3), dtype=np.float32), np.empty((0, 3), dtype=np.int64)
    NT = 2 * (W - 1)

    # --- Runs of coplanar half-cells ---
    # Neighbours share an edge, so they are coplanar iff their slope along the
    # rows agrees (exact: differences of float32 values in float64).
    z64 = z.astype(np.float64)
    dh = np.diff(z64, axis=1)  # [H, W-1]
    dv = np.diff(z64, axis=0)  # [H-1, W]
    brk = np.empty((H - 1, NT - 1), dtype=bool)
    brk[:, 0::2] = dh[:-1] != dh[1:]  # (2j | 2j+1): across the diagonal
    brk[:, 1::2] = dh[:-1, :-1] != dh[1:, 1:]  # (2j+1 | 2j+2): across x = j+1
    half_valid = np.repeat(quad_valid, 2, axis=1)
    brk |= half_valid[:, 1:] != half_valid[:, :-1]
    r, t0, t1 = _row_runs(brk)
    sel = half_valid[r, t0]
    r, t0, t1 = r[sel], t0[sel], t1[sel]
    xlt, xrt = t0 // 2, (t1 + 1) // 2  # top edge span (row r)
    xlb, xrb = (t0 + 1) // 2, t1 // 2 + 1  # bottom edge span (row r+1)
    kl = t0 & 1  # left side: vertical (0) or diagonal (1)
    kr = 1 - (t1 & 1)
    n_tr = len(r)
    skey = r * NT + t0  # sorted
    ekey = r * NT + t1  # sorted

    # --- Crease vertices that can go ---
    # e_X: the two triangles on either side of the edge from vertex (i, j)
    # towards X are coplanar; a candidate has exactly two planes meeting along
    # a straight vertical / diagonal line through it.
    e_se = dh[1:-1, 1:] == dh[2:, 1:]
    e_nw = dh[:-2, :-1] == dh[1:-1, :-1]
    e_s = dh[1:-1, :-1] == dh[2:, 1:]
    e_n = dh[:-2, :-1] == dh[1:-1, 1:]
    e_w = dv[:-1, :-2] == dv[1:, 1:-1]
    e_e = dv[:-1, 1:-1] == dv[1:, 2:]
    inner = _n_valid_quads(quad_valid)[1:-1, 1:-1] == 4
    cand_v = inner & ~e_s & ~e_n & e_e & e_se & e_w & e_nw
    cand_d = inner & ~e_se & ~e_nw & e_e & e_s & e_w & e_n
    cand = np.zeros((H, W), dtype=bool)
    cand[1:-1, 1:-1] = cand_v | cand_d
    diag_grid = np.zeros((H, W), dtype=bool)
    diag_grid[1:-1, 1:-1] = cand_d

    # --- Stacks ---
    # The trapezoid below with the same span on the shared row and collinear
    # sides starts at half-cell 2*xlb + kl and ends at 2*xrb - (t1 & 1); it is
    # in the same plane iff its slope across the shared row agrees too.
    j0 = t0 // 2
    gi = np.where(
        kl == 0, z64[r + 1, j0] - z64[r, j0], z64[r + 1, j0 + 1] - z64[r, j0 + 1]
    )
    below, found = _lookup(skey, (r + 1) * NT + 2 * xlb + kl)
    # Only strips are stacked: both ends of the shared row are crease vertices
    # that could go (a stack corner can't, so stacking elsewhere only costs).
    link = (
        found
        & (r + 1 < H - 1)
        & (xrb > xlb)
        & cand[r + 1, xlb]
        & cand[r + 1, xrb]
        & (t1[below] == 2 * xrb - (t1 & 1))
        & (gi[below] == gi)
    )
    prv = np.full(n_tr, -1)
    prv[below[link]] = np.flatnonzero(link)
    root = np.where(prv < 0, np.arange(n_tr), prv)
    while True:
        nroot = root[root]
        if np.array_equal(nroot, root):
            break
        root = nroot
    height = np.bincount(root, minlength=n_tr)
    heads = np.flatnonzero(prv < 0)
    single = height[root] == 1

    # --- Kept vertices: stack corners (every trapezoid's corners for now) ---
    keep = np.zeros((H, W), dtype=bool)
    keep[r, xlt] = True
    keep[r, xrt] = True
    keep[r + 1, xlb] = True
    keep[r + 1, xrb] = True
    key = np.flatnonzero(keep)


    c = np.flatnonzero(cand.ravel()[key])
    ci, cj = np.divmod(key[c], W)
    cdiag = diag_grid[ci, cj]
    # trapezoids around the vertex: left side ends at / right side starts at
    # the crease in the slab above (s-1) and below (s)
    lu = np.searchsorted(ekey, (ci - 1) * NT + 2 * cj - 1 - cdiag)
    ld = np.searchsorted(ekey, ci * NT + 2 * cj - 1 + cdiag)
    ru = np.searchsorted(skey, (ci - 1) * NT + 2 * cj - cdiag)
    rd = np.searchsorted(skey, ci * NT + 2 * cj + cdiag)
    # a side is either inside a stack (nothing to do) or leaves a gap that a
    # mountain along the crease fills
    single_l = root[lu] != root[ld]
    single_r = root[ru] != root[rd]
    ok = np.ones(len(c), dtype=bool)
    # A gap side needs the next kept vertex on that side: of two neighbouring
    # candidates with a gap side between them, keep every other one.
    conflict = (c[1:] == c[:-1] + 1) & (ci[1:] == ci[:-1]) & single_r[:-1]
    run_start = np.ones(len(c), dtype=bool)
    run_start[1:] = ~conflict
    start_pos = np.maximum.accumulate(np.where(run_start, np.arange(len(c)), 0))
    drop = (np.arange(len(c)) - start_pos) % 2 == 0
    kept = np.ones(len(key), dtype=bool)
    kept[c[drop]] = False
    key = key[kept]
    ki, kj = np.divmod(key, W)
    n_keep = len(key)

    # --- Top surface ---
    ring, poly = _stack_rings(
        r[heads], height[heads], xlt[heads], kl[heads], xrt[heads], kr[heads], key, W
    )
    top = [_convex_zipper(ring, poly, ki, kj)]
    if drop.any():
        m = _mountains(
            ci[drop], cj[drop], cdiag[drop], single_l[drop], single_r[drop], key, W
        )
        top.append(np.searchsorted(key, m))
    top_tris = _ccw(np.concatenate(top), ki, kj)

    # --- Bottom face: the valid region at z=0, through the kept outline vertices ---
    n_valid = _n_valid_quads(quad_valid)
    outline = (n_valid > 0) & (n_valid < 4)
    bkey = key[outline.ravel()[key]]
    rb, b0, b1 = _row_runs(quad_valid[:, 1:] != quad_valid[:, :-1])
    sel = quad_valid[rb, b0]
    rb, b0, b1 = rb[sel], b0[sel], b1[sel] + 1
    bottom_tris = _zipper_strips(rb, b0, b1, b0, b1, bkey, W)
    bottom_tris = np.searchsorted(key, bkey)[bottom_tris][:, [0, 2, 1]] + n_keep

    # --- Walls: one quad per outline edge between consecutive kept vertices ---
    # (P, Q) is the top surface's boundary edge, directed with the surface on
    # its left seen from above; the wall runs Q -> P along the top.
    # Horizontal edges: consecutive kept vertices on the same row.
    q0 = np.flatnonzero(ki[1:] == ki[:-1])
    row, col = ki[q0], kj[q0]
    below_ok = np.zeros(len(q0), dtype=bool)
    above_ok = np.zeros(len(q0), dtype=bool)
    m = row < H - 1
    below_ok[m] = quad_valid[row[m], col[m]]
    m = row > 0
    above_ok[m] = quad_valid[row[m] - 1, col[m]]
    tb = q0[below_ok & ~above_ok]  # region below: walk towards -x
    bb = q0[above_ok & ~below_ok]  # region above: walk towards +x
    P = [tb + 1, bb]
    Q = [tb, bb + 1]
    # Vertical edges: one per row, both ends are kept (a run ends there).
    lpad = np.zeros((H - 1, W + 1), dtype=bool)
    lpad[:, 1:-1] = quad_valid
    li, lc = np.nonzero(lpad[:, 1:] & ~lpad[:, :-1])  # region right of x = lc
    ri_, rc = np.nonzero(lpad[:, :-1] & ~lpad[:, 1:])  # region left of x = rc
    P += [np.searchsorted(key, li * W + lc), np.searchsorted(key, (ri_ + 1) * W + rc)]
    Q += [np.searchsorted(key, (li + 1) * W + lc), np.searchsorted(key, ri_ * W + rc)]
    P = np.concatenate(P)
    Q = np.concatenate(Q)
    Pb, Qb = P + n_keep, Q + n_keep
    # With background_height 0 a vertex at height 0 coincides with its
    # bottom twin: the wall triangle through both has no area. Dropping it
    # keeps the edges paired (the top edge then meets the bottom one).
    zt = z[ki, kj]
    wall_tris = np.concatenate(
        [np.stack([Q, P, Pb], axis=1)[zt[P] != 0], np.stack([Q, Pb, Qb], axis=1)[zt[Q] != 0]]
    )

    # --- Vertices: top kept vertices, then the same at z=0 ---
    verts = np.empty((2 * n_keep, 3), dtype=np.float32)
    verts[:n_keep, 0] = verts[n_keep:, 0] = xs[kj]
    verts[:n_keep, 1] = verts[n_keep:, 1] = ys[ki]
    verts[:n_keep, 2] = z[ki, kj]
    verts[n_keep:, 2] = 0
    return verts, np.concatenate([top_tris, wall_tris, bottom_tris])
