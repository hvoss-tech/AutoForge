#!/usr/bin/env python3
"""Mesh-export loop harness: re-export the five fixed inputs
(bench_output/mesh_export/<image>/stl_input.npz, made once by
mesh_gen_inputs.py) with the current generate_stl, time it, count the
triangles and check the result:

  * closed, 2-manifold, consistently oriented (every directed edge exactly
    once and its reverse exactly once), no degenerate triangles,
  * same volume and surface area as the baseline STL (the original
    per-pixel mesh, final_model.stl from the pipeline run),
  * every vertex is a vertex of the per-pixel mesh (bit-identical float32),
    and the top surface is the same: z sampled at 400k random points in grid
    coordinates on the new triangles equals the per-pixel interpolation
    (to 1e-9: exact up to float64 barycentrics).

    python benchmarks/mesh_export_harness.py                 # check + time
    python benchmarks/mesh_export_harness.py --log "desc"    # + append to results_export.tsv
    python benchmarks/mesh_export_harness.py --log "desc" --discard

Columns of results_export.tsv (tab separated):
    git_commit, polygons, export_time (ms), discard, description
polygons = triangles summed over the five images, export_time = sum of the
per-image median of --reps timed generate_stl calls.
"""
import argparse
import importlib
import json
import os
import statistics
import subprocess
import sys
import time

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))
IN = os.path.join(ROOT, "bench_output", "mesh_export")
TSV = os.path.join(ROOT, "results_export.tsv")
IMAGES = ["bird", "lofi", "panorama", "cat", "mandala"]
TMP = os.environ.get("MESH_TMP", "/tmp")

STL_DTYPE = np.dtype([("normal", "<f4", (3,)), ("v", "<f4", (3, 3)), ("attr", "<u2")])


def read_stl(path):
    with open(path, "rb") as f:
        data = f.read()
    n = int(np.frombuffer(data, "<u4", 1, 80)[0])
    assert len(data) == 84 + 50 * n, "bad STL size"
    return np.frombuffer(data, STL_DTYPE, n, 84)["v"].copy()


def mesh_stats(tris):
    t = tris.astype(np.float64)
    cr = np.cross(t[:, 1] - t[:, 0], t[:, 2] - t[:, 0])
    area = 0.5 * np.linalg.norm(cr, axis=1)
    vol = np.einsum("ij,ij->i", t[:, 0], np.cross(t[:, 1], t[:, 2])).sum() / 6.0
    return float(vol), float(area.sum()), area


def check_topology(tris, area):
    errs = []
    v = np.ascontiguousarray(tris.reshape(-1, 3))
    _, idx = np.unique(v.view(np.dtype((np.void, 12))).ravel(), return_inverse=True)
    f = idx.reshape(-1, 3).astype(np.int64)
    nv = int(idx.max()) + 1
    if np.any((f[:, 0] == f[:, 1]) | (f[:, 1] == f[:, 2]) | (f[:, 0] == f[:, 2])):
        errs.append("faces with repeated vertices")
    if np.any(area <= 0):
        errs.append(f"{int((area <= 0).sum())} zero-area faces")
    a = f.ravel()
    b = f[:, [1, 2, 0]].ravel()
    d = a * nv + b
    ds = np.sort(d)
    if np.any(ds[1:] == ds[:-1]):
        errs.append(f"{int((ds[1:] == ds[:-1]).sum())} duplicated directed edges (non-manifold / flipped)")
    r = np.sort(b * nv + a)
    if len(r) != len(ds) or np.any(r != ds):
        errs.append("directed edges without a reverse partner (open / inconsistently oriented)")
    return errs, nv


def ref_height(z, bg, scale, x, y, H):
    # the per-pixel mesh: vertex (i, j) at (j*s, (H-1-i)*s), quads split along (i,j)-(i+1,j+1)
    u = x / scale
    v = (H - 1) - y / scale
    j = np.clip(np.floor(u).astype(np.int64), 0, z.shape[1] - 2)
    i = np.clip(np.floor(v).astype(np.int64), 0, z.shape[0] - 2)
    fu = u - j
    fv = v - i
    a = z[i, j]; b = z[i, j + 1]; c = z[i + 1, j + 1]; d = z[i + 1, j]
    upper = fu >= fv  # triangle (i,j),(i,j+1),(i+1,j+1)
    zu = a + (b - a) * fu + (c - b) * fv
    zl = a + (d - a) * fv + (c - d) * fu
    return np.where(upper, zu, zl) + bg


def sample_top(tris, pts):
    """z of the upward-facing triangle containing each (x, y) point (nan if none)."""
    t = tris.astype(np.float64)
    n = np.cross(t[:, 1] - t[:, 0], t[:, 2] - t[:, 0])
    up = t[n[:, 2] > 1e-12]
    out = np.full(len(pts), np.nan)
    hits = np.zeros(len(pts), np.int64)
    # bucket points on a coarse grid, test every triangle against the points of its buckets
    cell = 2.0
    px, py = pts[:, 0], pts[:, 1]
    gx = np.floor(px / cell).astype(np.int64); gy = np.floor(py / cell).astype(np.int64)
    GX = gx.max() + 1; GY = gy.max() + 1
    key = gy * GX + gx
    order = np.argsort(key, kind="stable")
    ks = key[order]
    starts = np.searchsorted(ks, np.arange(GX * GY)); ends = np.searchsorted(ks, np.arange(GX * GY), "right")
    x0 = np.floor(up[:, :, 0].min(1) / cell).astype(np.int64); x1 = np.floor(up[:, :, 0].max(1) / cell).astype(np.int64)
    y0 = np.floor(up[:, :, 1].min(1) / cell).astype(np.int64); y1 = np.floor(up[:, :, 1].max(1) / cell).astype(np.int64)
    x0 = np.clip(x0, 0, GX - 1); x1 = np.clip(x1, 0, GX - 1); y0 = np.clip(y0, 0, GY - 1); y1 = np.clip(y1, 0, GY - 1)
    nb = (x1 - x0 + 1) * (y1 - y0 + 1)
    CH = 200000
    tri_ids = np.arange(len(up))
    for s in range(0, len(up), CH):
        ti = np.repeat(tri_ids[s:s + CH], nb[s:s + CH])
        off = np.arange(len(ti)) - np.repeat(np.cumsum(nb[s:s + CH]) - nb[s:s + CH], nb[s:s + CH])
        w = x1[ti] - x0[ti] + 1
        bk = (y0[ti] + off // w) * GX + (x0[ti] + off % w)
        cnt = ends[bk] - starts[bk]
        ti2 = np.repeat(ti, cnt)
        po = order[np.repeat(starts[bk], cnt) + (np.arange(cnt.sum()) - np.repeat(np.cumsum(cnt) - cnt, cnt))]
        A = up[ti2, 0]; B = up[ti2, 1]; C = up[ti2, 2]
        P = pts[po]
        den = (B[:, 1] - C[:, 1]) * (A[:, 0] - C[:, 0]) + (C[:, 0] - B[:, 0]) * (A[:, 1] - C[:, 1])
        l1 = ((B[:, 1] - C[:, 1]) * (P[:, 0] - C[:, 0]) + (C[:, 0] - B[:, 0]) * (P[:, 1] - C[:, 1])) / den
        l2 = ((C[:, 1] - A[:, 1]) * (P[:, 0] - C[:, 0]) + (A[:, 0] - C[:, 0]) * (P[:, 1] - C[:, 1])) / den
        l3 = 1 - l1 - l2
        eps = -1e-9
        inside = (l1 >= eps) & (l2 >= eps) & (l3 >= eps)
        zz = l1 * A[:, 2] + l2 * B[:, 2] + l3 * C[:, 2]
        po = po[inside]; zz = zz[inside]
        np.add.at(hits, po, 1)
        out[po] = zz  # a point on a shared edge hits twice - same z either way
    return out, hits


def validate(image, path, ref):
    d = np.load(os.path.join(IN, image, "stl_input.npz"))
    z = d["height_map"].astype(np.float64)
    H, W = z.shape
    bg = float(d["background_height"]); size = float(d["maximum_x_y_size"])
    tris = read_stl(path)
    vol, area, fa = mesh_stats(tris)
    errs, nv = check_topology(tris, fa)
    if abs(vol - ref["volume"]) > 1e-7 * abs(ref["volume"]):
        errs.append(f"volume {vol!r} != ref {ref['volume']!r}")
    if abs(area - ref["area"]) > 1e-7 * ref["area"]:
        errs.append(f"area {area!r} != ref {ref['area']!r}")
    # Exact check in grid space: every vertex must be a grid vertex of the
    # per-pixel mesh (same float32 x, y, z), then z sampled on the new top
    # surface with integer grid coordinates must equal the per-pixel mesh's
    # interpolation (no float32 x/y rounding involved, so this is exact).
    scale = size / max(W - 1, H - 1)
    xs = np.arange(W, dtype=np.float32) * scale
    ys = (H - 1 - np.arange(H)).astype(np.float32) * scale
    zt = d["height_map"].astype(np.float32) + bg
    v = tris.reshape(-1, 3)
    gj = np.clip(np.rint(v[:, 0] / np.float32(scale)).astype(np.int64), 0, W - 1)
    gi = np.clip(H - 1 - np.rint(v[:, 1] / np.float32(scale)).astype(np.int64), 0, H - 1)
    on_grid = (xs[gj] == v[:, 0]) & (ys[gi] == v[:, 1]) & ((zt[gi, gj] == v[:, 2]) | (v[:, 2] == 0))
    if not on_grid.all():
        errs.append(f"{int((~on_grid).sum())} vertices are not per-pixel mesh vertices")
    g = np.stack([gj.astype(np.float64), (H - 1 - gi).astype(np.float64), v[:, 2].astype(np.float64)], 1).reshape(-1, 3, 3)
    rng = np.random.default_rng(0)
    m = 400000
    pts = np.stack([rng.uniform(1e-6, W - 1 - 1e-6, m), rng.uniform(1e-6, H - 1 - 1e-6, m)], 1)
    got, hits = sample_top(g, pts)
    want = ref_height(zt.astype(np.float64), 0.0, 1.0, pts[:, 0], pts[:, 1], H)
    r_dz = float("nan")
    if not d["has_alpha"]:
        miss = np.isnan(got)
        if miss.any():
            errs.append(f"{int(miss.sum())} sample points not covered by the top surface")
        r_dz = np.nanmax(np.abs(got - want))
        if r_dz > 1e-9:
            errs.append(f"top surface differs by up to {r_dz:.3g} mm")
    return {"triangles": len(tris), "vertices": nv, "dz": r_dz, "volume": vol, "area": area, "errors": errs}


def reference(image):
    cache = os.path.join(IN, image, "ref_stats.json")
    if os.path.exists(cache):
        with open(cache) as f:
            return json.load(f)
    tris = read_stl(os.path.join(IN, image, "final_model.stl"))
    vol, area, fa = mesh_stats(tris)
    errs, nv = check_topology(tris, fa)
    r = {"volume": vol, "area": area, "triangles": len(tris), "vertices": nv, "errors": errs}
    with open(cache, "w") as f:
        json.dump(r, f)
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", default=",".join(IMAGES))
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--log")
    ap.add_argument("--discard", action="store_true")
    ap.add_argument("--no-check", action="store_true")
    a = ap.parse_args()
    from autoforge.Helper import OutputHelper
    importlib.reload(OutputHelper)
    total_tris = 0
    total_ms = 0.0
    ok = True
    for image in a.images.split(","):
        d = np.load(os.path.join(IN, image, "stl_input.npz"))
        hm = d["height_map"]
        alpha = d["alpha_mask"] if d["has_alpha"] else None
        out = os.path.join(TMP, f"mesh_export_{image}.stl")
        args = (hm, out, float(d["background_height"]), float(d["maximum_x_y_size"]))
        OutputHelper.generate_stl(*args, alpha_mask=alpha)  # warm-up
        times = []
        for _ in range(a.reps):
            t = time.perf_counter()
            OutputHelper.generate_stl(*args, alpha_mask=alpha)
            times.append((time.perf_counter() - t) * 1000)
        ms = statistics.median(times)
        ref = reference(image)
        if a.no_check:
            n = int(np.frombuffer(open(out, "rb").read(84), "<u4", 1, 80)[0])
            r = {"triangles": n, "errors": []}
        else:
            r = validate(image, out, ref)
        ok &= not r["errors"]
        total_tris += r["triangles"]
        total_ms += ms
        print(f"{image:9s} tris {r['triangles']:9d} ({r['triangles'] / ref['triangles']:.3f} of ref)"
              f"  time {ms:8.1f} ms  (min {min(times):.1f})  dz {r.get('dz', float('nan')):.1e}  {'OK' if not r['errors'] else 'FAIL ' + '; '.join(r['errors'])}",
              flush=True)
    print(f"TOTAL tris {total_tris}  time {total_ms:.1f} ms  {'ALL OK' if ok else 'FAILED'}")
    if a.log:
        commit = subprocess.run(["git", "rev-parse", "--short=8", "HEAD"], cwd=ROOT,
                                capture_output=True, text=True).stdout.strip()
        new = not os.path.exists(TSV)
        with open(TSV, "a") as f:
            if new:
                f.write("git_commit\tpolygons\texport_time (ms)\tdiscard\tdescription\n")
            desc = a.log if ok else "INVALID MESH - " + a.log
            f.write(f"{commit}\t{total_tris}\t{total_ms:.1f}\t{'yes' if (a.discard or not ok) else 'no'}\t{desc}\n")


if __name__ == "__main__":
    main()
