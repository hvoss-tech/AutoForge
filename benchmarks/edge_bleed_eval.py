#!/usr/bin/env python3
"""Edge-bleed model evaluation.

Runs the CLI pipeline (8 colours / 20 swaps, pruning, 4000 it, 150mm) on the
five mesh-export images with the current code and saves each final solution
(integer heights, layer stack, filament colours/TDs, base colour, target).
``--score`` then composites every saved solution under both edge models,
with a scorer that does not depend on the code version, and reports the mean
Lab error:

  one-sided (old): presence + s * mean8(presence), clamped to 1
  two-sided (new): (1 - s) * presence + s * mean8(presence)

    python benchmarks/edge_bleed_eval.py --tag old          # solve + save
    python benchmarks/edge_bleed_eval.py --score old new    # table + renders
"""
import argparse
import os
import subprocess
import sys

import numpy as np
import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
OUT = os.path.join(ROOT, "bench_output", "edge_bleed")
IMAGES = ["bird", "lofi", "panorama", "cat", "mandala"]
FLAGS = [
    "--csv_file", os.path.join(ROOT, "benchmarks", "webui_filaments.csv"),
    "--stl_output_size", "150", "--iterations", "4000",
    "--constrained_opt", "--pruning_max_colors", "8", "--pruning_max_swaps", "20",
    "--perform_pruning", "--no-visualize", "--disable_visualization_for_gradio", "1",
    "--random_seed", "42",
]
S = 0.1


def driver(dump):
    sys.path.insert(0, os.path.join(ROOT, "src"))
    from autoforge import auto_forge

    orig = auto_forge._post_optimize_and_export

    def wrapped(args, optimizer, *a, **k):
        loss = orig(args, optimizer, *a, **k)
        output_target = a[2]
        dg, dh = optimizer.get_discretized_solution(best=True)
        np.savez(
            dump,
            z=dh.cpu().numpy().astype(np.int32),
            dg=dg.cpu().numpy().astype(np.int64),
            colors=optimizer.material_colors.detach().float().cpu().numpy(),
            tds=optimizer.material_TDs.detach().float().cpu().numpy(),
            bg=optimizer.background.detach().float().cpu().numpy(),
            target=output_target.detach().float().cpu().numpy(),
            h=float(optimizer.h),
            loss=float(loss),
        )
        return loss

    auto_forge._post_optimize_and_export = wrapped
    auto_forge.start(auto_forge.parse_args())


def composite(z, dg, colors, tds, bg, h, mode, device="cuda"):
    from autoforge.Helper.OptimizerHelper import _layer_opacity, layer_coverage_params, material_run_starts

    z = torch.as_tensor(z, device=device).long()
    dg = torch.as_tensor(dg, device=device)
    colors = torch.as_tensor(colors, device=device)
    tds = torch.as_tensor(tds, device=device)
    bg = torch.as_tensor(bg, device=device)
    L = len(dg)
    layer_colors = colors[dg]
    layer_tds = tds[dg].clamp(1e-8, 1e8)
    p = (torch.arange(L, device=device).view(-1, 1, 1) < z[None]).float()
    k = torch.full((1, 1, 3, 3), 0.125, device=device)
    k[0, 0, 1, 1] = 0
    nb = torch.nn.functional.conv2d(p[:, None], k, padding=1)[:, 0]
    m = (p + S * nb).clamp(0, 1) if mode == "one" else (1 - S) * p + S * nb
    run_start = material_run_starts(layer_colors, layer_tds)
    reach, slow_reach, cov_w = layer_coverage_params(layer_colors, layer_tds, run_start, bg, h)
    zero = torch.zeros_like(z, dtype=torch.float32)
    opac, _, _ = _layer_opacity(m * h, reach, slow_reach, cov_w, run_start, 0, h, zero, zero)
    opac = torch.flip(opac, [0])
    cols = torch.flip(layer_colors, [0])
    trans = 1 - opac
    prev = torch.cat([torch.ones_like(trans[:1]), trans[:-1]], 0)
    remain = torch.cumprod(prev, 0)
    comp = ((remain * opac)[..., None] * cols.view(-1, 1, 1, 3)).sum(0)
    comp = comp + (remain[-1] * trans[-1])[..., None] * bg
    return comp * 255.0


def lab_err(comp, target):
    from autoforge.Helper.ImageHelper import srgb_to_lab

    a = srgb_to_lab(comp.clamp(0, 255))
    b = srgb_to_lab(torch.as_tensor(target, device=comp.device))
    return float(((a - b) ** 2).mean())  # the optimizer's loss: mean over pixels and channels


def score(tags):
    import cv2

    sys.path.insert(0, os.path.join(ROOT, "src"))
    print(f"{'image':10s}" + "".join(f"  {t}@one-sided  {t}@two-sided" for t in tags))
    tot = {(t, m): 0.0 for t in tags for m in ("one", "two")}
    for im in IMAGES:
        row = f"{im:10s}"
        renders = []
        for t in tags:
            d = np.load(os.path.join(OUT, t, im, "solution.npz"))
            for mode in ("one", "two"):
                c = composite(d["z"], d["dg"], d["colors"], d["tds"], d["bg"], float(d["h"]), mode)
                e = lab_err(c, d["target"])
                tot[(t, mode)] += e / len(IMAGES)
                row += f"  {e:12.2f}  {'':>1s}"
                if mode == "two":
                    renders.append(cv2.cvtColor(c.clamp(0, 255).byte().cpu().numpy(), cv2.COLOR_RGB2BGR))
        print(row)
        tgt = cv2.cvtColor(d["target"].astype(np.uint8), cv2.COLOR_RGB2BGR)
        cv2.imwrite(os.path.join(OUT, f"cmp_{im}.png"), np.concatenate([tgt] + renders, 1))
    print(f"{'MEAN':10s}" + "".join(f"  {tot[(t, 'one')]:12.2f}   {tot[(t, 'two')]:12.2f}" for t in tags))


def report(tags):
    sys.path.insert(0, os.path.join(ROOT, "src"))
    from autoforge.Helper.FusedComposite import spike_mask

    print(f"{'tag':12s} {'final loss':>10s} {'spikes found':>12s} {'loss pre-rm':>11s} {'rm cost':>8s} {'spikes left':>11s}")
    for t in tags:
        rows = []
        for im in IMAGES:
            d = np.load(os.path.join(OUT, t, im, "solution.npz"))
            with open(os.path.join(OUT, t, im, "spike_removal_stats.txt")) as f:
                kv = dict(x.split("=") for x in f.readline().strip().split(",")[1:])
            left = int(spike_mask(torch.as_tensor(d["z"], device="cuda"), 1).sum())
            rows.append((float(d["loss"]), int(kv["spikes"]), float(kv["loss_before"]),
                         float(kv["loss_after"]) - float(kv["loss_before"]), left))
            print(f"  {im:10s} {rows[-1][0]:10.2f} {rows[-1][1]:12d} {rows[-1][2]:11.2f} {rows[-1][3]:8.2f} {rows[-1][4]:11d}")
        m = np.mean(rows, axis=0)
        print(f"{t:12s} {m[0]:10.2f} {m[1]:12.0f} {m[2]:11.2f} {m[3]:8.2f} {m[4]:11.0f}")


if __name__ == "__main__":
    if "--driver" in sys.argv:
        i = sys.argv.index("--driver")
        dump = sys.argv[i + 1]
        sys.argv = [sys.argv[0]] + sys.argv[i + 2:]
        driver(dump)
        sys.exit(0)
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag")
    ap.add_argument("--images", default=",".join(IMAGES))
    ap.add_argument("--score", nargs="+")
    ap.add_argument("--report", nargs="+", help="tags: final loss, spikes found / loss cost of their removal, spikes left")
    ap.add_argument("--extra", default="", help="extra CLI flags for this tag, e.g. '--pixel_height_smoothness 1.0'")
    a = ap.parse_args()
    if a.report:
        report(a.report)
        sys.exit(0)
    if a.score:
        score(a.score)
        sys.exit(0)
    for im in a.images.split(","):
        out = os.path.join(OUT, a.tag, im)
        os.makedirs(out, exist_ok=True)
        cmd = [sys.executable, __file__, "--driver", os.path.join(out, "solution.npz"),
               "--input_image", os.path.join(ROOT, "images", "test_images", im + ".jpg"),
               *FLAGS, *a.extra.split(), "--output_folder", out]
        with open(os.path.join(out, "log.txt"), "w") as log:
            p = subprocess.run(cmd, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
        print(im, "rc", p.returncode, flush=True)
