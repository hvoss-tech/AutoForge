#!/usr/bin/env python3
"""Loss benchmark over the images in images/test_images.

Runs the full pipeline (optimization + pruning) on every test image at a
small print size, several runs in parallel, and reports the mean final loss,
mean per-run wall time and the peak per-process VRAM.

    python benchmarks/loss_harness.py --tag baseline [--jobs 5] [--seed 42]

Per-image results go to <out>/<tag>/results.json; a one-line summary is
printed as ``HARNESS_RESULT {...}``. ``--log "description"`` also appends a
row to results_loss.tsv (git_commit, loss, execution_speed (ms),
vram_usage (mb), discard, description).
"""
import argparse
import glob
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def run_one(image, out_dir, args):
    name = os.path.splitext(os.path.basename(image))[0]
    folder = os.path.join(out_dir, name)
    os.makedirs(folder, exist_ok=True)
    cmd = [
        sys.executable,
        os.path.join(ROOT, "benchmarks", "run_experiment.py"),
        "--input_image", image,
        "--csv_file", os.path.join(ROOT, args.csv),
        "--output_folder", folder,
        "--stl_output_size", str(args.size),
        "--iterations", str(args.iterations),
        "--perform_pruning",
        "--no-visualize",
        "--disable_visualization_for_gradio", "1",
        "--random_seed", str(args.seed),
    ] + args.extra
    t0 = time.time()
    with open(os.path.join(folder, "log.txt"), "w") as log:
        proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=log, text=True, cwd=ROOT)
    with open(os.path.join(folder, "stdout.txt"), "w") as f:
        f.write(proc.stdout)
    for line in proc.stdout.splitlines():
        if line.startswith("EXPERIMENT_RESULT "):
            res = json.loads(line[len("EXPERIMENT_RESULT "):])
            res["image"] = name
            return res
    return {"image": name, "loss": float("nan"), "elapsed_s": time.time() - t0,
            "peak_nvidia_smi_gb": 0.0, "error": proc.returncode}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tag", required=True)
    p.add_argument("--jobs", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--size", type=int, default=50)
    p.add_argument("--iterations", type=int, default=6000)
    p.add_argument("--csv", default="bambulab.csv")
    p.add_argument("--images", default="images/test_images/*")
    p.add_argument("--out", default=os.environ.get("HARNESS_OUT", os.path.join(ROOT, "bench_output", "loss")))
    p.add_argument("--log", default=None, help="append a row to results_loss.tsv with this description")
    p.add_argument("--discard", default="no")
    p.add_argument("--deterministic", action="store_true",
                   help="bit-reproducible runs (for exact loss comparisons; slower, don't use for timing)")
    p.add_argument("extra", nargs="*", help="extra autoforge args (after --)")
    args = p.parse_args()

    if args.deterministic:
        os.environ["AF_DETERMINISTIC"] = "1"
    images = sorted(glob.glob(os.path.join(ROOT, args.images)))
    out_dir = os.path.join(args.out, args.tag)
    os.makedirs(out_dir, exist_ok=True)
    t0 = time.time()
    with ThreadPoolExecutor(args.jobs) as ex:
        results = list(ex.map(lambda im: run_one(im, out_dir, args), images))
    wall = time.time() - t0

    losses = [r["loss"] for r in results]
    summary = {
        "tag": args.tag,
        "mean_loss": sum(losses) / len(losses),
        "mean_elapsed_ms": 1000 * sum(r["elapsed_s"] for r in results) / len(results),
        "max_vram_mb": 1000 * max(r.get("peak_nvidia_smi_gb", 0.0) for r in results),
        "wall_s": wall,
        "per_image": {r["image"]: r["loss"] for r in results},
    }
    with open(os.path.join(out_dir, "results.json"), "w") as f:
        json.dump({"summary": summary, "runs": results}, f, indent=1)
    for r in results:
        print(f"{r['image']:>18s}  loss={r['loss']:9.3f}  t={r['elapsed_s']:6.1f}s")
    print("HARNESS_RESULT " + json.dumps(summary))

    if args.log is not None:
        commit = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, text=True).strip()
        tsv = os.path.join(ROOT, "results_loss.tsv")
        new = not os.path.exists(tsv)
        with open(tsv, "a") as f:
            if new:
                f.write("git_commit\tloss\texecution_speed (ms)\tvram_usage (mb)\tdiscard\tdescription\n")
            desc = args.log.replace("\t", " ").replace("\n", " ")
            f.write(f"{commit}\t{summary['mean_loss']:.4f}\t{summary['mean_elapsed_ms']:.0f}\t"
                    f"{summary['max_vram_mb']:.0f}\t{args.discard}\t{desc}\n")


if __name__ == "__main__":
    main()
