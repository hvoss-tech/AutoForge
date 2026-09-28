#!/usr/bin/env python3
"""Loss benchmark of the optimization alone (no pruning, no post-processing).

Two images from images/test_images, 100mm, 3000 iterations, the webui's active
filaments (benchmarks/webui_filaments.csv), fixed seed, deterministic
(bit-reproducible) runs under ``nice``. ``--no-perform_pruning
--minimal_postprocess``: the reported loss is the full-resolution discrete loss
of the trained solution, straight out of the optimizer.

    python benchmarks/opt_n1_harness.py --tag base --log "baseline"
    python benchmarks/opt_n1_harness.py --tag exp1 --log "..." -- --some_flag

``--log`` appends a row to results_loss_n1_opt.tsv: git_commit, mean loss,
mean execution time (ms per run), mean peak VRAM (MB), discard, description.
"""
import argparse
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
IMAGES = ["cat", "lofi"]
TSV = os.path.join(ROOT, "results_loss_n1_opt.tsv")


def run_one(image, seed, out_dir, args):
    folder = os.path.join(out_dir, f"{image}_s{seed}")
    os.makedirs(folder, exist_ok=True)
    cmd = [
        "nice", "-n", "10",
        sys.executable, os.path.join(ROOT, "benchmarks", "run_experiment.py"),
        "--input_image", os.path.join(ROOT, "images", "test_images", image + ".jpg"),
        "--csv_file", os.path.join(ROOT, args.csv),
        "--output_folder", folder,
        "--stl_output_size", str(args.size),
        "--iterations", str(args.iterations),
        "--no-perform_pruning", "--minimal_postprocess", "--no-visualize",
        "--disable_visualization_for_gradio", "1",
        "--random_seed", str(seed),
    ] + args.extra
    t0 = time.time()
    with open(os.path.join(folder, "log.txt"), "w") as log:
        proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=log, text=True, cwd=ROOT)
    with open(os.path.join(folder, "stdout.txt"), "w") as f:
        f.write(proc.stdout)
    res = {"loss": float("nan"), "elapsed_s": time.time() - t0, "peak_nvidia_smi_gb": 0.0,
           "error": proc.returncode}
    for line in proc.stdout.splitlines():
        if line.startswith("EXPERIMENT_RESULT "):
            res = json.loads(line[len("EXPERIMENT_RESULT "):])
    res.update({"image": image, "seed": seed})
    return res


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tag", required=True)
    p.add_argument("--jobs", type=int, default=3)
    p.add_argument("--seeds", default="42,7,123")
    p.add_argument("--size", type=int, default=100)
    p.add_argument("--iterations", type=int, default=3000)
    p.add_argument("--csv", default="benchmarks/webui_filaments.csv")
    p.add_argument("--images", default=None, help="comma separated subset of image names")
    p.add_argument("--out", default=os.environ.get("HARNESS_OUT", os.path.join(ROOT, "bench_output", "opt_n1")))
    p.add_argument("--log", default=None, help="append a row to results_loss_n1_opt.tsv")
    p.add_argument("--discard", default="n")
    p.add_argument("--nondeterministic", action="store_true")
    p.add_argument("extra", nargs="*", help="extra autoforge args (after --)")
    args = p.parse_args()

    if not args.nondeterministic:
        os.environ["AF_DETERMINISTIC"] = "1"
    images = args.images.split(",") if args.images else IMAGES
    seeds = [int(s) for s in args.seeds.split(",")]
    out_dir = os.path.join(args.out, args.tag)
    os.makedirs(out_dir, exist_ok=True)
    commit = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, text=True).strip()

    jobs = [(im, s) for s in seeds for im in images]
    t0 = time.time()
    with ThreadPoolExecutor(args.jobs) as ex:
        results = list(ex.map(lambda j: run_one(j[0], j[1], out_dir, args), jobs))
    wall = time.time() - t0

    n = len(results)
    mean_loss = sum(r["loss"] for r in results) / n
    mean_ms = 1000 * sum(r["elapsed_s"] for r in results) / n
    mean_vram = 1000 * sum(r.get("peak_nvidia_smi_gb", 0.0) for r in results) / n
    mean_rough = sum(r.get("roughness", float("nan")) for r in results) / n
    summary = {"tag": args.tag, "commit": commit, "mean_loss": mean_loss, "mean_ms": mean_ms,
               "mean_peak_vram_mb": mean_vram, "mean_roughness": mean_rough, "wall_s": wall,
               "per_run": {f"{r['image']}_s{r['seed']}": [round(r["loss"], 4), round(r["elapsed_s"], 1)]
                           for r in results}}
    with open(os.path.join(out_dir, "results.json"), "w") as f:
        json.dump({"summary": summary, "runs": results}, f, indent=1)
    for r in results:
        print(f"{r['image']:>10s} s{r['seed']:<4d} loss={r['loss']:9.4f}  t={r['elapsed_s']:6.1f}s  "
              f"vram={1000 * r.get('peak_nvidia_smi_gb', 0):.0f}MB rough={r.get('roughness', float('nan')):.4f}")
    print("HARNESS_RESULT " + json.dumps(summary))

    if args.log is not None:
        new = not os.path.exists(TSV)
        with open(TSV, "a") as f:
            if new:
                f.write("git_commit\tloss\texecution_speed (ms)\tvram_usage (mb)\tdiscard (y/n)\tdescription\n")
            per = " ".join(f"{k}={v[0]}" for k, v in summary["per_run"].items())
            desc = f"{args.log} [{per}] rough={mean_rough:.4f}".replace("\t", " ").replace("\n", " ")
            f.write(f"{commit}\t{mean_loss:.4f}\t{mean_ms:.0f}\t{mean_vram:.0f}\t{args.discard}\t{desc}\n")


if __name__ == "__main__":
    main()
