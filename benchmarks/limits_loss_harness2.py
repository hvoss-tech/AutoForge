#!/usr/bin/env python3
"""Full-pipeline loss benchmark with and without colour/swap limits.

3 representative images from images/test_images (panorama, corgi, cat), 50mm, 3000 iterations,
pruning on, fixed seed, deterministic (bit-reproducible) runs. Three
configurations per image (unlimited, 4 colours, 10 swaps), all 38 filaments of the webui; limited runs use --constrained_opt like the webui.

    python benchmarks/limits_loss_harness2.py --tag base --log "baseline"
    python benchmarks/limits_loss_harness2.py --tag exp1 --log "..." -- --some_flag

``--log`` appends a row to results_loss_new_2.tsv: git_commit, execution time
(mean over all runs, s), vram usage (mean of the per-run peaks, MB), the mean
loss of every configuration, discard, description. A run whose final colour /
swap counts exceed its limits is reported and noted in the description.
"""
import argparse
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

IMAGES = ["panorama", "corgi", "cat"]
U = 100
# (column, max colours, max swaps)
CONFIGS = [
    ("loss_unlimited", U, U), ("loss_c_4", 4, U), ("loss_s_10", U, 10),
]
TSV = os.path.join(ROOT, "results_loss_new_2.tsv")


def run_one(image, config, out_dir, args):
    col, max_c, max_s = config
    folder = os.path.join(out_dir, col, image)
    os.makedirs(folder, exist_ok=True)
    cmd = [
        sys.executable, os.path.join(ROOT, "benchmarks", "run_experiment.py"),
        "--input_image", os.path.join(ROOT, "images", "test_images", image + ".jpg"),
        "--csv_file", os.path.join(ROOT, args.csv),
        "--output_folder", folder,
        "--stl_output_size", str(args.size),
        "--iterations", str(args.iterations),
        "--perform_pruning", "--no-visualize",
        "--disable_visualization_for_gradio", "1",
        "--random_seed", str(args.seed),
    ]
    if (max_c, max_s) != (U, U):
        cmd += ["--constrained_opt", "--pruning_max_colors", str(max_c),
                "--pruning_max_swaps", str(max_s)]
    cmd += args.extra
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
    try:
        with open(os.path.join(folder, "final_counts.json")) as f:
            res.update(json.load(f))
    except OSError:
        res.update({"colors": -1, "swaps": -1})
    res["ok"] = 0 <= res["colors"] <= max_c and 0 <= res["swaps"] <= max_s
    res.update({"image": image, "config": col})
    return res


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tag", required=True)
    p.add_argument("--jobs", type=int, default=6)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--size", type=int, default=50)
    p.add_argument("--iterations", type=int, default=3000)
    p.add_argument("--csv", default="benchmarks/webui_filaments.csv", help="the webui's active filaments")
    p.add_argument("--configs", default=None, help="comma separated subset of column names")
    p.add_argument("--images", default=None, help="comma separated subset of image names")
    p.add_argument("--out", default=os.environ.get("HARNESS_OUT", os.path.join(ROOT, "bench_output", "limits2")))
    p.add_argument("--log", default=None, help="append a row to results_loss_new_2.tsv")
    p.add_argument("--discard", default="no")
    p.add_argument("--nondeterministic", action="store_true")
    p.add_argument("extra", nargs="*", help="extra autoforge args (after --)")
    args = p.parse_args()

    if not args.nondeterministic:
        os.environ["AF_DETERMINISTIC"] = "1"
    configs = [c for c in CONFIGS if not args.configs or c[0] in args.configs.split(",")]
    images = args.images.split(",") if args.images else IMAGES
    out_dir = os.path.join(args.out, args.tag)
    os.makedirs(out_dir, exist_ok=True)
    commit = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, text=True).strip()

    t0 = time.time()
    jobs = [(im, c) for c in configs for im in images]
    with ThreadPoolExecutor(args.jobs) as ex:
        results = list(ex.map(lambda j: run_one(j[0], j[1], out_dir, args), jobs))
    wall = time.time() - t0

    means = {c[0]: sum(r["loss"] for r in results if r["config"] == c[0]) / len(images) for c in configs}
    mean_s = sum(r["elapsed_s"] for r in results) / len(results)
    mean_vram = 1000 * sum(r.get("peak_nvidia_smi_gb", 0.0) for r in results) / len(results)
    violations = [f"{r['config']}/{r['image']}(c={r['colors']},s={r['swaps']})" for r in results if not r["ok"]]
    summary = {"tag": args.tag, "commit": commit, "means": means, "mean_elapsed_s": mean_s,
               "mean_peak_vram_mb": mean_vram, "wall_s": wall, "violations": violations,
               "per_run": {f"{r['config']}/{r['image']}": [round(r["loss"], 3), r["colors"], r["swaps"],
                                                            round(r["elapsed_s"], 1)] for r in results}}
    with open(os.path.join(out_dir, "results.json"), "w") as f:
        json.dump({"summary": summary, "runs": results}, f, indent=1)
    for r in results:
        print(f"{r['config']:>15s} {r['image']:>10s}  loss={r['loss']:9.3f}  c={r['colors']:3d} "
              f"s={r['swaps']:3d} {'' if r['ok'] else 'VIOLATION'}  t={r['elapsed_s']:6.1f}s")
    print("HARNESS_RESULT " + json.dumps({k: v for k, v in summary.items() if k != "per_run"}))

    if args.log is not None:
        new = not os.path.exists(TSV)
        with open(TSV, "a") as f:
            if new:
                f.write("git_commit\texecution_time (s, avg)\tvram_usage (mb, avg peak)\t"
                        + "\t".join(c[0] for c in CONFIGS) + "\tdiscard\tdescription\n")
            desc = args.log
            if violations:
                desc += f" [LIMIT VIOLATIONS: {' '.join(violations)}]"
            desc = desc.replace("\t", " ").replace("\n", " ")
            cells = [f"{means[c[0]]:.4f}" if c[0] in means else "-" for c in CONFIGS]
            f.write(f"{commit}\t{mean_s:.1f}\t{mean_vram:.0f}\t" + "\t".join(cells)
                    + f"\t{args.discard}\t{desc}\n")


if __name__ == "__main__":
    main()
