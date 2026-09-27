#!/usr/bin/env python3
"""Colour/swap-limit benchmark: pruning baseline vs. limits held during optimization.

Limit configurations: unlimited, 4/8/16 colours, 10/20/30 swaps, fixed seed,
50mm, 3000 iterations. The pruning baseline trains each image once and prunes
that one solution to every limit (--prune_sweep); the "opt" mode trains once
per image and limit, since the limit shapes the training. Only training + the colour/swap limit
handling run (``--minimal_postprocess``); every other refinement is off so
the two approaches are compared on exactly that part of the pipeline.

    # baseline: unconstrained training, then colour + swap pruning
    python benchmarks/prune_harness.py --tag base --mode prune --log "baseline"
    # experiment: limits enforced by the optimizer itself, pruning off
    python benchmarks/prune_harness.py --tag exp1 --mode opt --log "..." -- --some_flag

Each run's final colour/swap counts are checked against its limits; a
violation is reported (and noted in the logged description). ``--log``
appends a row to results_prune.tsv: git commit, mean loss per configuration,
discard, description.
"""
import argparse
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

IMAGES = ["bird", "cat", "landscape", "mandala", "colorful_women"]
UNLIMITED = 100
# (column name, pruning_max_colors, pruning_max_swaps)
CONFIGS = [
    ("loss_unlimited", UNLIMITED, UNLIMITED),
    ("loss_c_4", 4, UNLIMITED),
    ("loss_c_8", 8, UNLIMITED),
    ("loss_c_16", 16, UNLIMITED),
    ("loss_s_10", UNLIMITED, 10),
    ("loss_s_20", UNLIMITED, 20),
    ("loss_s_30", UNLIMITED, 30),
]
TSV = os.path.join(ROOT, "results_prune.tsv")


def _run(image, folder, args, limit_args):
    os.makedirs(folder, exist_ok=True)
    cmd = [
        sys.executable,
        os.path.join(ROOT, "benchmarks", "run_experiment.py"),
        "--input_image", os.path.join(ROOT, "images", "test_images", image + ".jpg"),
        "--csv_file", os.path.join(ROOT, args.csv),
        "--output_folder", folder,
        "--stl_output_size", str(args.size),
        "--iterations", str(args.iterations),
        "--perform_pruning" if args.mode == "prune" else "--no-perform_pruning",
        "--minimal_postprocess",
        "--no-visualize",
        "--disable_visualization_for_gradio", "1",
        "--random_seed", str(args.seed),
    ] + limit_args + args.extra
    t0 = time.time()
    with open(os.path.join(folder, "log.txt"), "w") as log:
        proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=log, text=True, cwd=ROOT)
    with open(os.path.join(folder, "stdout.txt"), "w") as f:
        f.write(proc.stdout)
    res = {"loss": float("nan"), "elapsed_s": time.time() - t0, "error": proc.returncode}
    for line in proc.stdout.splitlines():
        if line.startswith("EXPERIMENT_RESULT "):
            res = json.loads(line[len("EXPERIMENT_RESULT "):])
    return res


def _finish(res, image, col, max_colors, max_swaps):
    # "colors" is the print's filament count, the base included.
    res["ok"] = 0 <= res["colors"] <= max_colors and 0 <= res["swaps"] <= max_swaps
    res.update({"image": image, "config": col})
    return res


def run_sweep(image, configs, out_dir, args):
    """Baseline: train once, prune the same solution to every limit."""
    folder = os.path.join(out_dir, image)
    sweep = ",".join(f"{c}:{s}" for _, c, s in configs)
    base = _run(image, folder, args, ["--prune_sweep", sweep])
    try:
        with open(os.path.join(folder, "prune_sweep.json")) as f:
            sweep_res = json.load(f)
    except OSError:
        sweep_res = [{"loss": float("nan"), "colors": -1, "swaps": -1}] * len(configs)
    return [
        _finish({**base, "loss": r["loss"], "colors": r["colors"], "swaps": r["swaps"]},
                image, col, c, s)
        for r, (col, c, s) in zip(sweep_res, configs)
    ]


def run_one(image, config, out_dir, args):
    col, max_colors, max_swaps = config
    folder = os.path.join(out_dir, col, image)
    res = _run(image, folder, args,
               ["--pruning_max_colors", str(max_colors), "--pruning_max_swaps", str(max_swaps)])
    try:
        with open(os.path.join(folder, "final_counts.json")) as f:
            res.update(json.load(f))
    except OSError:
        res.update({"colors": -1, "swaps": -1})
    return _finish(res, image, col, max_colors, max_swaps)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tag", required=True)
    p.add_argument("--mode", choices=["prune", "opt"], required=True,
                   help="prune: unconstrained training + colour/swap pruning; opt: pruning off")
    p.add_argument("--jobs", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--size", type=int, default=50)
    p.add_argument("--iterations", type=int, default=3000)
    p.add_argument("--csv", default="bambulab.csv")
    p.add_argument("--configs", default=None, help="comma separated subset of column names")
    p.add_argument("--images", default=None, help="comma separated subset of image names")
    p.add_argument("--out", default=os.environ.get("HARNESS_OUT", os.path.join(ROOT, "bench_output", "prune")))
    p.add_argument("--log", default=None, help="append a row to results_prune.tsv with this description")
    p.add_argument("--discard", default="no")
    p.add_argument("--nondeterministic", action="store_true",
                   help="skip the bit-reproducible mode (~25%% faster, but +-5%% run-to-run noise on the means)")
    p.add_argument("extra", nargs="*", help="extra autoforge args (after --)")
    args = p.parse_args()

    if not args.nondeterministic:
        os.environ["AF_DETERMINISTIC"] = "1"
    configs = CONFIGS
    if args.configs:
        wanted = args.configs.split(",")
        configs = [c for c in CONFIGS if c[0] in wanted]
    images = args.images.split(",") if args.images else IMAGES
    out_dir = os.path.join(args.out, args.tag)
    os.makedirs(out_dir, exist_ok=True)

    commit = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, text=True).strip()
    t0 = time.time()
    with ThreadPoolExecutor(args.jobs) as ex:
        if args.mode == "prune":
            per_image = list(ex.map(lambda im: run_sweep(im, configs, out_dir, args), images))
            results = [r for col, _, _ in configs for rs in per_image for r in rs if r["config"] == col]
        else:
            jobs = [(im, c) for c in configs for im in images]
            results = list(ex.map(lambda j: run_one(j[0], j[1], out_dir, args), jobs))
    wall = time.time() - t0

    means = {}
    for col, _, _ in configs:
        rs = [r for r in results if r["config"] == col]
        means[col] = sum(r["loss"] for r in rs) / len(rs)
    violations = [f"{r['config']}/{r['image']}(c={r['colors']},s={r['swaps']})"
                  for r in results if not r["ok"]]
    summary = {"tag": args.tag, "mode": args.mode, "means": means, "wall_s": wall,
               "violations": violations,
               "per_run": {f"{r['config']}/{r['image']}": [round(r["loss"], 3), r["colors"], r["swaps"]]
                           for r in results}}
    with open(os.path.join(out_dir, "results.json"), "w") as f:
        json.dump({"summary": summary, "runs": results}, f, indent=1)
    for r in results:
        print(f"{r['config']:>15s} {r['image']:>15s}  loss={r['loss']:9.3f}  "
              f"colors={r['colors']:3d} swaps={r['swaps']:3d} {'' if r['ok'] else 'VIOLATION'}  "
              f"t={r['elapsed_s']:6.1f}s")
    print("HARNESS_RESULT " + json.dumps({k: v for k, v in summary.items() if k != "per_run"}))

    if args.log is not None:
        new = not os.path.exists(TSV)
        with open(TSV, "a") as f:
            if new:
                f.write("git_commit\t" + "\t".join(c[0] for c in CONFIGS) + "\tdiscard\tdescription\n")
            desc = args.log
            if violations:
                desc += f" [LIMIT VIOLATIONS: {' '.join(violations)}]"
            desc = desc.replace("\t", " ").replace("\n", " ")
            cells = [f"{means[c[0]]:.4f}" if c[0] in means else "-" for c in CONFIGS]
            f.write(f"{commit}\t" + "\t".join(cells) + f"\t{args.discard}\t{desc}\n")


if __name__ == "__main__":
    main()
