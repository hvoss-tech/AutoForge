#!/usr/bin/env python3
"""Interleaved solo timing of two worktrees (A = reference, B = candidate).

The GPU is shared with other work, so a single run's wall time swings by 2x;
alternating A and B runs one at a time and comparing the minimum (and median)
of each side is what makes a +-10% comparison possible.

    python benchmarks/opt_n1_timing.py /path/to/A /path/to/B --reps 4 [-- extra args]
"""
import argparse
import json
import os
import statistics
import subprocess
import sys


def run(worktree, image, seed, extra, out):
    folder = os.path.join(out, os.path.basename(worktree.rstrip("/")))
    os.makedirs(folder, exist_ok=True)
    cmd = [
        "nice", "-n", "10", sys.executable, os.path.join(worktree, "benchmarks", "run_experiment.py"),
        "--input_image", os.path.join(worktree, "images", "test_images", image + ".jpg"),
        "--csv_file", os.path.join(worktree, "benchmarks", "webui_filaments.csv"),
        "--output_folder", folder, "--stl_output_size", "100", "--iterations", "3000",
        "--no-perform_pruning", "--minimal_postprocess", "--no-visualize",
        "--disable_visualization_for_gradio", "1", "--random_seed", str(seed),
    ] + extra
    env = dict(os.environ, AF_DETERMINISTIC="1")
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True, cwd=worktree, env=env)
    for line in p.stdout.splitlines():
        if line.startswith("EXPERIMENT_RESULT "):
            return json.loads(line[len("EXPERIMENT_RESULT "):])
    raise RuntimeError(f"run failed in {worktree}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("a")
    p.add_argument("b")
    p.add_argument("--reps", type=int, default=4)
    p.add_argument("--image", default="cat")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", default=os.environ.get("HARNESS_OUT", "/tmp/opt_n1_timing"))
    p.add_argument("extra", nargs="*")
    args = p.parse_args()
    t = {"a": [], "b": []}
    for _ in range(args.reps):
        for side in ("a", "b"):
            r = run(getattr(args, side), args.image, args.seed, args.extra if side == "b" else [], args.out)
            t[side].append(r["elapsed_s"])
            print(f"{side} {r['elapsed_s']:.1f}s loss={r['loss']:.4f}", flush=True)
    res = {s: {"min": min(v), "median": statistics.median(v), "all": [round(x, 1) for x in v]} for s, v in t.items()}
    res["ratio_min"] = res["b"]["min"] / res["a"]["min"]
    res["ratio_median"] = res["b"]["median"] / res["a"]["median"]
    print("TIMING_RESULT " + json.dumps(res))


if __name__ == "__main__":
    main()
