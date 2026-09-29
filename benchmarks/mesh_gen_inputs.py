#!/usr/bin/env python3
"""Mesh-export loop: produce the fixed inputs once.

Runs the full pipeline (constrained optimization, 8 colours / 20 swaps, all
pruning steps, 6000 iterations, 200mm) on five images and saves exactly what
auto_forge hands to generate_stl (height map in mm, background height, size,
alpha mask) plus the baseline STL. Every export experiment then re-exports
from these files (benchmarks/mesh_export_harness.py).

    python benchmarks/mesh_gen_inputs.py [--images bird,lofi] [--driver]
"""
import argparse
import os
import subprocess
import sys

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
OUT = os.path.join(ROOT, "bench_output", "mesh_export")
IMAGES = ["bird", "lofi", "panorama", "cat", "mandala"]
FLAGS = [
    "--csv_file", os.path.join(ROOT, "benchmarks", "webui_filaments.csv"),
    "--stl_output_size", "200", "--iterations", "6000",
    "--constrained_opt", "--pruning_max_colors", "8", "--pruning_max_swaps", "20",
    "--perform_pruning", "--no-visualize", "--disable_visualization_for_gradio", "1",
    "--random_seed", "42",
]


def driver(out):
    # In-process: capture generate_stl's arguments, then run the real export.
    sys.path.insert(0, os.path.join(ROOT, "src"))
    from autoforge import auto_forge

    orig = auto_forge.generate_stl

    def capture(height_map, filename, background_height, maximum_x_y_size, alpha_mask=None):
        np.savez(
            os.path.join(out, "stl_input.npz"),
            height_map=height_map,
            background_height=np.float64(background_height),
            maximum_x_y_size=np.float64(maximum_x_y_size),
            alpha_mask=np.zeros(0) if alpha_mask is None else alpha_mask,
            has_alpha=alpha_mask is not None,
        )
        return orig(height_map, filename, background_height, maximum_x_y_size, alpha_mask=alpha_mask)

    auto_forge.generate_stl = capture
    auto_forge.start(auto_forge.parse_args())


if __name__ == "__main__":
    if "--driver" in sys.argv:
        i = sys.argv.index("--driver")
        out = sys.argv[i + 1]
        sys.argv = [sys.argv[0]] + sys.argv[i + 2:]
        driver(out)
        sys.exit(0)
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", default=",".join(IMAGES))
    a = ap.parse_args()
    for image in a.images.split(","):
        out = os.path.join(OUT, image)
        os.makedirs(out, exist_ok=True)
        cmd = [sys.executable, __file__, "--driver", out,
               "--input_image", os.path.join(ROOT, "images", "test_images", image + ".jpg"),
               *FLAGS, "--output_folder", out]
        with open(os.path.join(out, "log.txt"), "w") as log:
            p = subprocess.run(cmd, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
        print(image, "rc", p.returncode, flush=True)
