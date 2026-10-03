#!/usr/bin/env python3
"""
Grid-runner for autoforge.py

Usage example (inline defaults):
    python run_autoforge_grid.py
"""

import json
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from itertools import product
from pathlib import Path
from time import strftime

from tqdm import tqdm

# ---------- editable section ---------- #
# Any CLI flag accepted by autoforge.py can be placed here.
DEFAULT_ARGS: dict[str, object] = {
    "--csv_file": "../../../bambulab.csv",
    "--iterations": 6000,
    "--num_init_rounds": 8,
    "--stl_output_size": 50,
    "--disable_visualization_for_gradio": 1,
    "--warmup_fraction": 1.0,
    "--fast_pruning_percent": 0.05,
    "--learning_rate_warmup_fraction": 0.01,
}

SWEEP_PARAM = "--learning_rate"  # param to overwrite per run
SWEEP_VALUES = [
    0.01,
    0.02,
    0.03,
]  # ,0.04,0.05,0.06,0.07,0.08,0.09,0.1]  # values to sweep over

IMAGES_DIR = Path("/home/scsadmin/AutoForge/images/test_images")
BASE_OUTPUT_DIR = Path("output_grid")  # all run folders are created inside here
MAX_WORKERS = 16  # parallel jobs
# ---------- end editable section ------ #


def make_cmd(image_path: Path, param_value, run_dir: Path, idx: int) -> list[str]:
    """Assemble the command line for one run."""
    cmd: list[str] = [
        sys.executable,
        "../auto_forge.py",
        "--input_image",
        str(image_path),
        "--output_folder",
        str(run_dir),
        "--random_seed",
        str(idx + 1),
    ]

    # default args
    for flag, val in DEFAULT_ARGS.items():
        if isinstance(val, bool):
            if val:  # store_true flag
                cmd.append(flag)
        else:
            cmd.extend([flag, str(val)])

    # swept parameter
    cmd.extend([SWEEP_PARAM, str(param_value)])
    return cmd


def run_single(image_path: Path, param_value, idx) -> dict:
    """Worker: launch subprocess, parse loss, return result dict."""
    run_dir = BASE_OUTPUT_DIR / (
        f"{image_path.stem}_{idx}_{SWEEP_PARAM.lstrip('-')}={param_value}"
    )
    run_dir.mkdir(parents=True, exist_ok=True)

    cmd = make_cmd(image_path, param_value, run_dir, idx)

    proc = subprocess.run(cmd, capture_output=True, text=True)
    # print output
    if proc.returncode != 0:
        print(f"Running: {' '.join(cmd)}")
        print(f"Return code: {proc.returncode}")
        print(f"Stdout: {proc.stdout.strip()}")
        print(f"Stderr: {proc.stderr.strip()}")

    loss_file = run_dir / "final_loss.txt"
    loss = None
    if loss_file.exists():
        try:
            loss = float(loss_file.read_text().strip())
        except ValueError:
            pass

    return {
        "image": str(image_path),
        "param_value": param_value,
        "loss": loss,
        "output_folder": str(run_dir),
        "returncode": proc.returncode,
        "stdout": proc.stdout,
        "stderr": proc.stderr,
    }


def main():
    values = SWEEP_VALUES

    images = [
        p for p in IMAGES_DIR.iterdir() if p.suffix.lower() in {".png", ".jpg", ".jpeg"}
    ]

    BASE_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    ts = strftime("%Y%m%d_%H%M%S")
    # run grid
    results = []
    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futures = {
            pool.submit(run_single, img, val, idx): (img, val)
            for idx in range(1)
            for img, val in product(images, values)
        }
        for fut in tqdm(as_completed(futures), total=len(futures), desc="Running grid"):
            res = fut.result()
            results.append(res)
            print(
                f"✓ finished {Path(res['image']).name} "
                f"{SWEEP_PARAM}={res['param_value']} loss={res['loss']}"
            )

            # write summary

            out_path = BASE_OUTPUT_DIR / f"out_dict_{ts}.json"
            with open(out_path, "w", encoding="utf-8") as fp:
                json.dump(results, fp, indent=2)
    print(f"Results saved to {out_path}")


if __name__ == "__main__":
    main()
