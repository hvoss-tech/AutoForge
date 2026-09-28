#!/usr/bin/env python3
"""Speed loop harness: full pipeline (optimization + all pruning + export) at
150mm, 4000 iterations, three runs (seeds 42, 7, 123) run one after another
(never in parallel, so they don't share the GPU). Reports the mean wall time,
the peak VRAM and the mean loss, and optionally appends a row to
results_speed_n.tsv.

    python benchmarks/speed_n_harness.py --tag base --log "baseline" [--discard no] [-- extra flags]

Columns of results_speed_n.tsv:
    git commit, discard, execution (ms), vram peak (MB), avg loss, description
(the mean loss and its change against the baseline go into the description).
"""
import argparse
import json
import os
import subprocess
import sys
import time

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
TSV = os.path.join(ROOT, "results_speed_n.tsv")
OUT = os.environ.get("HARNESS_OUT", os.path.join(ROOT, "bench_output", "speed_n"))
SEEDS = [42, 7, 123]
BASE_FLAGS = [
    "--input_image", "input.png", "--csv_file", "bambulab.csv",
    "--stl_output_size", "150", "--iterations", "4000",
    "--perform_pruning", "--no-visualize", "--disable_visualization_for_gradio", "1",
]


def run_one(tag, seed, extra, reuse=False):
    out = os.path.join(OUT, f"{tag}_s{seed}")
    cached = os.path.join(out, "result.json")
    if reuse and os.path.exists(cached):
        # A screening run of the same code (see --reuse).
        with open(cached) as f:
            return json.load(f)
    cmd = [sys.executable, os.path.join(ROOT, "benchmarks", "run_experiment.py"), *BASE_FLAGS,
           "--output_folder", out, "--random_seed", str(seed), *extra]
    t = time.perf_counter()
    p = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    wall = time.perf_counter() - t
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "log.txt"), "w") as f:
        f.write(p.stdout + "\n--- stderr ---\n" + p.stderr)
    for line in p.stdout.splitlines():
        if line.startswith("EXPERIMENT_RESULT "):
            r = json.loads(line[len("EXPERIMENT_RESULT "):])
            r["proc_wall_s"] = wall
            with open(cached, "w") as f:
                json.dump(r, f)
            return r
    raise RuntimeError(f"run {tag} seed {seed} failed:\n{p.stderr[-3000:]}")


def baseline_loss():
    """Mean loss of the first row (the baseline), parsed from its description."""
    if not os.path.exists(TSV):
        return None
    with open(TSV) as f:
        rows = [l.rstrip("\n").split("\t") for l in f][1:]
    for r in rows:
        if "loss=" in r[-1]:
            return float(r[-1].split("loss=")[1].split()[0])
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--log", default=None, help="description; appends a row to results_speed_n.tsv")
    ap.add_argument("--discard", default="no")
    ap.add_argument("--seeds", default=",".join(map(str, SEEDS)))
    ap.add_argument("--reuse", action="store_true",
                    help="reuse this tag's finished runs (e.g. the one-seed screening run) instead of re-running them")
    ap.add_argument("extra", nargs=argparse.REMAINDER)
    a = ap.parse_args()
    extra = [x for x in a.extra if x != "--"]
    res = []
    for s in (int(x) for x in a.seeds.split(",")):
        r = run_one(a.tag, s, extra, a.reuse)
        print(f"seed {s}: {r['elapsed_s']*1000:.0f} ms  loss {r['loss']:.4f}  "
              f"reserved {r['peak_vram_reserved_gb']*1000:.0f} MB  smi {r['peak_nvidia_smi_gb']*1000:.0f} MB", flush=True)
        res.append(r)
    ms = sum(r["elapsed_s"] for r in res) / len(res) * 1000
    loss = sum(r["loss"] for r in res) / len(res)
    vram_res = max(r["peak_vram_reserved_gb"] for r in res) * 1000
    vram_smi = max(r["peak_nvidia_smi_gb"] for r in res) * 1000
    base = baseline_loss()
    rel = f" ({(loss / base - 1) * 100:+.2f}% vs base)" if base else ""
    print(f"SUMMARY {a.tag}: {ms:.0f} ms  loss={loss:.4f}{rel}  reserved {vram_res:.0f} MB  smi {vram_smi:.0f} MB  "
          f"losses={[round(r['loss'], 3) for r in res]}  times={[round(r['elapsed_s'], 1) for r in res]}")
    if a.log is not None:
        commit = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, text=True).strip()
        new = not os.path.exists(TSV)
        with open(TSV, "a") as f:
            if new:
                f.write("git commit\tdiscard\texecution (ms)\tvram peak (MB)\tavg loss\tdescription\n")
            desc = (f"{a.log.replace(chr(9), ' ')} | loss={loss:.4f}{rel} smi_peak={vram_smi:.0f}MB "
                    f"times={[round(r['elapsed_s'], 1) for r in res]}")
            f.write(f"{commit}\t{a.discard}\t{ms:.0f}\t{vram_res:.0f}\t{loss:.4f}\t{desc}\n")


if __name__ == "__main__":
    main()
