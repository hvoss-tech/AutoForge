"""auto_forge.py

High-level orchestration module for the AutoForge optimization pipeline.

Responsibilities:
- Parse CLI / config file arguments.
- Load image and material properties.
- (Optionally) auto-select a background filament color based on dominant image color.
- Initialize a height map using one of several strategies (k-means clustering or depth estimation).
- Build and run the filament optimization loop (differentiable + periodic discretization checks).
- Optionally prune the solution to respect practical printer constraints (materials, swaps, layers).
- Export final artifacts: preview PNG, STL(s), swap instructions, project file, metadata.

The implementation intentionally keeps side-effects (disk writes / prints) order-stable to
preserve prior behavior. Helper functions are factored out for readability; no functional
behavior should have changed relative to the previous monolithic version.
"""

import argparse
import copy
import json
import sys
import os
import shutil
import traceback
from typing import Optional, Tuple, List

# Must be set before the CUDA context is created (i.e. before any actual
# CUDA op - `import torch` alone doesn't trigger this). PyTorch's default
# caching allocator can end up reserving noticeably more driver memory than
# what's actually live at the peak, because it services allocation requests
# from same-sized "segments" and can't split/merge across their boundaries -
# with the mixed transient tensor sizes this pipeline creates (see
# composite_image_cont/_disc), that fragmentation is significant. Measured
# on this repo's benchmark (stl_output_size=150, 500 iters): peak reserved
# 2.10GB -> 1.46GB, peak nvidia-smi-reported process VRAM 2.28GB -> 1.67GB,
# with identical loss - pure allocator behavior, zero precision/algorithm
# impact. setdefault() so a user's own explicit setting always wins.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import configargparse
import cv2
import torch
import numpy as np
from tqdm import tqdm

from autoforge.Helper import PruningHelper
from autoforge.Helper.AmpUtils import safe_autocast
from autoforge.Helper.DeviceUtils import empty_cache
from autoforge.Helper.FilamentHelper import hex_to_rgb, load_materials
from autoforge.Helper.Heightmaps.FastTSPHeightMap import (
    run_init_threads,
)

from autoforge.Helper.ImageHelper import image_to_uint8, resize_image, imread, imwrite, to_bgr_or_bgra_uint8
from autoforge.Helper.OtherHelper import set_seed, perform_basic_check, get_device
from autoforge.Helper.OutputHelper import (
    generate_stl,
    generate_swap_instructions,
    generate_project_file,
    generate_flatforge_stls,
    model_size_mm,
)
from autoforge.Modules.Optimizer import FilamentOptimizer

# TorchScript's tensor-expression fuser compiles a fused kernel the first
# time each scripted function sees a new input signature. The big [L,H,W]
# composites are Triton kernels now (see FusedComposite), so what is left
# for it is small per-layer chains, where the first-call compiles (seconds
# each, dozens of signatures across training and pruning) cost far more than
# the fused kernels ever save. AF_TEXPR_FUSER=1 turns it back on.
if os.environ.get("AF_TEXPR_FUSER", "0") != "1":
    torch._C._jit_set_texpr_fuser_enabled(False)

# No version check: comparing version strings sorted "10.0" below "2.0", and
# every supported torch has this call (the try covers one that doesn't).
try:
    torch.set_float32_matmul_precision("high")
except Exception as e:
    print("Warning: Could not set float32 matmul precision to high. Error:", e)


def parse_args() -> argparse.Namespace:
    """Create and parse command-line & config-file arguments.

    Returns:
        argparse.Namespace: Populated arguments structure. Some parameters may be adjusted later
        (e.g., num_init_cluster_layers when -1 to infer from max_layers).
    """
    return build_parser().parse_args()


def cli_defaults() -> dict:
    """Every CLI option's default, by its attribute name. The webui builds
    its run settings on top of these, so a new CLI option (and its default)
    reaches webui runs too instead of falling back to whatever getattr
    default the code that reads it happens to use."""
    return {
        action.dest: action.default
        for action in build_parser()._actions
        if action.dest not in ("help", "config") and action.default is not argparse.SUPPRESS
    }


def build_parser() -> configargparse.ArgParser:
    """The CLI's argument parser (see parse_args)."""
    parser = configargparse.ArgParser()
    parser.add_argument("--config", is_config_file=True, help="Path to config file")

    parser.add_argument(
        "--input_image", type=str, required=True, help="Path to input image"
    )
    parser.add_argument(
        "--csv_file",
        type=str,
        default="",
        help="Path to CSV file with material data",
    )
    parser.add_argument(
        "--json_file",
        type=str,
        default="",
        help="Path to json file with material data",
    )
    parser.add_argument(
        "--output_folder", type=str, default="output", help="Folder to write outputs"
    )

    parser.add_argument(
        "--iterations", type=int, default=6000, help="Number of optimization iterations"
    )

    parser.add_argument(
        "--warmup_fraction",
        type=float,
        default=1.0,
        help="Fraction of iterations for keeping the tau at the initial value",
    )

    parser.add_argument(
        "--learning_rate_warmup_fraction",
        type=float,
        default=0.01,
        help="Fraction of iterations that the learning rate is increasing (warmup)",
    )

    parser.add_argument(
        "--init_tau",
        type=float,
        default=1.0,
        help="Initial tau value for Gumbel-Softmax",
    )

    parser.add_argument(
        "--final_tau",
        type=float,
        default=0.01,
        help="Final tau value for Gumbel-Softmax",
    )

    parser.add_argument(
        "--learning_rate",
        type=float,
        default=0.015,
        help="Learning rate for optimization",
    )

    parser.add_argument(
        "--layer_height", type=float, default=0.04, help="Layer thickness in mm"
    )

    parser.add_argument(
        "--max_layers", type=int, default=75, help="Maximum number of layers"
    )

    parser.add_argument(
        "--min_layers",
        type=int,
        default=0,
        help="Minimum number of layers. Used for pruning.",
    )

    parser.add_argument(
        "--background_height",
        type=float,
        default=0.24,
        help="Height of the background in mm",
    )

    parser.add_argument(
        "--background_color", type=str, default="#000000", help="Background color"
    )

    parser.add_argument(
        "--auto_background_color",
        default=True,
        help="Automatically set background color to the closest filament color matching the dominant image color. Overrides --background_color.",
        action=argparse.BooleanOptionalAction,
    )

    parser.add_argument(
        "--visualize",
        default=True,
        help="Enable visualization during optimization",
        action=argparse.BooleanOptionalAction,
    )

    # Instead of an output_size parameter, we use stl_output_size and nozzle_diameter.
    parser.add_argument(
        "--stl_output_size",
        type=int,
        default=150,
        help="Size of the longest dimension of the output STL file in mm",
    )

    parser.add_argument(
        "--processing_reduction_factor",
        type=int,
        default=2,
        help="Reduction factor for reducing the processing size compared to the output size (default: 2 - half resolution)",
    )

    parser.add_argument(
        "--nozzle_diameter",
        type=float,
        default=0.4,
        help="Diameter of the printer nozzle in mm (details smaller than half this value will be ignored)",
    )

    parser.add_argument(
        "--delta_grid",
        type=int,
        default=4,
        help="Downscale factor of the trained smooth per-pixel height field (0 = off)",
    )
    parser.add_argument(
        "--height_assign",
        default=True,
        help="At every discrete check, re-assign every pixel's height exactly for the current layer stack "
        "(smoothed per-pixel assignment); the gradient steps train the materials in between",
        action=argparse.BooleanOptionalAction,
    )
    parser.add_argument(
        "--height_assign_smoothness",
        type=float,
        default=2.0,
        help="--height_assign: cost per layer of height difference to each neighbour, traded against colour error",
    )
    parser.add_argument(
        "--early_stopping",
        type=int,
        default=3000,
        help="Number of steps without improvement before stopping",
    )

    parser.add_argument(
        "--perform_pruning",
        default=True,
        help="Perform pruning after optimization",
        action=argparse.BooleanOptionalAction,
    )

    parser.add_argument(
        "--fast_pruning",
        default=True,
        help="Use fast pruning method",
        action=argparse.BooleanOptionalAction,
    )
    parser.add_argument(
        "--fast_pruning_percent",
        type=float,
        default=0.25,
        help="Percentage of increment search for fast pruning",
    )

    parser.add_argument(
        "--spike_removal",
        default=True,
        help="Enable spike removal on the discrete height map before/after pruning to smooth isolated tall pixels",
        action=argparse.BooleanOptionalAction,
    )
    parser.add_argument(
        "--spike_threshold_layers",
        type=int,
        default=1,
        help="Minimum layer delta above the local median to treat a pixel as a spike in a 3x3 window",
    )

    parser.add_argument(
        "--spike_removal_passes",
        type=int,
        default=4,
        help="Number of spike removal passes (4 approximates old BFS; higher = smoother but slower)",
    )

    parser.add_argument(
        "--pixel_height_refine",
        default=True,
        help="After pruning, refine every pixel's height individually by exact coordinate descent on the discrete loss (spike-aware after spike removal)",
        action=argparse.BooleanOptionalAction,
    )
    parser.add_argument(
        "--stack_search",
        default=True,
        help="With --pixel_height_refine: search the layer materials under a free-per-pixel-height palette proxy and keep the new stack if it lowers the real loss",
        action=argparse.BooleanOptionalAction,
    )
    parser.add_argument(
        "--stack_search_rounds",
        type=int,
        default=60,
        help="Maximum number of perturb-and-descend rounds for --stack_search",
    )
    parser.add_argument(
        "--edge_bleed",
        type=float,
        default=0.25,
        help="Edge bleed strength (0-1): every layer's presence at a pixel blends with the mean of its 8 neighbours' by "
        "this weight, so at a height step the taller side thins and shows its lower layers and the lower side takes on "
        "some of the taller one's (0 = no bleed)",
    )
    parser.add_argument(
        "--pixel_height_smoothness",
        type=float,
        default=1.0,
        help="Pixel height refine: cost per layer of height difference to each neighbour, traded against colour error (keeps the height map printable; 0 = colour only)",
    )
    parser.add_argument(
        "--pixel_height_anchor",
        type=float,
        default=0.0,
        help="Pixel height refine: quadratic pull toward the height map pruning started from (0 = off)",
    )
    parser.add_argument(
        "--pixel_height_refine_radius",
        type=int,
        default=3,
        help="Candidate heights for --pixel_height_refine: -1 tries every height; r > 0 tries current +-r plus each pixel's free-height palette optimum (much faster)",
    )
    parser.add_argument(
        "--stack_search_patience",
        type=int,
        default=15,
        help="Stop the --stack_search after this many rounds without improvement",
    )
    parser.add_argument(
        "--pixel_height_refine_sweeps",
        type=int,
        default=2,
        help="Number of sweeps over the image for --pixel_height_refine",
    )

    parser.add_argument(
        "--pruning_max_colors",
        type=int,
        default=100,
        help="Max number of colors allowed after pruning",
    )
    parser.add_argument(
        "--pruning_max_swaps",
        type=int,
        default=100,
        help="Max number of swaps allowed after pruning",
    )

    parser.add_argument(
        "--optimize_background",
        default=True,
        help="With --auto_background_color: the optimizer chooses the base filament like any layer's, starting from the filament closest to the image's dominant color (the base keeps its --background_height)",
        action=argparse.BooleanOptionalAction,
    )

    parser.add_argument(
        "--spike_refine_blocks",
        type=str,
        default="1212121",
        help="Block sizes of the spike-aware pixel height refine passes after spike removal (digits, e.g. 121)",
    )
    parser.add_argument(
        "--stack_history_candidates",
        type=int,
        default=6,
        help="Besides the best stack of the search, also height-refine this many of its last improvements and keep whichever gives the lowest real loss",
    )
    parser.add_argument(
        "--layer_material_refine_window",
        type=int,
        default=2,
        help="After the spike-aware refine, re-pick each layer's material among those within this many layers of it under the real loss (heights fixed), then refine the heights again; 0 disables",
    )
    parser.add_argument(
        "--plateau_refine",
        default=True,
        help="After the spike-aware refine, move whole equal-height plateaus (region moves) under the real loss",
        action=argparse.BooleanOptionalAction,
    )
    parser.add_argument(
        "--plateau_refine_target",
        default=False,
        action=argparse.BooleanOptionalAction,
        help="Also region moves over connected areas of similar target colour",
    )
    parser.add_argument("--plateau_refine_passes", type=int, default=1)
    parser.add_argument(
        "--skip_legacy_prune",
        default=False,
        help="Skip the greedy colour/swap/layer pruning phases when the solution is already within the limits",
        action=argparse.BooleanOptionalAction,
    )
    parser.add_argument(
        "--palette_search",
        default=False,
        help="After the stack search, also try global palette moves (replace a material everywhere) and keep the stack if the real loss improves",
        action=argparse.BooleanOptionalAction,
    )

    parser.add_argument(
        "--constrained_opt",
        default=False,
        help="Hold --pruning_max_colors / --pruning_max_swaps during the optimization itself instead of pruning afterwards",
        action=argparse.BooleanOptionalAction,
    )
    parser.add_argument(
        "--constraint_rho",
        type=float,
        default=0.0,
        help="--constrained_opt: final weight of the pull toward the nearest stack within the limits",
    )
    parser.add_argument(
        "--constraint_start",
        type=float,
        default=0.1,
        help="--constrained_opt: fraction of the iterations after which the pull starts ramping up",
    )
    parser.add_argument(
        "--constraint_full",
        type=float,
        default=0.6,
        help="--constrained_opt: fraction of the iterations at which the pull reaches --constraint_rho",
    )

    parser.add_argument(
        "--constraint_swap_margin",
        type=float,
        default=1.0,
        help="--constrained_opt: the expected swaps are held at this fraction of the swap limit",
    )
    parser.add_argument(
        "--constraint_dual_lr",
        type=float,
        default=0.01,
        help="--constrained_opt: dual ascent step for the multipliers on the expected colour/swap counts",
    )
    parser.add_argument(
        "--constraint_linear_moves",
        default=False,
        help="--constrained_opt: add linearized (Frank-Wolfe) moves to the local search",
        action=argparse.BooleanOptionalAction,
    )

    parser.add_argument(
        "--constraint_pin",
        default=True,
        help="--constrained_opt: in the final phase train the heights on the best feasible stack (pinned), alternating with the stack search",
        action=argparse.BooleanOptionalAction,
    )

    parser.add_argument(
        "--constraint_search_rounds",
        type=int,
        default=3,
        help="--constrained_opt: rounds of feasible local search on the best stack at each discrete check once the pull is at full strength",
    )

    parser.add_argument(
        "--intermediate_search_interval",
        type=int,
        default=1000,
        help="Every N training steps run a short layer-stack search (within the colour/swap limits) and a per-pixel "
        "height refine on the best solution, and continue training from the result (0 = off)",
    )
    parser.add_argument(
        "--intermediate_search_start",
        type=int,
        default=2000,
        help="--intermediate_search_interval: first step of the searches (the early steps explore; 0 = first at the interval)",
    )
    parser.add_argument(
        "--intermediate_search_rounds",
        type=int,
        default=15,
        help="--intermediate_search_interval: maximum perturb-and-descend rounds of each stack search",
    )
    parser.add_argument(
        "--intermediate_search_patience",
        type=int,
        default=5,
        help="--intermediate_search_interval: stop each stack search after this many rounds without improvement",
    )
    parser.add_argument(
        "--intermediate_search_pin",
        type=float,
        default=0.0,
        help="--intermediate_search_interval: logit margin (+-value) the found stack is written back into training "
        "with; 0 leaves the stack logits to training and hands back only the heights",
    )

    parser.add_argument(
        "--prune_sweep",
        type=str,
        default="",
        help="Experiment mode (with --minimal_postprocess): prune the trained solution to each 'colors:swaps' limit in this comma separated list",
    )

    parser.add_argument(
        "--minimal_postprocess",
        default=False,
        help="Experiment mode: after training skip every refinement step; with --perform_pruning only the colour and swap pruning run",
        action=argparse.BooleanOptionalAction,
    )

    parser.add_argument(
        "--pruning_max_layer",
        type=int,
        default=75,
        help="Max number of layers allowed after pruning",
    )

    parser.add_argument(
        "--pruning_batch_size",
        type=int,
        default=8,
        help="Number of pruning candidates to evaluate per batch (0 = disabled, 8 = good default, higher = faster but slightly more VRAM used)",
    )

    parser.add_argument(
        "--cuda_graph",
        default=True,
        action=argparse.BooleanOptionalAction,
        help="Capture the training forward/backward into a CUDA/HIP graph "
        "(NVIDIA and AMD GPUs only; ignored on Apple Metal and CPU). "
        "Large speedup for the launch-overhead-bound training loop; "
        "--no-cuda_graph falls back to plain eager steps.",
    )

    parser.add_argument(
        "--random_seed",
        type=int,
        default=0,
        help="Specify the random seed, or use 0 for automatic generation",
    )

    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Torch device to run on, e.g. 'cuda', 'cuda:1', 'mps', 'cpu'. "
        "Defaults to auto-detection (CUDA/ROCm, then Apple Metal, then CPU). "
        "Can also be set with the AUTOFORGE_DEVICE environment variable.",
    )

    parser.add_argument(
        "--mps",
        action="store_true",
        help="Deprecated: Apple Metal (MPS) is now detected automatically. "
        "Kept only to force MPS on a machine that also exposes a CUDA GPU; "
        "prefer --device mps.",
    )

    parser.add_argument(
        "--run_name", type=str, help="Name of the run used for TensorBoard logging"
    )

    parser.add_argument(
        "--tensorboard", action="store_true", help="Enable TensorBoard logging"
    )

    parser.add_argument(
        "--num_init_rounds",
        type=int,
        default=1,
        # Historically defaulted to 64, intended as a "best of N random
        # restarts" search over the kmeans heightmap init. As currently
        # implemented, additional rounds are pure overhead with zero effect
        # on the result: the expensive over-clustering (Stage 1) is shared
        # across all rounds already, the weighted-KMeans refinement (Stage
        # 2, in _refine_clusters) is called with a hardcoded
        # random_state=0 regardless of round, and the cluster ordering
        # (tsp_order_mst_path) is a fully deterministic algorithm - so
        # every round produces byte-identical clustering/ordering. The only
        # thing that varies per round is the pixel *sample* used for the
        # silhouette quality estimate, and the final round selection
        # (`min(results, key=lambda x: x[2])`) doesn't even use that
        # silhouette score, only the (round-invariant) ordering metric.
        # Verified empirically (2 different --random_seed values,
        # stl_output_size=150): rounds=1 produces bit-identical loss to the
        # old rounds=64 default while being significantly faster. If the
        # random_state=0 in _refine_clusters is ever intentionally
        # unfixed to restore real per-round diversity, this default should
        # be revisited.
        help="Number of rounds to choose the starting height map from.",
    )

    parser.add_argument(
        "--num_init_threads",
        type=int,
        default=4,
        help="Number of threads to use during the init heightmap search.",
    )

    parser.add_argument(
        "--num_init_cluster_layers",
        type=int,
        default=-1,
        help="Number of layers to cluster the image into.",
    )

    parser.add_argument(
        "--disable_visualization_for_gradio",
        type=int,
        default=0,
        help="Simple switch to disable the matplotlib render window for gradio rendering.",
    )

    parser.add_argument(
        "--best_of",
        type=int,
        default=1,
        help="Run the program multiple times and output the best result.",
    )

    parser.add_argument(
        "--discrete_check",
        type=int,
        default=100,
        help="Modulo how often to check for new discrete results.",
    )

    parser.add_argument(
        "--flatforge",
        default=False,
        help="Enable FlatForge mode to generate separate STL files for each color",
        action=argparse.BooleanOptionalAction,
    )

    parser.add_argument(
        "--cap_layers",
        type=int,
        default=0,
        help="Number of complete clear/transparent layers to add on top in FlatForge mode",
    )

    # New: choose heightmap initializer
    parser.add_argument(
        "--init_heightmap_method",
        type=str,
        choices=["kmeans", "depth"],
        default="kmeans",
        help="Initializer for the height map: 'kmeans' (fast, default) or 'depth' (requires transformers).",
    )
    # New priority mask argument (optional)
    parser.add_argument(
        "--priority_mask",
        type=str,
        default="",
        help="Optional path to a priority mask image (same dimensions as input image). Non-empty: apply weighted loss (0.1 outside, 1.0 at max inside).",
    )
    parser.add_argument(
        "--priority_mask_strength",
        type=float,
        default=10.0,
        help="How many times more a fully white pixel of --priority_mask counts than a black one (>= 1; default 10).",
    )

    return parser


def _compute_dominant_image_color(
    img_rgb: np.ndarray, alpha: Optional[np.ndarray]
) -> Optional[Tuple[str, np.ndarray]]:
    """Compute an approximate dominant color of the input image.

    Strategy:
    - Optionally downscale very large images for efficiency.
    - Ignore (mostly) transparent pixels if alpha channel is provided.
    - Use frequency counts (np.unique) over exact RGB triplets.

    Args:
        img_rgb: Image array in RGB order (H,W,3) uint8.
        alpha: Optional alpha mask (H,W,1) or (H,W) uint8; pixels <128 are ignored.

    Returns:
        (hex_color, normalized_rgb) where hex_color is a '#RRGGBB' string and normalized_rgb
        is float32 in [0,1]^3. Returns None if no valid pixels remain.
    """
    try:
        # Downscale if needed (max side 300 px)
        h, w = img_rgb.shape[:2]
        max_side = max(h, w)
        target_side = 300
        alpha_small: Optional[np.ndarray] = None
        if max_side > target_side:
            scale = target_side / max_side
            new_w = max(1, int(w * scale))
            new_h = max(1, int(h * scale))
            img_small = cv2.resize(
                img_rgb, (new_w, new_h), interpolation=cv2.INTER_AREA
            )
            if alpha is not None:
                alpha_small = cv2.resize(
                    alpha, (new_w, new_h), interpolation=cv2.INTER_NEAREST
                )
        else:
            img_small = img_rgb
            alpha_small = alpha
        # Build mask for valid pixels (ignore transparent)
        if alpha_small is not None:
            valid_mask = (
                alpha_small[..., 0] if alpha_small.ndim == 3 else alpha_small
            ) >= 128
        else:
            valid_mask = np.ones(img_small.shape[:2], dtype=bool)
        if valid_mask.sum() == 0:
            return None
        pixels = img_small[valid_mask]
        # Use np.unique to find most frequent RGB triplet
        unique_colors, counts = np.unique(
            pixels.reshape(-1, 3), axis=0, return_counts=True
        )
        idx = int(np.argmax(counts))
        dom_rgb_uint8 = unique_colors[idx]
        dom_rgb_norm = dom_rgb_uint8.astype(np.float32) / 255.0
        hex_color = "#" + "".join(f"{c:02X}" for c in dom_rgb_uint8)
        return hex_color, dom_rgb_norm
    except Exception:
        traceback.print_exc()
        return None


def _auto_select_background_color(
    args,
    img_rgb: np.ndarray,
    alpha: Optional[np.ndarray],
    material_colors_np: np.ndarray,
    material_names: List[str],
    colors_list: List[str],
) -> None:
    """Optionally override the user-provided background color with a closest material color.

    When --auto_background_color is set:
    - Determine dominant image color (ignoring transparency).
    - Find closest filament (Euclidean in normalized RGB).
    - Persist metadata to 'auto_background_color.txt'.

    Side effects: Mutates args.background_color and attaches background_material_* fields.

    Args:
        args: Global argument namespace (mutated).
        img_rgb: Full-resolution RGB image (uint8).
        alpha: Optional alpha channel for transparency filtering.
        material_colors_np: (N,3) array of filament RGB colors in [0,1].
        material_names: List of filament names.
        colors_list: List of filament hex color strings (#RRGGBB).
    """
    if not args.auto_background_color:
        return
    res = _compute_dominant_image_color(img_rgb, alpha)
    if res is not None:
        dominant_hex, dominant_rgb = res
        diffs = material_colors_np - dominant_rgb[None, :]
        dists = np.linalg.norm(diffs, axis=1)
        closest_idx = int(np.argmin(dists))
        chosen_hex = colors_list[closest_idx]
        print(
            f"Auto background color: dominant image color {dominant_hex} -> closest filament {chosen_hex} (index {closest_idx})."
        )
        args.background_color = chosen_hex
        args.background_material_index = closest_idx
        try:
            args.background_material_name = material_names[closest_idx]
        except Exception:
            args.background_material_name = None
        try:
            with open(
                os.path.join(args.output_folder, "auto_background_color.txt"), "w", encoding="utf-8"
            ) as f:
                f.write(f"dominant_image_color={dominant_hex}\n")
                f.write(f"chosen_filament_color={chosen_hex}\n")
                f.write(f"closest_filament_index={closest_idx}\n")
                if getattr(args, "background_material_name", None):
                    f.write(f"closest_filament_name={args.background_material_name}\n")
        except Exception:
            traceback.print_exc()
    else:
        print(
            "Warning: Auto background color computation failed; using provided --background_color."
        )


def _prepare_background_and_materials(
    args,
    device: torch.device,
    material_colors_np: np.ndarray,
    material_TDs_np: np.ndarray,
) -> Tuple[Tuple[float, float, float], torch.Tensor, torch.Tensor, torch.Tensor]:
    """Create torch tensors for materials & background color.

    Args:
        args: Global arguments (uses background_color hex string).
        device: Torch device for tensor placement.
        material_colors_np: (N,3) float32 array in [0,1].
        material_TDs_np: (N,*) array of material transmission / diffusion parameters.

    Returns:
        (rgb_tuple, background_tensor, material_colors_tensor, material_TDs_tensor)
    """
    rgb_tuple = tuple(hex_to_rgb(args.background_color))
    background = torch.tensor(rgb_tuple, dtype=torch.float32, device=device)
    material_colors = torch.tensor(
        material_colors_np, dtype=torch.float32, device=device
    )
    material_TDs = torch.tensor(material_TDs_np, dtype=torch.float32, device=device)
    return rgb_tuple, background, material_colors, material_TDs


def _compute_pixel_sizes(args) -> Tuple[int, int]:
    """Derive pixel dimensions for solving vs. output STL size.

    We oversample relative to nozzle_diameter to capture detail, then optionally downscale
    for the differentiable optimization pass.

    Returns:
        (computed_output_size, computed_processing_size)
    """
    computed_output_size = int(round(args.stl_output_size * 2 / args.nozzle_diameter))
    reduced_size = int(round(computed_output_size / args.processing_reduction_factor))
    # Below MIN_PROCESSING_SIZE, applying the reduction factor barely moves
    # the needle on training cost (the image is already tiny) but throws away
    # a large fraction of the already-small pixel budget, which measurably
    # hurts final loss - verified empirically (stl_output_size=20,
    # nozzle=0.4, reduction_factor=2: processing_size 50->100 improved mean
    # loss ~30% across 4 seeds for ~0% time/VRAM cost). Clamping only kicks
    # in for small outputs; anything already at/above this resolution
    # (e.g. the default stl_output_size=150 case, processing_size=375) is
    # completely unaffected.
    MIN_PROCESSING_SIZE = 100
    computed_processing_size = min(
        computed_output_size, max(MIN_PROCESSING_SIZE, reduced_size)
    )
    print(f"Computed solving pixel size: {computed_output_size}")
    return computed_output_size, computed_processing_size


def _load_priority_mask(
    args, output_img_np: np.ndarray, device: torch.device
) -> Optional[torch.Tensor]:
    """Load and resize a priority / focus mask if provided.

    The mask scales heights during initialization and can later weight loss terms.

    Behavior:
    - Reads image; converts RGBA/RGB to grayscale.
    - Resizes to full-resolution output size.
    - Persists a diagnostic PNG after normalization.

    Returns:
        focus_map_full: Float32 tensor (H,W) in [0,1] or None if no mask provided.
    """
    focus_map_full = None
    if args.priority_mask != "":
        # 8-bit first: a 16-bit mask divided by 255 below gave weights up
        # to 257 instead of 0..1.
        pm = to_bgr_or_bgra_uint8(imread(args.priority_mask, cv2.IMREAD_UNCHANGED))
        pm = cv2.cvtColor(pm[:, :, :3], cv2.COLOR_BGR2GRAY)
        tgt_h, tgt_w = output_img_np.shape[:2]
        pm_resized = cv2.resize(pm, (tgt_w, tgt_h), interpolation=cv2.INTER_LINEAR)
        pm_float = pm_resized.astype(np.float32) / 255.0
        focus_map_full = torch.tensor(pm_float, dtype=torch.float32, device=device)
        # ImageHelper.imwrite: cv2.imwrite writes nothing to a non-ASCII
        # path on Windows.
        imwrite(
            os.path.join(args.output_folder, "priority_mask_resized.png"),
            (pm_float * 255).astype(np.uint8),
        )
        focus_map_full = focus_map_full * priority_mask_scale(
            getattr(args, "priority_mask_strength", DEFAULT_PRIORITY_MASK_STRENGTH)
        )
    return focus_map_full


DEFAULT_PRIORITY_MASK_STRENGTH = 10.0


def priority_mask_scale(strength: float) -> float:
    """Factor for a 0..1 priority mask so a fully masked pixel counts
    `strength` times an unmasked one under compute_loss's weighting
    (0.1 + 0.9 * mask): (0.1 + 0.9 * k) / 0.1 = strength. The default
    strength of 10 gives exactly 1, i.e. the mask as loaded."""
    strength = max(1.0, float(strength))
    return (strength - 1.0) / 9.0


def _initialize_heightmap(
    args,
    output_img_np: np.ndarray,
    bg_rgb: Tuple[float, float, float],
    material_colors_np: np.ndarray,
    random_seed: int,
    progress=None,
) -> Tuple[np.ndarray, Optional[np.ndarray], np.ndarray]:
    """Initialize the height map logits & labels using selected method.

    Methods:
        depth  : Uses an external depth estimation model (requires transformers).
        kmeans : Clusters pixel colors into layer assignments (default).

    Returns:
        pixel_height_logits_init: (H,W) float32 numpy array of raw logits.
        global_logits_init     : (L,*) global logits array or None (depth variant may not use it).
        pixel_height_labels    : (H,W) int array of discrete initial layer indices.
    """
    print("Initalizing height map. This can take a moment...")
    if args.init_heightmap_method == "depth":
        try:
            from autoforge.Helper.Heightmaps.DepthEstimateHeightMap import (
                init_height_map_depth_color_adjusted,
            )
        except Exception:
            print(
                "Error: depth initializer requested but could not be imported. Install 'transformers' and try again.",
                file=sys.stderr,
            )
            raise
        pixel_height_logits_init, pixel_height_labels = (
            init_height_map_depth_color_adjusted(
                output_img_np,
                args.max_layers,
                random_seed=random_seed,
                focus_map=None,
            )
        )
        global_logits_init = None
    else:
        pixel_height_logits_init, global_logits_init, pixel_height_labels = (
            run_init_threads(
                output_img_np,
                args.max_layers,
                args.layer_height,
                bg_rgb,
                random_seed=random_seed,
                num_threads=args.num_init_threads,
                init_method="kmeans",
                cluster_layers=args.num_init_cluster_layers,
                material_colors=material_colors_np,
                focus_map=None,
                num_runs=args.num_init_rounds,
                progress=progress,
            )
        )
    return pixel_height_logits_init, global_logits_init, pixel_height_labels


def _prepare_processing_targets(
    output_img_np: np.ndarray,
    computed_processing_size: int,
    device: torch.device,
    focus_map_full: Optional[torch.Tensor],
    alpha_full: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
    """Create downscaled optimization target & focus map for faster iterations.

    Args:
        output_img_np: Full-resolution RGB image (float or uint8 expected).
        computed_processing_size: Target square size for processing (maintains aspect via resize helper).
        device: Torch device.
        focus_map_full: Optional full-resolution focus map tensor.
        alpha_full: Optional full-resolution alpha mask (numpy array) in 0-255 range.

    Returns:
        processing_img_np  : Downscaled numpy image (H_p,W_p,3).
        processing_target  : Torch tensor version (float32) on device.
        focus_map_proc     : Optional downscaled focus map tensor (H_p,W_p).
        alpha_proc         : Optional downscaled alpha mask tensor (float32) on device.
    """
    processing_img_np = resize_image(output_img_np, computed_processing_size)
    processing_target = torch.tensor(
        processing_img_np, dtype=torch.float32, device=device
    )

    focus_map_proc = None
    if focus_map_full is not None:
        fm_proc_np = cv2.resize(
            focus_map_full.cpu().numpy().astype(np.float32),
            (processing_target.shape[1], processing_target.shape[0]),
            interpolation=cv2.INTER_LINEAR,
        )
        focus_map_proc = torch.tensor(fm_proc_np, dtype=torch.float32, device=device)

    alpha_proc = None
    if alpha_full is not None:
        alpha_proc_np = cv2.resize(
            alpha_full.astype(np.float32),
            (processing_target.shape[1], processing_target.shape[0]),
            interpolation=cv2.INTER_NEAREST,
        )
        alpha_proc = torch.tensor(alpha_proc_np, dtype=torch.float32, device=device)

    return processing_img_np, processing_target, focus_map_proc, alpha_proc


def _build_optimizer(
    args,
    processing_target: torch.Tensor,
    processing_pixel_height_logits_init: np.ndarray,
    processing_pixel_height_labels: np.ndarray,
    global_logits_init,
    material_colors: torch.Tensor,
    material_TDs: torch.Tensor,
    background: torch.Tensor,
    device: torch.device,
    perception_loss_module,
    focus_map_proc: Optional[torch.Tensor],
    alpha_proc: Optional[torch.Tensor] = None,
) -> FilamentOptimizer:
    """Instantiate the FilamentOptimizer with initial tensors and configuration.

    Args mirror the optimizer's constructor; this function simply centralizes assembly.

    Returns:
        FilamentOptimizer: Ready-to-run optimizer instance.
    """
    optimizer = FilamentOptimizer(
        args=args,
        target=processing_target,
        pixel_height_logits_init=processing_pixel_height_logits_init,
        pixel_height_labels=processing_pixel_height_labels,
        global_logits_init=global_logits_init,
        material_colors=material_colors,
        material_TDs=material_TDs,
        background=background,
        device=device,
        perception_loss_module=perception_loss_module,
        focus_map=focus_map_proc,
        alpha=alpha_proc,
    )
    return optimizer


def _run_optimization_loop(
    optimizer: FilamentOptimizer, args, device: torch.device
) -> None:
    """Execute the main gradient-based optimization iterations.

    Features:
    - Automatic mixed precision (bfloat16 unless MPS).
    - Periodic visualization & tensorboard logging (every 100 iterations).
    - Discrete solution snapshots controlled via --discrete_check.
    - Early stopping after a patience window (--early_stopping).

    Args:
        optimizer: Configured FilamentOptimizer instance.
        args: Global argument namespace.
        device: Torch device for autocast context.
    """
    print("Starting optimization...")
    tbar = tqdm(range(args.iterations))
    with safe_autocast(device):
        for i in tbar:
            loss_val = optimizer.step(record_best=i % args.discrete_check == 0)

            optimizer.visualize(interval=500)
            optimizer.log_to_tensorboard(interval=500)

            if (i + 1) % 100 == 0:
                tbar.set_description(
                    f"Iteration {i + 1}, Loss = {loss_val.item():.4f}, best validation Loss = {optimizer.best_discrete_loss:.4f}, learning_rate= {optimizer.current_learning_rate:.6f}"
                )
            if (
                optimizer.best_step is not None
                and optimizer.num_steps_done - optimizer.best_step > args.early_stopping
            ):
                print(
                    "Early stopping after",
                    args.early_stopping,
                    "steps without improvement.",
                )
                break
    # Free the captured graph's private memory pool before the (higher
    # resolution) post-processing phases start allocating. `loss_val` aliases
    # storage inside that pool on the graph path, so it has to go first or the
    # pool stays pinned for the rest of the run.
    loss_val = None
    del loss_val
    optimizer.release_cuda_graph()


def _prune_colors_swaps_only(optimizer: FilamentOptimizer, args) -> None:
    """Only the colour and swap reductions of ``FilamentOptimizer.prune``,
    with the same limits and the same non-worsening guard."""
    from autoforge.Helper.PruningHelper import prune_num_colors, prune_num_swaps

    max_colors = max(1, args.pruning_max_colors - (2 if args.flatforge else 1))
    optimizer._run_non_worsening(
        "Reducing colors",
        lambda: prune_num_colors(
            optimizer,
            max_colors,
            optimizer.vis_tau,
            None,
            fast=args.fast_pruning,
            chunking_percent=args.fast_pruning_percent,
            pruning_batch_size=args.pruning_batch_size,
        ),
        forced=optimizer.solution_counts()[0] > max_colors,
    )
    optimizer._run_non_worsening(
        "Reducing swaps",
        lambda: prune_num_swaps(
            optimizer,
            args.pruning_max_swaps,
            optimizer.vis_tau,
            None,
            fast=args.fast_pruning,
            chunking_percent=args.fast_pruning_percent,
            pruning_batch_size=args.pruning_batch_size,
        ),
        forced=optimizer.solution_counts()[1] > args.pruning_max_swaps,
    )


def _prune_sweep(optimizer: FilamentOptimizer, args) -> None:
    """Experiment harness: prune one trained solution to several colour/swap
    limits (``--prune_sweep "C:S,C:S,..."``), each from the same starting
    point, and write every result to prune_sweep.json. The last
    configuration's result stays in the optimizer for the normal export."""
    snapshot = optimizer.solution_snapshot()
    results = []
    for spec in args.prune_sweep.split(","):
        colors, swaps = (int(v) for v in spec.split(":"))
        optimizer.restore_solution_snapshot(
            {**snapshot, "best_params": {k: v.clone() for k, v in snapshot["best_params"].items()}}
        )
        args.pruning_max_colors, args.pruning_max_swaps = colors, swaps
        _prune_colors_swaps_only(optimizer, args)
        loss = PruningHelper.get_initial_loss(
            optimizer.best_params["global_logits"].shape[0], optimizer
        )
        _, n_swaps, _ = optimizer.solution_counts()
        n_colors = optimizer.print_colors()
        results.append({"max_colors": colors, "max_swaps": swaps, "loss": loss,
                        "colors": n_colors, "swaps": n_swaps})
        print(f"PRUNE_SWEEP {colors}:{swaps} loss={loss:.4f} colors={n_colors} swaps={n_swaps}")
    with open(os.path.join(args.output_folder, "prune_sweep.json"), "w", encoding="utf-8") as f:
        json.dump(results, f)


def _discretize_height_only_best(optimizer: FilamentOptimizer) -> torch.Tensor:
    """Discrete heights [H,W] (layers) of the best solution at the solver
    resolution."""
    from autoforge.Modules.Optimizer import _discretize_height_only

    eff = optimizer._apply_height_offset(
        optimizer.best_params["pixel_height_logits"], optimizer.best_params["height_offsets"]
    )
    return _discretize_height_only(eff, optimizer.h, optimizer.max_layers)


def best_heights_at_output_res(
    best_proc: Optional[torch.Tensor],
    proc_init: Optional[np.ndarray],
    full_init: torch.Tensor,
    alpha: Optional[np.ndarray],
) -> torch.Tensor:
    """The best solution's per-pixel height logits at the output resolution.

    Training runs at the processing resolution from ``proc_init`` (the
    full-resolution init, downscaled). What it changed per pixel - the smooth
    height field and the intermediate stack search's single-pixel refines -
    is carried over onto the full-resolution init (nearest neighbour; exact
    when the two resolutions agree). Restoring the plain init instead threw
    the refined heights away while keeping the stack chosen for them, so the
    exported result came out in wrong colors. Shared by the CLI and the
    webui (pipeline_runner.export_results)."""
    if best_proc is None or proc_init is None:
        return full_init.clone()
    delta = best_proc.detach().float() - torch.from_numpy(np.asarray(proc_init)).to(best_proc.device).float()
    if delta.shape != full_init.shape:
        delta = torch.nn.functional.interpolate(
            delta[None, None], size=tuple(full_init.shape[-2:]), mode="nearest"
        )[0, 0]
    out = full_init + delta.to(device=full_init.device, dtype=full_init.dtype)
    if alpha is not None:
        a = alpha[..., 0] if alpha.ndim == 3 else alpha
        if a.shape == tuple(out.shape):
            transparent = torch.from_numpy(a < 128).to(out.device)
            out[transparent] = full_init[transparent]
    return out


def _post_optimize_and_export(
    args,
    optimizer: FilamentOptimizer,
    pixel_height_logits_init: np.ndarray,
    pixel_height_labels: np.ndarray,
    output_target: torch.Tensor,
    alpha: Optional[np.ndarray],
    material_colors_np: np.ndarray,
    material_TDs_np: np.ndarray,
    material_names: List[str],
    device: torch.device,
    focus_map_full: Optional[torch.Tensor],
    focus_map_proc: Optional[torch.Tensor],
    processing_pixel_height_logits_init: Optional[np.ndarray] = None,
) -> float:
    """Finalize solution, optionally prune, and write all output artifacts.

    Steps:
    - Restore full-resolution logits to optimizer and (optionally) height residual.
    - Replace focus map with full-res version if used.
    - Perform pruning (respecting color slots for background & clear in FlatForge mode).
    - Compute final loss estimate and persist to file.
    - Export preview PNG, STL(s), swap instructions & project file.

    Returns:
        float: The final reported loss (post-pruning).
    """
    post_opt_step = 0

    optimizer.log_to_tensorboard(
        interval=1, namespace="post_opt", step=(post_opt_step := post_opt_step + 1)
    )

    full_init = torch.from_numpy(pixel_height_logits_init).to(device)

    # Under colour/swap limits the heights are trained, not assigned, but the
    # stack is final: assigning every pixel its height for that stack at the
    # output resolution beats the trained cluster heights by far (the height
    # fine-tunes and seed search it replaces never came close).
    assigned = optimizer.height_assign and optimizer.best_params is not None
    if assigned:
        # The trained heights are per-pixel assignments at solver resolution
        # (the base logits moved during training), so the delta-on-init
        # transfer below does not apply: start from them, upsampled, and
        # assign again at the output resolution for the best stack.
        from autoforge.Helper.HeightAssign import heights_to_logits
        from autoforge.Helper.OptimizerHelper import batched_layer_material_indices

        with torch.no_grad():
            z_proc = _discretize_height_only_best(optimizer)
            best_dg = batched_layer_material_indices(
                optimizer.best_params["global_logits"],
                optimizer.vis_tau,
                optimizer.best_seed if optimizer.best_seed is not None and optimizer.best_seed >= 0 else 0,
            )
        z0_full = torch.nn.functional.interpolate(
            z_proc[None, None].float(), size=full_init.shape[-2:], mode="nearest"
        )[0, 0]
    else:
        optimizer.best_params["pixel_height_logits"] = best_heights_at_output_res(
            optimizer.best_params["pixel_height_logits"],
            processing_pixel_height_logits_init,
            full_init,
            alpha,
        )
    optimizer.pixel_height_logits = full_init.clone()
    optimizer.pixel_delta = None
    optimizer.target = output_target
    optimizer.pixel_height_labels = torch.tensor(
        pixel_height_labels, dtype=torch.int32, device=device
    )
    if focus_map_proc is not None and focus_map_full is not None:
        optimizer.focus_map = focus_map_full
    if assigned:
        with torch.no_grad():
            z_full = optimizer.assigned_heights(best_dg, heights_to_logits(z0_full, optimizer.max_layers))
            full_logits = heights_to_logits(z_full, optimizer.max_layers)
        optimizer.best_params["pixel_height_logits"] = full_logits
        optimizer.best_params["height_offsets"] = torch.zeros_like(optimizer.best_params["height_offsets"])
        optimizer.pixel_height_logits = full_logits.clone()

    with torch.no_grad():
        with safe_autocast(device):
            if args.minimal_postprocess:
                if args.perform_pruning and args.prune_sweep:
                    _prune_sweep(optimizer, args)
                elif args.perform_pruning:
                    _prune_colors_swaps_only(optimizer, args)
            elif args.perform_pruning:
                # Same post-hoc height-offset fine-tune as the no-pruning
                # path below, run before any pruning phase touches
                # best_params - pruning's own greedy color/swap/layer search
                # starts from whatever height is already there, so giving it
                # the best achievable height first (rather than only
                # fine-tuning once at the very end) lets every later phase
                # benefit, not just the final output.
                # With assigned heights (the default) every pixel already has
                # its best height for the stack at the output resolution and
                # the cluster offsets are zero: the offset fine-tune and the
                # seed search (a different seed reads out a different stack,
                # judged at heights assigned for this one) cannot win there.
                if not assigned:
                    optimizer.fine_tune_height_offsets(num_steps=200)

                # Adjust pruning_max_colors to account for background and clear filament
                # pruning_max_colors = total filaments needed
                # Need to reserve slots: 1 for background (always), 1 for clear (FlatForge only)
                max_colors_for_pruning = args.pruning_max_colors

                if args.flatforge:
                    # FlatForge: pruning_max_colors = colored + clear + background
                    # Reserve 2 slots (1 clear + 1 background)
                    max_colors_for_pruning = max(1, args.pruning_max_colors - 2)
                else:
                    # Traditional: pruning_max_colors = colored + background
                    # Reserve 1 slot for background
                    max_colors_for_pruning = max(1, args.pruning_max_colors - 1)

                optimizer.prune(
                    max_colors_allowed=max_colors_for_pruning,
                    max_swaps_allowed=args.pruning_max_swaps,
                    min_layers_allowed=args.min_layers,
                    max_layers_allowed=args.pruning_max_layer,
                    search_seed=not assigned,
                    fine_tune_height=not assigned,
                    fast_pruning=args.fast_pruning,
                    fast_pruning_percent=args.fast_pruning_percent,
                    pruning_batch_size=args.pruning_batch_size,
                )
                optimizer.log_to_tensorboard(
                    interval=1,
                    namespace="post_opt",
                    step=(post_opt_step := post_opt_step + 1),
                )
            else:
                # Pruning's rng_seed_search (a cheap post-hoc search over the
                # material-selection RNG seed, given the already-converged
                # best_params) normally only runs as pruning's first phase -
                # skipping pruning entirely also skipped this refinement, even
                # though it's independent of pruning and equally applicable
                # here. Run it unconditionally so the non-pruning path gets
                # the same free quality improvement.
                # Baseline measured now, at the full output resolution the
                # search scores at - best_discrete_loss comes from training
                # at the processing resolution, so "beat the start" compared
                # numbers from two different image sizes (see prune()).
                start_loss = optimizer.solution_loss()
                if start_loss is not None:
                    optimizer.rng_seed_search(start_loss, 50, autoset_seed=True)
                from autoforge.Helper.PruningHelper import optimise_swap_positions

                optimise_swap_positions(optimizer, max_passes=3)
                optimizer.fine_tune_height_offsets(num_steps=200)

            disc_global, disc_height_image = optimizer.get_discretized_solution(
                best=True
            )

            final_loss = PruningHelper.get_initial_loss(
                optimizer.best_params["global_logits"].shape[0], optimizer
            )
            with open(os.path.join(args.output_folder, "final_loss.txt"), "w", encoding="utf-8") as f:
                f.write(f"{final_loss}")
            _, n_swaps, _ = optimizer.solution_counts()
            with open(os.path.join(args.output_folder, "final_counts.json"), "w", encoding="utf-8") as f:
                # colors: filaments of the print, the base included.
                json.dump({"colors": optimizer.print_colors(), "swaps": n_swaps}, f)

            print("Done. Saving outputs...")
            comp_disc = optimizer.get_best_discretized_image()
            args.max_layers = optimizer.max_layers

            optimizer.log_to_tensorboard(
                interval=1,
                namespace="post_opt",
                step=(post_opt_step := post_opt_step + 1),
            )

            comp_disc_np = image_to_uint8(comp_disc)
            comp_disc_np = cv2.cvtColor(comp_disc_np, cv2.COLOR_RGB2BGR)
            imwrite(
                os.path.join(args.output_folder, "final_model.png"), comp_disc_np
            )

            # Generate STL files
            if args.flatforge:
                # FlatForge mode: Generate separate STL files for each color
                print("FlatForge mode enabled. Generating separate STL files...")
                generate_flatforge_stls(
                    disc_global.cpu().numpy(),
                    disc_height_image.cpu().numpy(),
                    material_colors_np,
                    material_names,
                    material_TDs_np,
                    args.layer_height,
                    args.background_height,
                    args.background_color,
                    args.stl_output_size,
                    args.output_folder,
                    cap_layers=args.cap_layers,
                    alpha_mask=alpha,
                )
            else:
                # Traditional mode: Generate single STL file
                stl_filename = os.path.join(args.output_folder, "final_model.stl")
                height_map_mm = (
                    disc_height_image.cpu().numpy().astype(np.float32)
                ) * args.layer_height
                generate_stl(
                    height_map_mm,
                    stl_filename,
                    args.background_height,
                    maximum_x_y_size=args.stl_output_size,
                    alpha_mask=alpha,
                )

            if not args.flatforge:
                background_layers = int(round(args.background_height / args.layer_height))
                swap_instructions = generate_swap_instructions(
                    disc_global.cpu().numpy(),
                    disc_height_image.cpu().numpy(),
                    args.layer_height,
                    background_layers,
                    args.background_height,
                    material_names,
                    getattr(args, "background_material_name", None),
                    optimizer.base_material,
                )
                with open(
                    os.path.join(args.output_folder, "swap_instructions.txt"), "w", encoding="utf-8"
                ) as f:
                    for line in swap_instructions:
                        f.write(line + "\n")

                project_filename = os.path.join(args.output_folder, "project_file.hfp")
                generate_project_file(
                    project_filename,
                    args,
                    disc_global.cpu().numpy(),
                    disc_height_image.cpu().numpy(),
                    *model_size_mm(
                        output_target.shape[1],
                        output_target.shape[0],
                        args.stl_output_size,
                    ),
                    os.path.join(args.output_folder, "final_model.stl"),
                    args.csv_file,
                )

            print("All done. Outputs in:", args.output_folder)
            print("Happy Printing!")
            return final_loss


def start(args) -> float:
    """Entry point for a single optimization run.

    Orchestrates the entire pipeline:
    - Validation & device selection.
    - Material & image loading (+ optional auto background selection).
    - Resolution computation & resizing.
    - Heightmap initialization.
    - Optimizer construction & iterative optimization loop.
    - Post-processing, pruning, and output generation.

    Args:
        args: Parsed argument namespace.

    Returns:
        float: Final loss value for this run (after pruning/export).
    """
    if args.num_init_cluster_layers == -1:
        args.num_init_cluster_layers = args.max_layers

    # check if csv or json is given
    if args.csv_file == "" and args.json_file == "":
        print("Error: No CSV or JSON file given. Please provide one of them.")
        sys.exit(1)

    device = get_device(args)

    os.makedirs(args.output_folder, exist_ok=True)

    perform_basic_check(args)

    random_seed = set_seed(args)

    # Load materials (we keep colors_list for potential auto background)
    material_colors_np, material_TDs_np, material_names, colors_list = load_materials(
        args
    )

    # Read input image early (needed for auto background color)
    # Grayscale, grayscale+alpha and 16-bit images are brought to 8-bit
    # BGR(A) first - the code below indexes img.shape[2] and assumes 0-255.
    img = to_bgr_or_bgra_uint8(imread(args.input_image, cv2.IMREAD_UNCHANGED))
    alpha = None
    if img.shape[2] == 4:
        alpha = img[:, :, 3]
        alpha = alpha[..., None]
        img = img[:, :, :3]

    # Convert image from BGR to RGB for color analysis
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Auto background color selection (optional)
    _auto_select_background_color(
        args, img_rgb, alpha, material_colors_np, material_names, colors_list
    )

    # Prepare background color tensor and material tensors
    bg_rgb, background, material_colors, material_TDs = (
        _prepare_background_and_materials(
            args, device, material_colors_np, material_TDs_np
        )
    )

    # Compute sizes
    computed_output_size, computed_processing_size = _compute_pixel_sizes(args)

    # Resize alpha if present (match final resolution) after computing size
    if alpha is not None:
        alpha = resize_image(alpha, computed_output_size)

    # For the final resolution
    output_img_np = resize_image(img_rgb, computed_output_size)
    output_target = torch.tensor(output_img_np, dtype=torch.float32, device=device)

    # Priority mask handling (full-res)
    focus_map_full = _load_priority_mask(args, output_img_np, device)

    # Initialize heightmap
    pixel_height_logits_init, global_logits_init, pixel_height_labels = (
        _initialize_heightmap(
            args,
            output_img_np,
            bg_rgb,
            material_colors_np,
            random_seed,
        )
    )

    # Apply alpha mask to full-res logits BEFORE creating processing copies
    if alpha is not None:
        pixel_height_logits_init[alpha < 128] = -13.815512

    # Prepare processing targets and focus map (processing-res)
    processing_img_np, processing_target, focus_map_proc, alpha_proc = _prepare_processing_targets(
        output_img_np, computed_processing_size, device, focus_map_full,
        alpha_full=alpha,
    )

    # Downscale initial logits/labels to processing resolution
    processing_pixel_height_logits_init = cv2.resize(
        src=pixel_height_logits_init,
        interpolation=cv2.INTER_NEAREST,
        dsize=(processing_target.shape[1], processing_target.shape[0]),
    )
    processing_pixel_height_labels = cv2.resize(
        src=pixel_height_labels,
        interpolation=cv2.INTER_NEAREST,
        dsize=(processing_target.shape[1], processing_target.shape[0]),
    )

    perception_loss_module = None

    # Build optimizer
    optimizer = _build_optimizer(
        args,
        processing_target,
        processing_pixel_height_logits_init,
        processing_pixel_height_labels,
        global_logits_init,
        material_colors,
        material_TDs,
        background,
        device,
        perception_loss_module,
        focus_map_proc,
        alpha_proc=alpha_proc,
    )

    # Run optimization loop
    _run_optimization_loop(optimizer, args, device)

    # record_best only fires every args.discrete_check steps (default 100),
    # so with e.g. 1000 iterations the last checkpoint is at step 900 - the
    # fully-trained final state (steps 900-999) is never evaluated as a best-
    # solution candidate unless iterations happens to be a multiple of
    # discrete_check. Repeated checks here, using the exact same (already
    # training-proven) discretize+evaluate path with a fresh random seed each
    # time, can only improve or hold best_discrete_loss - and a single check
    # risks an unlucky seed making the (possibly genuinely superior) final
    # continuous state look worse than an earlier checkpoint by chance, so it
    # never gets picked as the base for rng_seed_search's later, more
    # thorough search. This training loop has already finished, so the extra
    # calls (cheap: one composite each) don't touch training dynamics/timing.
    #
    # (Tried swapping this for rng_seed_search's faster shared-eff-thick
    # batched machinery instead of the plain per-attempt composite loop
    # below, hoping to afford far more attempts for the same time budget -
    # see results.tsv discard entry. It wasn't reliably cheaper per-attempt
    # at this benchmark's small image size, needed ~300 attempts to beat
    # this loop's 60, and pushed total time past what the extra loss
    # reduction was worth. Kept the simpler, cheaper, already-validated loop.)
    optimizer.finish_training()
    optimizer.finalize_background(args, material_names)
    if optimizer.bg_logits is not None:
        try:
            with open(os.path.join(args.output_folder, "auto_background_color.txt"), "a", encoding="utf-8") as f:
                f.write(f"optimized_filament_color={args.background_color}\n")
                f.write(f"optimized_filament_index={args.background_material_index}\n")
        except Exception:
            traceback.print_exc()

    empty_cache(device)

    # Post-process, prune, and export outputs
    final_loss = _post_optimize_and_export(
        args,
        optimizer,
        pixel_height_logits_init,
        pixel_height_labels,
        output_target,
        alpha,
        material_colors_np,
        material_TDs_np,
        material_names,
        device,
        focus_map_full,
        focus_map_proc,
        processing_pixel_height_logits_init=processing_pixel_height_logits_init,
    )

    return final_loss


def main() -> None:
    """Support multi-run execution via --best_of; persist best run artifacts.

    If --best_of == 1, simply invokes a single start(). Otherwise:
    - Creates temporary run subfolders.
    - Tracks losses, reports statistics (best / median / std).
    - Moves files from best run folder into the final output folder.

    Note: Memory is periodically reclaimed (gc + GPU cache clears on whichever
    backend is active + closing matplotlib figures).
    """
    args = parse_args()
    final_output_folder = args.output_folder
    run_best_loss = 1000000000
    if args.best_of == 1:
        start(args)
    else:
        temp_output_folder = os.path.join(args.output_folder, "temp")
        ret = []
        for i in range(args.best_of):
            try:
                print(f"Run {i + 1}/{args.best_of}")
                run_folder = os.path.join(temp_output_folder, f"run_{i + 1}")
                # Each run gets its own copy: start() writes results back into
                # its args (the pruned max_layers, the auto-picked background
                # color, ...), and a shared namespace handed them to every
                # later run - run 2 onwards started with run 1's pruned layer
                # count instead of the requested --max_layers.
                run_args = copy.copy(args)
                run_args.output_folder = run_folder
                # A fixed seed is the first run's; the others count up from it
                # (the same seed every run made runs 2..N exact repeats).
                # Seed 0 already draws a fresh seed per run in set_seed().
                if args.random_seed != 0:
                    run_args.random_seed = args.random_seed + i
                print(f"Run {i + 1} seed: {run_args.random_seed if run_args.random_seed != 0 else 'random'}")
                os.makedirs(run_args.output_folder, exist_ok=True)
                run_loss = start(run_args)
                print(f"Run {i + 1} finished with loss: {run_loss}")
                if run_loss < run_best_loss:
                    run_best_loss = run_loss
                    print(f"New best loss found: {run_best_loss} in run {i + 1}")
                ret.append((run_folder, run_loss))
                empty_cache()
                import gc

                gc.collect()
                empty_cache()
                import matplotlib.pyplot as plt

                plt.close("all")
            except Exception:
                traceback.print_exc()
        if not ret:
            print(
                f"Error: all {args.best_of} runs failed; see the errors above.",
                file=sys.stderr,
            )
            sys.exit(1)
        best_run = min(ret, key=lambda x: x[1])
        best_run_folder = best_run[0]
        best_loss = best_run[1]

        losses = [x[1] for x in ret]
        median_loss = np.median(losses)
        std_loss = np.std(losses)
        print(f"Best run folder: {best_run_folder}")
        print(f"Best run loss: {best_loss}")
        print(f"Median loss: {median_loss}")
        print(f"Standard deviation of losses: {std_loss}")

        if not os.path.exists(final_output_folder):
            os.makedirs(final_output_folder)
        for file in os.listdir(best_run_folder):
            src_file = os.path.join(best_run_folder, file)
            dst_file = os.path.join(final_output_folder, file)
            if os.path.isfile(src_file):
                os.replace(src_file, dst_file)
        # Every run's files lived in temp/; the best one's are now in place.
        shutil.rmtree(temp_output_folder, ignore_errors=True)


if __name__ == "__main__":
    main()
