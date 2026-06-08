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
import sys
import os
import traceback
from typing import Optional, Tuple, List

import configargparse
import cv2
import torch
import numpy as np
from tqdm import tqdm

from autoforge.Helper import PruningHelper
from autoforge.Helper.AmpUtils import safe_autocast
from autoforge.Helper.FilamentHelper import hex_to_rgb, load_materials
from autoforge.Helper.Heightmaps.ChristofidesHeightMap import (
    run_init_threads,
)

from autoforge.Helper.ImageHelper import resize_image, imread
from autoforge.Helper.OtherHelper import set_seed, perform_basic_check, get_device
from autoforge.Helper.OutputHelper import (
    generate_stl,
    generate_swap_instructions,
    generate_project_file,
    generate_flatforge_stls,
)
from autoforge.Modules.Optimizer import FilamentOptimizer

# check if we can use torch.set_float32_matmul_precision('high')
if torch.__version__ >= "2.0.0":
    try:
        torch.set_float32_matmul_precision("high")
    except Exception as e:
        print("Warning: Could not set float32 matmul precision to high. Error:", e)
        pass


def parse_args() -> argparse.Namespace:
    """Create and parse command-line & config-file arguments.

    Returns:
        argparse.Namespace: Populated arguments structure. Some parameters may be adjusted later
        (e.g., num_init_cluster_layers when -1 to infer from max_layers).
    """
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
        "--early_stopping",
        type=int,
        default=2000,
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
        "--pruning_max_layer",
        type=int,
        default=75,
        help="Max number of layers allowed after pruning",
    )

    parser.add_argument(
        "--random_seed",
        type=int,
        default=0,
        help="Specify the random seed, or use 0 for automatic generation",
    )

    parser.add_argument(
        "--mps",
        action="store_true",
        help="Use the Metal Performance Shaders (MPS) backend, if available.",
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
        default=16,
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

    args = parser.parse_args()
    return args


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
                os.path.join(args.output_folder, "auto_background_color.txt"), "w"
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
    computed_processing_size = int(
        round(computed_output_size / args.processing_reduction_factor)
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
        pm = imread(args.priority_mask, cv2.IMREAD_UNCHANGED)
        if pm.ndim == 3:
            if pm.shape[2] == 4:
                pm = pm[:, :, :3]
            pm = cv2.cvtColor(pm, cv2.COLOR_BGR2GRAY)
        tgt_h, tgt_w = output_img_np.shape[:2]
        pm_resized = cv2.resize(pm, (tgt_w, tgt_h), interpolation=cv2.INTER_LINEAR)
        pm_float = pm_resized.astype(np.float32) / 255.0
        focus_map_full = torch.tensor(pm_float, dtype=torch.float32, device=device)
        cv2.imwrite(
            os.path.join(args.output_folder, "priority_mask_resized.png"),
            (pm_float * 255).astype(np.uint8),
        )
    return focus_map_full


def _initialize_heightmap(
    args,
    output_img_np: np.ndarray,
    bg_rgb: Tuple[float, float, float],
    material_colors_np: np.ndarray,
    random_seed: int,
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
            )
        )
    return pixel_height_logits_init, global_logits_init, pixel_height_labels


def _prepare_processing_targets(
    output_img_np: np.ndarray,
    computed_processing_size: int,
    device: torch.device,
    focus_map_full: Optional[torch.Tensor],
) -> Tuple[np.ndarray, torch.Tensor, Optional[torch.Tensor]]:
    """Create downscaled optimization target & focus map for faster iterations.

    Args:
        output_img_np: Full-resolution RGB image (float or uint8 expected).
        computed_processing_size: Target square size for processing (maintains aspect via resize helper).
        device: Torch device.
        focus_map_full: Optional full-resolution focus map tensor.

    Returns:
        processing_img_np  : Downscaled numpy image (H_p,W_p,3).
        processing_target  : Torch tensor version (float32) on device.
        focus_map_proc     : Optional downscaled focus map tensor (H_p,W_p).
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

    return processing_img_np, processing_target, focus_map_proc


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

            optimizer.visualize(interval=100)
            optimizer.log_to_tensorboard(interval=100)

            if (i + 1) % 100 == 0:
                tbar.set_description(
                    f"Iteration {i + 1}, Loss = {loss_val:.4f}, best validation Loss = {optimizer.best_discrete_loss:.4f}, learning_rate= {optimizer.current_learning_rate:.6f}"
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

    optimizer.pixel_height_logits = torch.from_numpy(
        pixel_height_logits_init
    ).to(device)
    optimizer.best_params["pixel_height_logits"] = torch.from_numpy(
        pixel_height_logits_init
    ).to(device)
    optimizer.target = output_target
    optimizer.pixel_height_labels = torch.tensor(
        pixel_height_labels, dtype=torch.int32, device=device
    )
    if focus_map_proc is not None and focus_map_full is not None:
        optimizer.focus_map = focus_map_full

    with torch.no_grad():
        with safe_autocast(device):
            if args.perform_pruning:
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
                    search_seed=True,
                    fast_pruning=args.fast_pruning,
                    fast_pruning_percent=args.fast_pruning_percent,
                )
                optimizer.log_to_tensorboard(
                    interval=1,
                    namespace="post_opt",
                    step=(post_opt_step := post_opt_step + 1),
                )

            disc_global, disc_height_image = optimizer.get_discretized_solution(
                best=True
            )

            final_loss = PruningHelper.get_initial_loss(
                optimizer.best_params["global_logits"].shape[0], optimizer
            )
            with open(os.path.join(args.output_folder, "final_loss.txt"), "w") as f:
                f.write(f"{final_loss}")

            print("Done. Saving outputs...")
            comp_disc = optimizer.get_best_discretized_image()
            args.max_layers = optimizer.max_layers

            optimizer.log_to_tensorboard(
                interval=1,
                namespace="post_opt",
                step=(post_opt_step := post_opt_step + 1),
            )

            comp_disc_np = comp_disc.cpu().numpy().astype(np.uint8)
            comp_disc_np = cv2.cvtColor(comp_disc_np, cv2.COLOR_RGB2BGR)
            cv2.imwrite(
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
                background_layers = int(args.background_height // args.layer_height)
                swap_instructions = generate_swap_instructions(
                    disc_global.cpu().numpy(),
                    disc_height_image.cpu().numpy(),
                    args.layer_height,
                    background_layers,
                    args.background_height,
                    material_names,
                    getattr(args, "background_material_name", None),
                )
                with open(
                    os.path.join(args.output_folder, "swap_instructions.txt"), "w"
                ) as f:
                    for line in swap_instructions:
                        f.write(line + "\n")

                project_filename = os.path.join(args.output_folder, "project_file.hfp")
                generate_project_file(
                    project_filename,
                    args,
                    disc_global.cpu().numpy(),
                    disc_height_image.cpu().numpy(),
                    output_target.shape[1],
                    output_target.shape[0],
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
    img = imread(args.input_image, cv2.IMREAD_UNCHANGED)
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
    processing_img_np, processing_target, focus_map_proc = _prepare_processing_targets(
        output_img_np, computed_processing_size, device, focus_map_full
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
    )

    # Run optimization loop
    _run_optimization_loop(optimizer, args, device)

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
    )

    return final_loss


def main() -> None:
    """Support multi-run execution via --best_of; persist best run artifacts.

    If --best_of == 1, simply invokes a single start(). Otherwise:
    - Creates temporary run subfolders.
    - Tracks losses, reports statistics (best / median / std).
    - Moves files from best run folder into the final output folder.

    Note: Memory is periodically reclaimed (gc + CUDA cache clears + closing matplotlib figures).
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
                args.output_folder = run_folder
                os.makedirs(args.output_folder, exist_ok=True)
                run_loss = start(args)
                print(f"Run {i + 1} finished with loss: {run_loss}")
                if run_loss < run_best_loss:
                    run_best_loss = run_loss
                    print(f"New best loss found: {run_best_loss} in run {i + 1}")
                ret.append((run_folder, run_loss))
                torch.cuda.empty_cache()
                import gc

                gc.collect()
                torch.cuda.empty_cache()
                import matplotlib.pyplot as plt

                plt.close("all")
            except Exception:
                traceback.print_exc()
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
                os.rename(src_file, dst_file)


if __name__ == "__main__":
    main()
