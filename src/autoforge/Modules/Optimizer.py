import argparse
import gc
import random
import os
import threading
import time
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from autoforge.Helper.CAdamW import CAdamW
from autoforge.Helper.DeviceUtils import (
    backend_of,
    empty_cache,
    supports_graph_capture,
    synchronize,
)
from autoforge.Helper.OptimizerHelper import (
    batched_layer_material_indices,
    composite_image_cont,
    composite_image_disc,
    PrecisionManager,
)

from autoforge.Loss.LossFunctions import loss_fn, compute_loss

# `matplotlib.pyplot` (~0.3s) and `torch.utils.tensorboard` (~1.0s) are
# expensive imports that pull in large dependency trees. Both features are
# opt-in (visualize defaults on, but is commonly disabled for headless/
# server use via --no-visualize; tensorboard defaults off), so they're
# imported lazily below, only when actually needed, instead of unconditionally
# at module load time. `plt` is populated as a module global the first time
# `self.visualize_flag` is True (see __init__); every other use site in this
# file is already gated behind that same flag.
plt = None


def _discretize_height_only(
    effective_logits: torch.Tensor, h: float, max_layers: int
) -> torch.Tensor:
    """
    The height-only half of ``FilamentOptimizer.discretize_solution`` -
    factored out so callers that already have ``effective_logits`` and only
    need the discrete height image (not the per-layer material assignment)
    can skip both the redundant _apply_height_offset recompute and the
    O(max_layers) Python-loop Gumbel-softmax material-selection loop that
    discretize_solution always does to produce its (unused, in that case)
    discrete_global return value.
    """
    pixel_heights = (max_layers * h) * torch.sigmoid(effective_logits)
    discrete_height_image = torch.round(pixel_heights / h).to(torch.int32)
    return torch.clamp(discrete_height_image, 0, max_layers)


def _compute_height_offset_term(
    optimizer, target_shape: torch.Size
) -> torch.Tensor:
    """
    The part of ``FilamentOptimizer._apply_height_offset`` that is
    independent of ``pixel_logits`` - i.e. everything except the final
    ``pixel_logits + offsets`` add. Depends only on ``height_offsets`` and
    ``pixel_height_labels`` (both fixed for an optimizer's whole lifetime -
    pruning never touches either) plus the target spatial shape (also fixed
    across every candidate within one pruning phase, since removing a layer
    changes global_logits' layer count but never pixel_height_logits'
    spatial [H,W] shape). Safe to compute once and reuse via
    ``cand_params["pixel_height_logits"] + shared_offsets`` instead of a
    fresh ``_apply_height_offset`` call (gather + possible bicubic
    interpolate) per candidate.
    """
    labels = optimizer.pixel_height_labels.to(torch.long)  # [H,W]
    offsets_1d = optimizer.best_params["height_offsets"].squeeze(-1)  # [L]
    gathered = offsets_1d[labels]  # [H,W]
    mask = (labels != 0).to(gathered.dtype)
    offsets = gathered * mask
    if offsets.shape != target_shape:
        offsets = F.interpolate(
            offsets.unsqueeze(0).unsqueeze(0),
            size=target_shape[-2:],
            mode="bicubic",
        ).squeeze(0).squeeze(0)
    return offsets


class FilamentOptimizer:
    def __init__(
        self,
        args: argparse.Namespace,
        target: torch.Tensor,
        pixel_height_logits_init: np.ndarray,
        pixel_height_labels: np.ndarray,
        global_logits_init: np.ndarray,
        material_colors: torch.Tensor,
        material_TDs: torch.Tensor,
        background: torch.Tensor,
        device: torch.device,
        perception_loss_module: Optional[torch.nn.Module],
        focus_map: Optional[torch.Tensor] = None,
        alpha: Optional[torch.Tensor] = None,
        preview_callback=None,
        preview_callback_interval: int = 25,
    ):
        """
        Initialize an optimizer instance.

        Args:
            args (argparse.Namespace): Command-line arguments.
            target (torch.Tensor): Target image tensor.
            pixel_height_logits_init (np.ndarray): Initial pixel height logits.
            material_colors (torch.Tensor): Tensor of material colors.
            material_TDs (torch.Tensor): Tensor of material transmission/opacity parameters.
            background (torch.Tensor): Background color tensor.
            device (torch.device): Device to run the optimization on.
            perception_loss_module (torch.nn.Module): Module to compute perceptual loss.
            focus_map (torch.Tensor | None): Optional priority mask [H,W] in [0,1]. Higher -> higher loss weight.
            alpha (torch.Tensor | None): Optional alpha mask [H,W] or [H,W,1] in 0-255. Pixels with alpha < 128 are masked out of loss.
        """
        self.args = args
        self.target = target  # smaller (solver) resolution, shape [H,W,3], float32
        self.H, self.W = target.shape[:2]

        self.precision = PrecisionManager(device)
        # `composite_image_cont`/`composite_image_disc` are @torch.jit.script
        # and don't observe the ambient torch.autocast context (see
        # composite_image_cont's docstring-adjacent comment) - the memory
        # win from mixed precision has to be threaded through explicitly as
        # a dtype argument instead. Reuse whatever dtype PrecisionManager
        # already selected for this device (None means: stay fp32, e.g. on
        # MPS/CPU-without-bf16/older GPUs - no behavior change there).
        self.composite_compute_dtype = (
            self.precision.autocast_dtype if self.precision.enabled else None
        )

        pixel_height_labels = np.round(pixel_height_labels)

        # replace entire entries of pixel_height_logits with 0
        # pixel_height_logits_init *= 1.0
        # set pixel_height_logits where pixel_height_labels is 0 to -13.815512 (the lowest init sigmoid value)
        # pixel_height_logits_init[pixel_height_labels == 0] = -13.815512

        self.pixel_height_logits = torch.tensor(
            pixel_height_logits_init, dtype=torch.float32, device=device
        )
        # Base logits are frozen
        self.pixel_height_logits.requires_grad_(False)

        print("layers", int(pixel_height_labels.flatten().max()))
        self.cluster_layers = int(pixel_height_labels.flatten().max()) + 1
        self.pixel_height_labels = torch.tensor(
            pixel_height_labels, dtype=torch.int32, device=device
        )
        self.height_offsets = torch.nn.Parameter(
            torch.zeros(self.cluster_layers, 1, device=device)
        )  # Trainable

        # Basic hyper-params
        self.material_colors = material_colors
        self.material_TDs = material_TDs
        self.background = background
        self.max_layers = args.max_layers
        self.h = args.layer_height
        self.learning_rate = args.learning_rate
        self.current_learning_rate = args.learning_rate
        self.final_tau = args.final_tau
        self.vis_tau = args.final_tau
        self.init_tau = args.init_tau
        self.device = device
        self.best_swaps = 0
        self.perception_loss_module = perception_loss_module
        self.visualize_flag = args.visualize

        # Priority mask
        self.focus_map = None
        if focus_map is not None:
            fm = focus_map
            if fm.dim() == 3 and fm.shape[-1] == 1:
                fm = fm.squeeze(-1)
            self.focus_map = fm.to(device=self.device, dtype=torch.float32)

        # Alpha mask for transparency
        self.alpha = None
        if alpha is not None:
            self.alpha = alpha.to(device=self.device, dtype=torch.float32)

        self.preview_callback = preview_callback
        self.preview_callback_interval = preview_callback_interval

        # Initialize TensorBoard writer (import is lazy - see module docstring)
        if args.tensorboard:
            from torch.utils.tensorboard import SummaryWriter

            if args.run_name:
                self.writer = SummaryWriter(log_dir=f"runs/{args.run_name}")
            else:
                self.writer = SummaryWriter()
        else:
            self.writer = None

        # Flag used by log_to_tensorboard()
        self.tensorboard_log = bool(getattr(args, "tensorboard", False))

        # Initialize global logits
        if global_logits_init is None:
            # We have an initial guess for 'global_logits'
            num_materials = material_colors.shape[0]
            global_logits_init = (
                torch.ones(
                    (self.max_layers, num_materials), dtype=torch.float32, device=device
                )
                * -1.0
            )
            for i in range(self.max_layers):
                global_logits_init[i, i % num_materials] = 1.0

            global_logits_init += torch.rand_like(global_logits_init) * 0.2 - 0.1
        # Convert only if numpy array
        if isinstance(global_logits_init, np.ndarray):
            global_logits_init = torch.from_numpy(global_logits_init).to(
                dtype=torch.float32, device=device
            )
        elif torch.is_tensor(global_logits_init):
            global_logits_init = global_logits_init.to(
                dtype=torch.float32, device=device
            )
        else:
            raise TypeError("global_logits_init must be a numpy array or torch Tensor")
        global_logits_init.requires_grad_(True)

        self.loss = None

        self.params = {
            "pixel_height_logits": self.pixel_height_logits,
            "global_logits": global_logits_init,
            "height_offsets": self.height_offsets,
        }

        # Tau schedule
        self.num_steps_done = 0
        self.warmup_steps = min(
            args.iterations - 1, args.warmup_fraction * args.iterations
        )
        self.decay_rate = (self.init_tau - self.final_tau) / (
            args.iterations - self.warmup_steps
        )

        # Initialize optimizer
        self.optimizer = CAdamW(
            [self.params["global_logits"], self.height_offsets],
            lr=self.learning_rate,
        )

        # Persistent Exponential(1) buffer for the material Gumbel-Softmax.
        # Drawing the noise here, in `step`, instead of letting
        # F.gumbel_softmax draw it inside composite_image_cont is what allows
        # the forward/backward to be CUDA-graph-captured without changing the
        # random stream: a captured graph replays with its own philox offset
        # sequence, so any RNG *inside* the capture silently diverges from the
        # eager path. See composite_image_cont's gumbel_exp argument.
        self._gumbel_exp = torch.empty_like(self.params["global_logits"])

        # CUDA graph state for the training step (see _maybe_capture_graph).
        self._graph = None
        self._graph_loss = None
        self._graph_tau = None
        self._graph_attempted = False
        self._graph_enabled = bool(getattr(args, "cuda_graph", True))
        # Capture only after a few real steps. Those steps double as the
        # warmup that CUDA graph capture requires (every lazily-initialized
        # handle, workspace and .grad buffer already exists by then), which is
        # why there is no separate side-stream warmup: running one costs a
        # second per-stream allocator pool for no benefit here - measured
        # peak reserved 220MB with a side-stream warmup vs 178MB without,
        # against 132MB for the eager baseline.
        self._graph_capture_after = 3

        # Setup best discrete solution tracking
        self.best_discrete_loss = float("inf")
        self.best_params = None
        self.best_tau = None
        self.best_seed = None
        self.best_step = None

        # If you want a figure for real-time visualization:
        if self.visualize_flag:
            global plt
            import matplotlib.pyplot as plt

            if self.args.disable_visualization_for_gradio != 1:
                plt.ion()
            self.fig, self.ax = plt.subplots(2, 3, figsize=(14, 6))

            self.target_im_ax = self.ax[0, 0].imshow(
                np.array(self.target.cpu(), dtype=np.uint8)
            )
            self.ax[0, 0].set_title("Target Image")

            self.current_comp_ax = self.ax[0, 1].imshow(
                np.zeros((self.H, self.W, 3), dtype=np.uint8)
            )
            self.ax[0, 1].set_title("Current Composite")

            self.best_comp_ax = self.ax[0, 2].imshow(
                np.zeros((self.H, self.W, 3), dtype=np.uint8)
            )
            self.ax[0, 2].set_title("Best Discrete Composite")
            if self.args.disable_visualization_for_gradio != 1:
                plt.pause(0.1)

            self.depth_map_ax = self.ax[1, 0].imshow(
                np.zeros((self.H, self.W), dtype=np.uint8), cmap="viridis"
            )
            self.ax[1, 0].set_title("Current Height Map")

            self.diff_depth_map_ax = self.ax[1, 1].imshow(
                np.zeros((self.H, self.W), dtype=np.uint8), cmap="viridis"
            )
            self.ax[1, 1].set_title("Height Map Changes")

            # Priority mask visualization in bottom-right
            if self.focus_map is not None:
                fm_np = self.focus_map.cpu().detach().numpy()
                # Normalize for display (robust to non [0,1] ranges)
                fm_min, fm_max = float(fm_np.min()), float(fm_np.max())
                if fm_max - fm_min > 1e-8:
                    fm_norm = (fm_np - fm_min) / (fm_max - fm_min)
                else:
                    fm_norm = np.zeros_like(fm_np)
                fm_uint8 = (fm_norm * 255).astype(np.uint8)
                self.priority_mask_ax = self.ax[1, 2].imshow(
                    fm_uint8, cmap="magma", vmin=0, vmax=255
                )
                self.ax[1, 2].set_title("Priority Mask")
            else:
                self.ax[1, 2].text(
                    0.5,
                    0.5,
                    "No Priority Mask",
                    ha="center",
                    va="center",
                    fontsize=10,
                    color="gray",
                    transform=self.ax[1, 2].transAxes,
                )
                self.ax[1, 2].set_axis_off()

            # Compute and store the initial height map for later difference computation.
            with torch.no_grad():
                initial_height = (self.max_layers * self.h) * torch.sigmoid(
                    self._apply_height_offset()
                )
            self.initial_height_map = initial_height.cpu().detach().numpy()

    def _apply_height_offset(
        self,
        pixel_logits: Optional[torch.Tensor] = None,
        height_offsets: Optional[torch.Tensor] = None,
    ):
        if pixel_logits is None:
            pixel_logits = self.pixel_height_logits
        if height_offsets is None:
            height_offsets = self.height_offsets

        # Differentiable gather of per-cluster offsets with zero for background (label==0)
        labels = self.pixel_height_labels.to(torch.long)  # [H,W]
        offsets_1d = height_offsets.squeeze(-1)  # [L]
        # Gradient multiplier: forward unchanged, grad wrt offsets_1d scaled by height_offsets_grad_scale
        s = getattr(self, "height_offsets_grad_scale", 1.0)
        offsets_1d = offsets_1d * s + offsets_1d.detach() * (1.0 - s)
        # Use advanced indexing to map each pixel's label to its cluster offset
        gathered = offsets_1d[labels]  # [H,W]
        mask = (labels != 0).to(gathered.dtype)
        offsets = gathered * mask  # zero-out background
        if offsets.shape != pixel_logits.shape:
            offsets = F.interpolate(
                offsets.unsqueeze(0).unsqueeze(0),
                size=pixel_logits.shape[-2:],
                mode="bicubic",
            ).squeeze(0).squeeze(0)
        return pixel_logits + offsets

    def _remove_height_offset(
        self,
        pixel_logits: Optional[torch.Tensor] = None,
        height_offsets: Optional[torch.Tensor] = None,
    ):
        if pixel_logits is None:
            pixel_logits = self.pixel_height_logits
        if height_offsets is None:
            height_offsets = self.height_offsets

        # Differentiable gather of per-cluster offsets with zero for background (label == 0)
        labels = self.pixel_height_labels.to(torch.long)  # [H, W]
        offsets_1d = height_offsets.squeeze(-1)  # [L]

        # Same gradient scaling behavior as apply
        s = getattr(self, "height_offsets_grad_scale", 1.0)
        offsets_1d = offsets_1d * s + offsets_1d.detach() * (1.0 - s)

        # Map each pixel label to its cluster offset
        gathered = offsets_1d[labels]  # [H, W]
        mask = (labels != 0).to(gathered.dtype)
        offsets = gathered * mask  # zero-out background
        if offsets.shape != pixel_logits.shape:
            offsets = F.interpolate(
                offsets.unsqueeze(0).unsqueeze(0),
                size=pixel_logits.shape[-2:],
                mode="bicubic",
            ).squeeze(0).squeeze(0)

        return pixel_logits - offsets

    def _get_tau(self):
        """
        Compute tau for height & global given how many steps we've done.

        Returns:
            Tuple[float, float]: Tau values for height and global.
        """
        i = self.num_steps_done
        tau_init = self.init_tau
        if i < self.warmup_steps:
            return tau_init, tau_init
        else:
            # simple linear decay
            t = max(
                self.final_tau, tau_init - self.decay_rate * (i - self.warmup_steps)
            )
            return t, t

    def _forward_backward(self, tau_height: float, tau_global: float):
        """Forward + backward for one training step (no parameter update).

        Split out of ``step`` so exactly this region can be CUDA-graph
        captured: it touches only static buffers (frozen base logits, target,
        material tables, the two parameters and their .grad tensors, and the
        pre-drawn ``_gumbel_exp``) and contains no RNG and no host sync.
        """
        effective_logits = self._apply_height_offset()

        loss = loss_fn(
            {
                "pixel_height_logits": effective_logits,
                "global_logits": self.params["global_logits"],
            },
            target=self.target,
            tau_height=tau_height,
            tau_global=tau_global,
            h=self.h,
            max_layers=self.max_layers,
            material_colors=self.material_colors,
            material_TDs=self.material_TDs,
            background=self.background,
            add_penalty_loss=10.0,
            focus_map=self.focus_map,
            alpha=self.alpha,
            compute_dtype=self.composite_compute_dtype,
            gumbel_exp=self._gumbel_exp,
        )

        if self.precision.scaler is not None:
            self.precision.scaler.scale(loss).backward()
        else:
            loss.backward()
        return loss

    def _optimizer_step(self):
        """The parameter update half of ``PrecisionManager.backward_and_step``."""
        if self.precision.scaler is not None:
            self.precision.scaler.step(self.optimizer)
            self.precision.scaler.update()
        else:
            self.optimizer.step()

    def _maybe_capture_graph(self, tau_height: float, tau_global: float) -> None:
        """Capture the forward/backward of one training step into a GPU graph.

        CUDA-family backends only - NVIDIA CUDA and AMD ROCm, where the same
        ``torch.cuda.CUDAGraph`` API maps onto HIP graphs. Apple Metal has no
        equivalent and CPU has nothing to gain, so ``supports_graph_capture``
        turns both away before any ``torch.cuda`` call is made and they take
        the eager path below unchanged.

        The training step is dominated by kernel-launch overhead rather than
        GPU work at realistic solver resolutions (measured at the default
        stl_output_size=50 -> 122x125 solver image: ~3.7ms wall per step of
        which only ~1.6ms is GPU-busy; a 62x smaller image ran at the same
        wall time). Replaying a captured graph collapses ~340 dispatched aten
        ops into a single launch: 3.74ms -> 1.61ms per step.

        Only the forward/backward is captured, not the optimizer step -
        CAdamW's bias correction derives its ``step_size`` from a Python-side
        step counter, which a capture would freeze at its capture-time value
        (bias_correction2 is still ~0.06 that early, i.e. an effective LR ~4x
        too small for the rest of training). Capturing the optimizer as well
        measured only 1.57ms vs 1.61ms, so there is nothing to gain from
        working around that.

        Capture is side-effect free: the warmup passes only accumulate into
        .grad (parameters are untouched by forward/backward), the RNG draw
        lives outside, and grads are zeroed afterwards.
        """
        if (
            not self._graph_enabled
            or self._graph_attempted
            or self.num_steps_done < self._graph_capture_after
            or not supports_graph_capture(self.device)
            or self.precision.scaler is not None
        ):
            return
        self._graph_attempted = True
        try:
            # Return the eager path's cached-but-free blocks to the driver so
            # the graph's private pool grows from a clean state instead of on
            # top of them (the private pool cannot reuse main-pool blocks).
            empty_cache(self.device)
            self.optimizer.zero_grad(set_to_none=False)

            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                captured_loss = self._forward_backward(tau_height, tau_global)
            # Keep only the storage, not the autograd graph - replays are pure
            # kernel replays and never touch autograd, and holding the graph
            # would pin capture-stream AccumulateGrad nodes for any later
            # eager fallback step.
            self._graph_loss = captured_loss.detach()
            del captured_loss
            self._graph = graph
            self._graph_tau = (tau_height, tau_global)

            # A capture that *succeeds* is not yet a capture that *replays
            # correctly*: on backends where graph support is thinner than
            # NVIDIA's (ROCm maps this onto HIP graphs) a bad replay shows up
            # as wrong numbers, not as an exception, and every subsequent
            # step would silently train on garbage gradients. So replay once
            # and check it against an eager step before handing the training
            # loop over to it. Costs one extra forward/backward, once.
            if not self._graph_replay_matches_eager(tau_height, tau_global):
                graph.reset()
                self._graph = None
                self._graph_loss = None
                self._graph_tau = None
                print(
                    f"Graph replay disagreed with the eager step on this "
                    f"{backend_of(self.device)} device; using eager steps."
                )
        except Exception as exc:  # pragma: no cover - hardware/driver dependent
            self._graph = None
            self._graph_loss = None
            self._graph_tau = None
            print(f"Graph capture unavailable ({exc}); using eager steps.")
        finally:
            self.optimizer.zero_grad(set_to_none=False)

    def _validated_params(self):
        return [self.params["global_logits"], self.height_offsets]

    def _graph_replay_matches_eager(
        self, tau_height: float, tau_global: float
    ) -> bool:
        """Does replaying the captured graph reproduce the eager loss+grads?

        Both runs see identical inputs - parameters are untouched by
        forward/backward and ``_gumbel_exp`` was drawn once for this step - so
        they should agree to within kernel non-determinism (backward reductions
        use atomics, so this is a tolerance check rather than bit equality).
        The failure this guards against is not a 1% drift but a replay that
        produces zeros or garbage, which no tolerance hides.
        """
        params = self._validated_params()
        try:
            self.optimizer.zero_grad(set_to_none=False)
            eager_loss = self._forward_backward(tau_height, tau_global).detach().clone()
            eager_grads = [p.grad.detach().clone() for p in params]

            self.optimizer.zero_grad(set_to_none=False)
            self._graph.replay()
            replay_loss = self._graph_loss.detach().clone()
            replay_grads = [p.grad.detach().clone() for p in params]
        except Exception:
            return False

        if not torch.isfinite(replay_loss).all():
            return False
        if not torch.allclose(eager_loss, replay_loss, rtol=1e-2, atol=1e-3):
            return False
        # An all-zero eager gradient carries no signal to compare against
        # (it would make a dead replay look correct); accept in that case,
        # since a step with no gradient cannot be trained wrong either.
        if not any(float(g.abs().max()) > 0.0 for g in eager_grads):
            return True
        for eager_g, replay_g in zip(eager_grads, replay_grads):
            if not torch.allclose(eager_g, replay_g, rtol=1e-2, atol=1e-4):
                return False
        return True

    def release_cuda_graph(self) -> None:
        """Drop the captured graph and its private memory pool.

        Called once training finishes so the pool can be reclaimed before the
        (higher-resolution, more memory-hungry) post-processing phases run.
        A no-op when nothing was captured, which is every Metal and CPU run.
        """
        if self._graph is None:
            return
        # Every kernel from the last replay has to have finished before the
        # graph's private pool goes away.
        synchronize(self.device)
        if self.loss is not None:
            # self.loss aliases the graph's private pool; detach it from that
            # storage before the pool goes away.
            self.loss = self.loss.clone()
        # The teardown order below is load-bearing. Calling empty_cache()
        # while the CUDAGraph object is still reachable - it is kept alive by
        # a reference cycle, so simply dropping the last name is not enough -
        # makes the run die with "an illegal memory access was encountered"
        # at the next sync. Reproduced 2/2 at --iterations 20 and fixed 3/3
        # by forcing the collection first; longer runs happened to hide it
        # because something else synced in between. reset() releases the pool
        # explicitly instead of relying on the finalizer, and gc.collect()
        # then guarantees the object itself is gone before empty_cache().
        self._graph.reset()
        self._graph = None
        self._graph_loss = None
        self._graph_tau = None
        gc.collect()
        synchronize(self.device)
        empty_cache(self.device)

    def step(self, record_best: bool = False):
        """
        Perform exactly one gradient-descent update step.

        Args:
            record_best (bool, optional): Whether to record the best discrete solution. Defaults to False.

        Returns:
            torch.Tensor: A detached 0-dim tensor holding the loss value of the current
            step. Call `.item()` on it when a Python float is actually needed (e.g. for
            display) - deferring that sync is the point (see note below).
        """
        if self.pixel_height_logits.grad is not None:
            self.pixel_height_logits.grad = None

        # set_to_none=False keeps every .grad tensor at a stable address, which
        # a captured graph requires (it replays writes to the exact
        # buffers recorded at capture time). Numerically a no-op: backward
        # accumulates onto exact zeros instead of assigning a fresh tensor.
        self.optimizer.zero_grad(set_to_none=False)

        warmup_steps = int(
            self.args.iterations * self.args.learning_rate_warmup_fraction
        )

        if self.num_steps_done < warmup_steps and warmup_steps > 0:
            lr_scale = self.num_steps_done / warmup_steps
            self.current_learning_rate = lr_scale * self.learning_rate
        else:
            self.current_learning_rate = self.learning_rate

        for g in self.optimizer.param_groups:
            g["lr"] = self.current_learning_rate

        tau_height, tau_global = self._get_tau()

        # Draw this step's Gumbel noise here, outside any captured region, so
        # the eager and graph-replay paths consume the identical random stream.
        self._gumbel_exp.exponential_()

        if self._graph is not None and self._graph_tau == (tau_height, tau_global):
            self._graph.replay()
            loss = self._graph_loss
            self._optimizer_step()
        else:
            loss = self._forward_backward(tau_height, tau_global)
            self._optimizer_step()
            # Drop the reference to this step's autograd graph *before*
            # attempting capture. A live graph keeps its AccumulateGrad nodes
            # alive, and those are cached per-parameter and reused - so the
            # capture would inherit AccumulateGrad nodes bound to the default
            # stream, which makes the capture stream depend on the legacy
            # stream and aborts capture with cudaErrorStreamCaptureImplicit
            # ("operation would make the legacy stream depend on a capturing
            # blocking stream"). Detaching lets them be freed and re-created
            # on the capture stream.
            loss = loss.detach()
            self._maybe_capture_graph(tau_height, tau_global)

        self.num_steps_done += 1

        if (
            self.preview_callback is not None
            and self.num_steps_done % self.preview_callback_interval == 0
        ):
            try:
                self.preview_callback(self, self.num_steps_done)
            except Exception:
                pass

        if record_best:
            self._maybe_update_best_discrete()
        # torch.cuda.empty_cache()

        # `.item()` forces a CUDA sync, blocking the CPU until every kernel
        # queued so far has actually finished on the GPU. The loss value is
        # only ever consumed at a coarse interval (progress bar text every
        # ~100 steps, tensorboard/visualize on their own intervals) - calling
        # `.item()` unconditionally here meant every single step paid a full
        # sync, serializing CPU kernel-launch overhead with GPU execution
        # instead of letting them overlap. Keep a detached tensor here and
        # let callers materialize it to a float only when they actually
        # display it.
        self.loss = loss.detach()

        return self.loss

    def discretize_solution(
        self,
        params: dict,
        tau_global: float,
        h: float,
        max_layers: int,
        rng_seed: int = -1,
    ):
        """
        Convert continuous logs to discrete layer counts and discrete color IDs.

        Args:
            params (dict): Dictionary containing the parameters 'pixel_height_logits' and 'global_logits'.
            tau_global (float): Temperature parameter for global material assignment.
            h (float): Height of each layer.
            max_layers (int): Maximum number of layers.
            rng_seed (int, optional): Random seed for deterministic sampling. Defaults to -1.

        Returns:
            tuple: A tuple containing:
                - torch.Tensor: Discrete global material assignments, shape [max_layers].
                - torch.Tensor: Discrete height image, shape [H, W].
        """
        pixel_logits = params["pixel_height_logits"]
        pixel_offset = params["height_offsets"]
        effective_logits = self._apply_height_offset(
            pixel_logits=pixel_logits, height_offsets=pixel_offset
        )

        global_logits = params["global_logits"]
        discrete_height_image = _discretize_height_only(effective_logits, h, max_layers)

        # Vectorized equivalent of the former per-layer
        # deterministic_gumbel_softmax + argmax loop (see
        # batched_layer_material_indices): identical selection, one kernel
        # instead of ~max_layers tiny ones.
        discrete_global = batched_layer_material_indices(
            global_logits, tau_global, rng_seed
        )
        return discrete_global, discrete_height_image

    def log_to_tensorboard(
        self, interval: int = 100, namespace: str = "", step: int = None
    ):
        """
        Log metrics and images to TensorBoard.

        Args:
            interval (int, optional): Interval for logging images. Defaults to 100.
            namespace (str, optional): Namespace prefix for logs. If provided, logs will be prefixed with this value. Defaults to "".
            step (int, optional): Optional override for the step number to log. Defaults to None.
        """
        with torch.no_grad():
            if not self.tensorboard_log or self.writer is None:
                return

            # Prepare namespace prefix
            prefix = f"{namespace}/" if namespace else ""

            steps = step if step is not None else self.num_steps_done

            # Log metrics
            self.writer.add_scalar(
                f"Loss/{prefix}best_discrete", self.best_discrete_loss, steps
            )
            self.writer.add_scalar(f"Loss/{prefix}best_swaps", self.best_swaps, steps)

            tau_height, tau_global = self._get_tau()

            # Metrics that are only relevant for the main optimization loop
            if not prefix:
                self.writer.add_scalar("Params/tau_height", tau_height, steps)
                self.writer.add_scalar("Params/tau_global", tau_global, steps)
                self.writer.add_scalar(
                    "Params/lr", self.optimizer.param_groups[0]["lr"], steps
                )
                self.writer.add_scalar("Loss/train", self.loss, steps)

            # Log images periodically
            if (steps + 1) % interval == 0:
                with torch.no_grad():
                    effective_logits = self._apply_height_offset()
                    comp_img = composite_image_cont(
                        effective_logits,
                        self.params["global_logits"],
                        tau_height,
                        tau_global,
                        self.h,
                        self.max_layers,
                        self.material_colors,
                        self.material_TDs,
                        self.background,
                    )
                    self.writer.add_images(
                        f"Current Output/{prefix}composite",
                        comp_img.permute(2, 0, 1).unsqueeze(0) / 255.0,
                        steps,
                    )

    def visualize(self, interval: int = 25):
        """
        Update the figure if visualize_flag is True.

        Args:
            interval (int, optional): Interval of steps to update the visualization. Defaults to 25.
        """
        if not self.visualize_flag:
            return

        # Update only every 'interval' steps for speed
        if (self.num_steps_done % interval) != 0:
            return

        with torch.no_grad():
            tau_h, tau_g = self._get_tau()
            effective_logits = self._apply_height_offset()
            comp = composite_image_cont(
                effective_logits,
                self.params["global_logits"],
                tau_h,
                tau_g,
                self.h,
                self.max_layers,
                self.material_colors,
                self.material_TDs,
                self.background,
            )
            comp_np = np.clip(comp.cpu().detach().numpy(), 0, 255).astype(np.uint8)
            self.current_comp_ax.set_data(comp_np)

            # Priority mask does not change over time; no update needed unless future edits require it.

            if self.best_params is not None:
                # Update the depth map correctly.
                effective_best_logits = self._apply_height_offset(
                    self.best_params["pixel_height_logits"],
                    self.best_params["height_offsets"],
                )
                best_comp = composite_image_disc(
                    effective_best_logits,
                    self.best_params["global_logits"],
                    self.vis_tau,
                    self.vis_tau,
                    self.h,
                    self.max_layers,
                    self.material_colors,
                    self.material_TDs,
                    self.background,
                    rng_seed=self.best_seed,
                )
                best_comp_np = np.clip(best_comp.cpu().detach().numpy(), 0, 255).astype(
                    np.uint8
                )
                self.best_comp_ax.set_data(best_comp_np)

            # Update the depth map correctly.
            effective_logits = self._apply_height_offset()
            height_map = (self.max_layers * self.h) * torch.sigmoid(effective_logits)
            height_map = height_map.cpu().detach().numpy()

            # Normalize safely, checking for a constant image.
            if np.allclose(height_map.max(), height_map.min()):
                height_map_norm = np.zeros_like(height_map)
            else:
                height_map_norm = (height_map - height_map.min()) / (
                    height_map.max() - height_map.min()
                )

            height_map_uint8 = (height_map_norm * 255).astype(np.uint8)
            self.depth_map_ax.set_data(height_map_uint8)
            self.depth_map_ax.set_clim(0, 255)

            # Compute and update the difference depth map (current - initial)
            diff_map = height_map - self.initial_height_map
            # print(diff_map.min(), diff_map.max())
            # Normalize the difference map safely.
            # if np.allclose(diff_map.max(), diff_map.min()):
            #     diff_map_norm = np.zeros_like(diff_map)
            # else:
            #     diff_map_norm = (diff_map - diff_map.min()) / (
            #         diff_map.max() - diff_map.min()
            #     )
            self.diff_depth_map_ax.set_data(diff_map)
            self.diff_depth_map_ax.set_clim(-2.5, 2.5)

            loss_display = (
                self.loss.item() if torch.is_tensor(self.loss) else self.loss
            )
            self.fig.suptitle(
                f"Step {self.num_steps_done}/{self.args.iterations}, Tau: {tau_g:.4f}, Loss: {loss_display:.4f}, Best Discrete Loss: {self.best_discrete_loss:.4f}"
            )
            if self.args.disable_visualization_for_gradio != 1:
                plt.pause(0.01)
            plt.savefig(self.args.output_folder + "/vis_temp.png")

    def _draw_prune_preview(self):
        """
        Lightweight matplotlib update for pruning steps.
        Only renders the best (discrete) composite — skips the full pipeline.
        """
        if not self.visualize_flag or not hasattr(self, "fig"):
            return

        with torch.no_grad():
            try:
                eff_logits = self._apply_height_offset(
                    self.best_params["pixel_height_logits"],
                    self.best_params["height_offsets"],
                )
                best_comp = composite_image_disc(
                    eff_logits,
                    self.best_params["global_logits"],
                    self.vis_tau,
                    self.vis_tau,
                    self.h,
                    self.max_layers,
                    self.material_colors,
                    self.material_TDs,
                    self.background,
                    rng_seed=self.best_seed,
                )
                comp_np = np.clip(best_comp.cpu().detach().numpy(), 0, 255).astype(np.uint8)
                self.best_comp_ax.set_data(comp_np)

                # Update the title to reflect we're in the pruning stage
                self.fig.suptitle(
                    f"Pruning — best discrete loss: {self.best_discrete_loss:.4f}"
                )

                # Repaint the GUI window
                self.fig.canvas.draw_idle()
                self.fig.canvas.manager.start_event_loop(0.01) if hasattr(
                    self.fig.canvas.manager, "start_event_loop"
                ) else plt.pause(0.01)
            except Exception:
                pass

    def get_current_parameters(self):
        """
        Return a copy of the current parameters (pixel_height_logits, global_logits).

        Returns:
            Dict[str, torch.Tensor]: Current parameters.
        """
        return {
            "pixel_height_logits": self.pixel_height_logits.detach().clone(),
            "global_logits": self.params["global_logits"].detach().clone(),
            "height_offsets": self.height_offsets.detach().clone(),
        }

    def get_discretized_solution(
        self, best: bool = False, custom_height_logits: torch.Tensor = None, apply_height_offset: bool = True
    ):
        """
        Return the discrete global assignment and the discrete pixel-height map
        for the current solution, using the current tau.

        Args:
            best (bool, optional): Whether to use the best solution. Defaults to False.
            custom_height_logits (torch.Tensor, optional): Custom height logits to use. We currently use this for the full size image. Defaults to None.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Discrete global assignment and pixel-height map.
        """
        if best and self.best_params is None:
            return None, None

        current_params = self.best_params.copy() if best else self.params
        if custom_height_logits is not None:
            current_params["pixel_height_logits"] = self._apply_height_offset(
                custom_height_logits
            ) if apply_height_offset else custom_height_logits

        if best:
            disc_global, disc_height_image = self.discretize_solution(
                self.best_params,
                self.vis_tau,
                self.h,
                self.max_layers,
                rng_seed=self.best_seed,
            )
            return disc_global, disc_height_image
        else:
            tau_height, tau_global = self._get_tau()
            with torch.no_grad():
                disc_global, disc_height_image = self.discretize_solution(
                    current_params,
                    tau_height,
                    self.h,
                    self.max_layers,
                    rng_seed=random.randrange(1, 1000000),
                )
            return disc_global, disc_height_image

    def get_best_discretized_image(
        self,
        custom_height_logits: torch.Tensor = None,
        custom_global_logits: torch.Tensor = None,
    ):
        with torch.no_grad():
            if custom_height_logits is not None:
                effective_logits = self._apply_height_offset(
                    custom_height_logits,
                    self.best_params["height_offsets"],
                )
            else:
                effective_logits = self._apply_height_offset(
                    self.best_params["pixel_height_logits"],
                    self.best_params["height_offsets"],
                )
            global_logits = (
                custom_global_logits
                if custom_global_logits is not None
                else self.best_params["global_logits"]
            )
            best_comp = composite_image_disc(
                effective_logits,
                global_logits,
                self.vis_tau,
                self.vis_tau,
                self.h,
                self.max_layers,
                self.material_colors,
                self.material_TDs,
                self.background,
                rng_seed=self.best_seed,
                # This runs under no_grad at *full* output resolution (up to
                # 4x the training loop's processing resolution), making it
                # the single largest transient allocation in the pipeline -
                # and it only feeds the preview PNG, not the STL geometry
                # (that comes from the height map), so the lower-precision
                # color math is a safe trade here.
                compute_dtype=self.composite_compute_dtype,
            )
        return best_comp

    def prune(
        self,
        max_colors_allowed: int,
        max_swaps_allowed: int,
        min_layers_allowed: int,
        max_layers_allowed: int,
        search_seed: bool = True,
        seed_search_count: int = 200,
        fine_tune_height: bool = True,
        pre_fine_tune_height: bool = False,
        fine_tune_steps: int = 50,
        fast_pruning: bool = False,
        fast_pruning_percent: float = 0.20,
        pruning_batch_size: int = 0,
        cancel_event: Optional[threading.Event] = None,
        pause_event: Optional[threading.Event] = None,
        apply_spike_removal: bool = True,
    ) -> bool:
        """Run the pruning pipeline (optional seed search and height polish
        -> color -> swap -> layer -> swap-position -> spike removal).

        ``search_seed``/``seed_search_count`` look for a better *material*
        discretization of the solution already found (the Gumbel draw is
        seeded, so a different seed can read the same logits into a visibly
        better set of per-layer colors). ``fine_tune_height``/
        ``fine_tune_steps`` polish the per-cluster height offsets. Both run
        before pruning, because every pruning phase is a greedy search scored
        against the current solution — handing them a better starting point
        improves everything downstream — and both are strictly
        non-worsening: they keep their result only if the real discrete loss
        went down (see ``rng_seed_search`` and ``polish_height_offsets``).

        ``cancel_event``/``pause_event`` are only checked *between* phases,
        not inside them — the phases themselves (prune_num_colors etc.) are
        greedy search algorithms with their own internal state and are not
        safe to interrupt mid-phase without risking an inconsistent result.
        This still gives a responsive-enough pause/cancel in practice, since
        each phase is the unit of work a user would want to stop between.

        Returns ``True`` if pruning ran to completion, ``False`` if it was
        cancelled partway through (the solution as of the last completed
        phase is kept either way — pruning phases only ever improve or hold
        the loss, never worsen it, so a partial run is still a valid result).
        """
        # Now run pruning
        from autoforge.Helper.PruningHelper import (
            prune_num_colors,
            prune_num_swaps,
            prune_redundant_layers,
            optimise_swap_positions,
            remove_height_spikes,
            _compute_loss_for_heightmap,
        )

        # How many times this result has been pruned. Pruning is a greedy
        # search, so running it again on the same limits keeps finding
        # improvements — but a few steps are one-off trades that must not be
        # repeated (see post_remove_spikes).
        self._prune_runs = getattr(self, "_prune_runs", 0) + 1

        def _wait_if_paused() -> bool:
            """Blocks while paused. Returns True if cancelled (while paused
            or otherwise)."""
            if pause_event is not None:
                while pause_event.is_set():
                    if cancel_event is not None and cancel_event.is_set():
                        return True
                    time.sleep(0.1)
            return cancel_event is not None and cancel_event.is_set()

        _guarded = self._run_non_worsening
        _measure = self.solution_loss
        _current_counts = self.solution_counts

        # Measure where we actually stand *before* anything else. This is
        # also what the seed search compares against: it used to be handed
        # `self.best_discrete_loss`, which was measured during training at the
        # *processing* resolution, while everything here scores at the full
        # output resolution — so "never regress below start_loss" was
        # comparing against a number from a different image size.
        dg, dh = self.get_discretized_solution(best=True)
        current_loss = (
            _compute_loss_for_heightmap(self, dg) if dh is not None else None
        )
        if current_loss is not None:
            print(f"Pre-prune discrete loss: {current_loss:.4f}")

        if search_seed and current_loss is not None:
            self._current_prune_phase = "Searching color seeds"
            _guarded(
                "Seed search",
                lambda: self.rng_seed_search(
                    current_loss,
                    seed_search_count,
                    autoset_seed=True,
                    progress_callback=self._prune_phase_progress,
                ),
                forced=False,
            )
            seed_loss = _measure()
            if seed_loss is not None and seed_loss < current_loss:
                print(f"Seed search: loss {current_loss:.4f} -> {seed_loss:.4f}")
                current_loss = seed_loss
            else:
                print("Seed search: no better seed found; keeping the current one")

        if _wait_if_paused():
            return False

        # Off by default because the CLI already runs its own (longer)
        # fine-tune before calling this; the webui turns it on, which is what
        # export_results used to do unconditionally.
        if pre_fine_tune_height and fine_tune_height:
            self._current_prune_phase = "Fine-tuning height"
            self.polish_height_offsets(
                num_steps=fine_tune_steps,
                progress_callback=self._prune_phase_progress,
            )

        # clear pytorch and system cache to reduce vram usage
        empty_cache(self.device)
        import gc

        gc.collect()
        empty_cache(self.device)

        # Build a combined callback that updates the matplotlib window (CLI)
        # and also fires the external WebSocket callback (WebUI), tagged
        # with which phase is currently running.
        self._current_prune_phase = None

        def _prune_callback(_optimizer, _percent):
            # Update the matplotlib preview during pruning steps
            self._draw_prune_preview()
            if self.preview_callback is not None:
                try:
                    self.preview_callback(_optimizer, _percent, phase=self._current_prune_phase)
                except TypeError:
                    # Callers that don't accept the `phase` kwarg (e.g. a
                    # plain matplotlib-only callback) still work.
                    self.preview_callback(_optimizer, _percent)
                except Exception:
                    pass

        if _wait_if_paused():
            return False

        self._current_prune_phase = "Reducing colors"
        _guarded(
            "Reducing colors",
            lambda: prune_num_colors(
                self,
                max_colors_allowed,
                self.vis_tau,
                None,
                fast=fast_pruning,
                chunking_percent=fast_pruning_percent,
                pruning_batch_size=pruning_batch_size,
                preview_callback=_prune_callback,
            ),
            forced=_current_counts()[0] > max_colors_allowed,
        )

        if _wait_if_paused():
            return False

        self._current_prune_phase = "Reducing swaps"
        _guarded(
            "Reducing swaps",
            lambda: prune_num_swaps(
                self,
                max_swaps_allowed,
                self.vis_tau,
                None,
                fast=fast_pruning,
                chunking_percent=fast_pruning_percent,
                pruning_batch_size=pruning_batch_size,
                preview_callback=_prune_callback,
            ),
            forced=_current_counts()[1] > max_swaps_allowed,
        )

        if _wait_if_paused():
            return False

        self._current_prune_phase = "Reducing layers"
        _guarded(
            "Reducing layers",
            lambda: prune_redundant_layers(
                self,
                None,
                min_layers_allowed,
                max_layers_allowed,
                fast=fast_pruning,
                chunking_percent=fast_pruning_percent,
                preview_callback=_prune_callback,
            ),
            forced=int(self.max_layers) > max_layers_allowed,
        )

        if _wait_if_paused():
            return False

        self._current_prune_phase = "Optimising swap positions"
        # Never forced: this phase only moves swap boundaries around, it never
        # removes one, so there is no limit it could be catching up with.
        _guarded(
            "Optimising swap positions",
            lambda: optimise_swap_positions(
                self,
                preview_callback=_prune_callback,
            ),
            forced=False,
        )

        if _wait_if_paused():
            return False

        # A second height polish, now that the layer stack has been reduced —
        # the offsets that were best for 75 layers are rarely best for 20.
        # Spikes are cleaned up by post_remove_spikes just below, which is
        # the ordering polish_height_offsets applies for the pre-pruning
        # pass; here it already falls out of the phase order.
        if fine_tune_height:
            self._current_prune_phase = "Fine-tuning height"
            self.fine_tune_height_offsets(
                num_steps=fine_tune_steps,
                progress_callback=self._prune_phase_progress,
            )
            _prune_callback(self, 95)

        if _wait_if_paused():
            return False

        # `apply_spike_removal=False` lets a caller that repeats prune() in a
        # loop (webui auto-repeat pruning) skip this on every intermediate
        # pass and run it only once the loop has actually converged:
        # cleaning up spikes after each pass, only to keep grinding the
        # color/swap/layer counts down again next pass, wasted the work and
        # (via allow_regression on the very first pass) paid its accuracy
        # cost on results that were about to be superseded anyway.
        if apply_spike_removal and getattr(self.args, "spike_removal", False):
            self._current_prune_phase = "Removing spikes"
            # Only the first prune of this result may trade accuracy for
            # printability; see post_remove_spikes. Without this, pruning the
            # same result again re-applied that trade every time, which is
            # what made repeated pruning visibly worse instead of better.
            self.post_remove_spikes(allow_regression=self._prune_runs <= 1)
        self._current_prune_phase = None
        # Calculate and Print current loss
        dg, dh = self.get_discretized_solution(best=True)
        if dh is not None:
            current_loss = _compute_loss_for_heightmap(self, dg)
            print(f"Post-prune discrete loss: {current_loss:.4f}")
        return True

    # Making "a phase never makes the result worse" actually true.
    #
    # Every pruning phase already refuses candidates that don't improve — but
    # each scores with its own fast path (`get_best_discretized_image` with
    # custom logits, `composite_image_disc` with a shared thickness prefix,
    # ...), and those don't agree to the last decimal with
    # `_compute_loss_for_heightmap`, which is what the result is finally judged
    # by. A phase could therefore accept a change its own metric called an
    # improvement while the real one got slightly worse — measured on a repeat
    # prune with every limit already satisfied: `prune_num_swaps` 71.60 ->
    # 72.66, `optimise_swap_positions` 71.98 -> 72.07.
    #
    # Rather than rewriting every tuned inner loop to share one scorer, each
    # phase runs as a transaction against the real metric and is rolled back if
    # it comes out worse. A phase that is *required* to reduce something the
    # solution still exceeds is exempt — trading accuracy for a printable
    # colour/swap/layer count is precisely its job — so this only ever protects
    # against reductions nobody asked for.
    PHASE_LOSS_EPS = 1e-6

    def solution_loss(self) -> Optional[float]:
        """The discrete loss of the current best solution, by the same measure
        the finished result is reported with. None if there's no solution."""
        from autoforge.Helper.PruningHelper import _compute_loss_for_heightmap

        disc_global, disc_height = self.get_discretized_solution(best=True)
        if disc_global is None or disc_height is None:
            return None
        return _compute_loss_for_heightmap(self, disc_global)

    def solution_counts(self) -> tuple[int, int, int]:
        """``(colors, swaps, layers)`` of the current solution, in the same
        terms the pruning limits are expressed in."""
        from autoforge.Helper.PruningHelper import find_color_bands

        disc_global, _disc_height = self.get_discretized_solution(best=True)
        if disc_global is None:
            return (0, 0, int(self.max_layers))
        return (
            int(torch.unique(disc_global).numel()),
            max(0, len(find_color_bands(disc_global)) - 1),
            int(self.max_layers),
        )

    def solution_snapshot(self) -> dict:
        """Everything a pruning phase can change, cloned."""
        return {
            "best_params": {
                k: (v.detach().clone() if isinstance(v, torch.Tensor) else v)
                for k, v in self.best_params.items()
            },
            "pixel_height_logits": self.pixel_height_logits,
            "max_layers": self.max_layers,
            "best_seed": self.best_seed,
        }

    def restore_solution_snapshot(self, snapshot: dict) -> None:
        # prune_redundant_layers replaces best_params wholesale and moves
        # max_layers with it, so the two have to go back together or
        # global_logits no longer matches the layer count.
        self.best_params = snapshot["best_params"]
        self.pixel_height_logits = snapshot["pixel_height_logits"]
        self.max_layers = snapshot["max_layers"]
        self.best_seed = snapshot["best_seed"]

    def _run_non_worsening(self, name: str, run, *, forced: bool) -> bool:
        """Run a pruning phase, rolling it back if it made things worse.

        ``forced`` marks a phase that still has a limit to meet, where a loss
        increase is the intended trade. Returns True if the phase's work was
        kept.
        """
        if forced:
            run()
            return True
        before = self.solution_loss()
        if before is None:
            run()
            return True
        snapshot = self.solution_snapshot()
        run()
        after = self.solution_loss()
        if after is not None and after > before + self.PHASE_LOSS_EPS:
            self.restore_solution_snapshot(snapshot)
            print(
                f"{name}: loss {before:.4f} -> {after:.4f} | "
                f"reverted (nothing was over its limit, so this phase had to improve or hold)"
            )
            return False
        return True

    def _prune_phase_progress(self, percent: float, loss: Optional[float] = None) -> None:
        """Report progress *within* the current pruning phase, as a plain
        0-100 fraction of that phase's own work, plus the best loss it has
        reached so far.

        The pruning phases that predate this report a stage-relative,
        non-monotonic number instead (see ``prune``'s ``_prune_callback`` and
        the webui's mapping of it), which is why the pre-pruning searches get
        their own straightforward channel. They also need the loss: neither
        changes the color/swap/layer counts at all, so the loss is the only
        place their progress is visible.
        """
        callback = self.preview_callback
        if callback is None:
            return
        phase = getattr(self, "_current_prune_phase", None)
        for kwargs in ({"phase": phase, "loss": loss}, {"phase": phase}, {}):
            try:
                callback(self, float(percent), **kwargs)
                return
            except TypeError:
                # Older callbacks accept fewer keywords; fall back in turn.
                continue
            except Exception:
                return

    def polish_height_offsets(
        self, num_steps: int = 50, progress_callback=None
    ) -> bool:
        """Fine-tune the height offsets, then clean up the spikes that move
        may have introduced, and keep the pair only if the result is better.

        ``fine_tune_height_offsets`` already reverts when it doesn't improve,
        but on its own that isn't enough here: it optimizes the *height map*,
        and a lower loss can still come with fresh single-pixel towers that
        print badly. Spike removal is therefore run immediately afterwards and
        the two are judged together — if the combined result is worse than
        where we started, both are rolled back.

        Returns True if the polished height was kept.
        """
        from autoforge.Helper.PruningHelper import _compute_loss_for_heightmap

        dg_before, dh_before = self.get_discretized_solution(best=True)
        if dh_before is None:
            return False
        pre_loss = _compute_loss_for_heightmap(self, dg_before)

        snapshot = {
            "height_offsets": self.best_params["height_offsets"].detach().clone(),
            "pixel_height_logits": self.best_params["pixel_height_logits"].detach().clone(),
        }
        live_logits = self.pixel_height_logits
        # fine_tune_height_offsets writes this when it improves; a revert has
        # to put it back, or later code that compares against it (the CLI's
        # no-pruning rng_seed_search) is measuring a solution we discarded.
        best_discrete_loss = self.best_discrete_loss

        improved = self.fine_tune_height_offsets(
            num_steps=num_steps, progress_callback=progress_callback
        )
        if improved and getattr(self.args, "spike_removal", False):
            # allow_regression=True on purpose: this spike pass is not judged
            # on its own, the combined result below is what decides.
            self.post_remove_spikes(allow_regression=True)

        dg_after, dh_after = self.get_discretized_solution(best=True)
        post_loss = (
            _compute_loss_for_heightmap(self, dg_after) if dh_after is not None else float("inf")
        )
        kept = post_loss <= pre_loss
        if not kept:
            self.best_params["height_offsets"] = snapshot["height_offsets"]
            self.best_params["pixel_height_logits"] = snapshot["pixel_height_logits"]
            self.pixel_height_logits = live_logits
            self.best_discrete_loss = best_discrete_loss
        print(
            f"Height polish: loss {pre_loss:.4f} -> {post_loss:.4f} | "
            f"{'kept' if kept else 'reverted (no improvement once spikes were cleaned up)'}"
        )
        return kept

    def post_remove_spikes(self, allow_regression: bool = True):
        """Smooth isolated tall pixels out of the final height map.

        Spike removal is about *printability*, not accuracy: single-pixel
        towers print badly however good they look in the loss. So the first
        pass is allowed to cost a little quality — that is the whole point of
        it — and ``allow_regression=True`` (the default, and what the CLI
        uses) keeps that behaviour.

        Repeat passes are a different matter. Pruning the same result again
        re-ran this unconditionally, and on a detailed image it reliably
        *raises* the loss (1015.32 -> 1016.01 on the benchmark image, every
        single run), so each extra pass paid the same accuracy cost again
        while the spikes it removes were already gone. With
        ``allow_regression=False`` the cleaned map is kept only if it does not
        make the result worse, which is what lets repeated pruning converge
        instead of drifting downwards. Mirrors ``fine_tune_height_offsets``,
        which has always reverted on no improvement.

        Returns True if the cleaned height map was kept.
        """
        from autoforge.Helper.PruningHelper import (
            remove_height_spikes,
            _compute_loss_for_heightmap,
        )

        dg_post, dh_post = self.get_discretized_solution(best=True)
        if dh_post is not None:
            original_logits = self.best_params["pixel_height_logits"]
            original_self_logits = self.pixel_height_logits
            pre_loss = _compute_loss_for_heightmap(self, dg_post)

            # Work on continuous height map to avoid discretization and numpy round-trips.
            eff_logits = self._apply_height_offset(
                self.best_params["pixel_height_logits"],
                self.best_params["height_offsets"],
            )
            height_map = torch.sigmoid(eff_logits) * float(self.max_layers)

            dh_clean, spikes = remove_height_spikes(
                height_map,
                threshold_layers=self.args.spike_threshold_layers,
                num_passes=getattr(self.args, "spike_removal_passes", 4),
            )

            normalized = dh_clean.clamp(0, self.max_layers) / float(self.max_layers)
            normalized = normalized.clamp(1e-6, 1 - 1e-6)
            cleaned_logits = self._remove_height_offset(
                pixel_logits=torch.log(normalized) - torch.log1p(-normalized),
                height_offsets=self.best_params["height_offsets"],
            )
            self.best_params["pixel_height_logits"] = cleaned_logits.to(self.device)
            self.pixel_height_logits = cleaned_logits.to(self.device)
            dg_post, dh_post = self.get_discretized_solution(best=True)
            post_loss = _compute_loss_for_heightmap(self, dg_post)
            kept = allow_regression or post_loss <= pre_loss
            if not kept:
                self.best_params["pixel_height_logits"] = original_logits
                self.pixel_height_logits = original_self_logits
            note = (
                # Either the first prune of this result, or a step that is
                # judged together with what follows it (polish_height_offsets).
                "kept despite the higher loss — spikes are a printability problem, not a loss problem"
                if kept and allow_regression and post_loss > pre_loss
                else "kept"
                if kept
                else "reverted (a repeat pass must not make the result worse)"
            )
            print(
                f"Spike removal: loss {pre_loss:.4f} -> {post_loss:.4f} | "
                f"spikes fixed {spikes} | {note}"
            )
            try:
                with open(
                    os.path.join(self.args.output_folder, "spike_removal_stats.txt"),
                    "a",
                ) as f:
                    f.write(
                        f"post_prune,threshold_layers={self.args.spike_threshold_layers},"
                        f"spikes={spikes},loss_before={pre_loss:.6f},loss_after={post_loss:.6f},"
                        f"kept={int(kept)}\n"
                    )
            except Exception:
                pass
            return kept
        return False

    def fine_tune_height_offsets(
        self, num_steps: int = 50, lr: float = 0.007, progress_callback=None
    ) -> bool:
        """
        Post-hoc refinement of ``height_offsets`` with the material
        assignment frozen at its current (hard, one-hot-like) discrete
        choice - a differentiable "polish" pass distinct from
        ``rng_seed_search``/``optimise_swap_positions`` (which only search
        over discrete choices, never adjust the continuous height directly).

        Critically, this optimizes at tau=1.0 (matching the main training
        loop's schedule, not ``self.vis_tau``) - tau=1.0 keeps the print-mask
        sigmoid's scale factor (``10.0/(tau_height+eps)``) small and the
        gradient well-behaved; at ``vis_tau`` (0.01) that scale factor
        explodes to ~1000, making every tested learning rate wildly
        unstable (verified empirically: even lr=0.001 at vis_tau made the
        result 3-8x worse). The soft tau=1.0 loss only drives the gradient
        step - the real discrete loss is re-checked every step, the best
        (loss, offsets) pair seen is tracked throughout, and the run stops
        early once ``patience`` steps pass without a new best. Only that
        best snapshot is ever kept, so a bad late-step trajectory can never
        regress the result below the starting point.

        Returns:
            bool: True if the fine-tuned offsets were kept (improved the
            real discrete loss), False if reverted.
        """
        from autoforge.Helper.PruningHelper import (
            _compute_loss_for_heightmap,
            disc_to_logits,
        )
        from autoforge.Loss.LossFunctions import loss_fn

        dg_cur, _ = self.get_discretized_solution(best=True)
        pre_loss = _compute_loss_for_heightmap(self, dg_cur)
        fixed_global_logits = disc_to_logits(
            dg_cur, self.material_colors.shape[0], big_pos=1e5
        ).detach()
        pixel_logits = self.best_params["pixel_height_logits"].detach()
        orig_offsets = self.best_params["height_offsets"].detach().clone()
        best_loss = pre_loss
        best_offsets = orig_offsets.clone()
        steps_since_best = 0
        patience = max(10, num_steps // 4)

        with torch.enable_grad():
            ft_offsets = orig_offsets.clone().requires_grad_(True)
            ft_optimizer = CAdamW([ft_offsets], lr=lr)
            tbar = tqdm(range(num_steps), desc="Fine-tuning height", leave=True)
            for step_idx in tbar:
                if progress_callback is not None:
                    progress_callback(100.0 * step_idx / max(num_steps, 1), best_loss)
                ft_optimizer.zero_grad()

                effective_logits = self._apply_height_offset(pixel_logits, ft_offsets)

                self.best_params["height_offsets"] = ft_offsets.detach()
                dg_step, _ = self.get_discretized_solution(best=True)
                s_loss = _compute_loss_for_heightmap(self, dg_step)

                if s_loss < best_loss:
                    best_loss = s_loss
                    best_offsets = ft_offsets.detach().clone()
                    steps_since_best = 0
                else:
                    steps_since_best += 1
                tbar.set_description(
                    f"Pre_Loss: {pre_loss:.4f} Best: {best_loss:.4f}"
                )
                if steps_since_best > patience:
                    break

                loss = loss_fn(
                    {
                        "pixel_height_logits": effective_logits,
                        "global_logits": fixed_global_logits,
                    },
                    target=self.target,
                    tau_height=1.0,
                    tau_global=1.0,
                    h=self.h,
                    max_layers=self.max_layers,
                    material_colors=self.material_colors,
                    material_TDs=self.material_TDs,
                    background=self.background,
                    add_penalty_loss=10.0,
                    focus_map=self.focus_map,
                    alpha=self.alpha,
                    compute_dtype=self.composite_compute_dtype,
                    # The only backward in the pipeline that runs at full
                    # *output* resolution, so it sets the whole run's VRAM
                    # high-water mark - use the layer-chunked composite.
                    low_memory=True,
                )
                loss.backward()
                ft_optimizer.step()

        if best_loss < pre_loss:
            self.best_params["height_offsets"] = best_offsets
            self.best_discrete_loss = best_loss
            return True
        self.best_params["height_offsets"] = orig_offsets
        return False

    def _maybe_update_best_discrete(self):
        """
        Discretize the current solution, compute the discrete-mode loss,
        and update the best solution if it improves.
        """
        seed = np.random.randint(0, 1000000)

        tau_g = self.vis_tau
        with torch.no_grad():
            effective_logits = self._apply_height_offset()

            # Discretize to get disc_global (per-layer material assignments)
            disc_global, disc_height_image = self.discretize_solution(
                self.params, tau_g, self.h, self.max_layers, rng_seed=seed
            )

            # Build discrete global logits from disc_global to avoid
            # re-running Gumbel-Softmax inside composite_image_disc.
            num_materials = self.material_colors.shape[0]
            from autoforge.Helper.PruningHelper import disc_to_logits
            disc_global_logits = disc_to_logits(
                disc_global, num_materials, big_pos=1e5
            )

            # Composite using the already-discretized global assignment
            comp_disc = composite_image_disc(
                effective_logits,
                disc_global_logits,
                tau_g,
                tau_g,
                self.h,
                self.max_layers,
                self.material_colors,
                self.material_TDs,
                self.background,
                rng_seed=seed,
                compute_dtype=self.composite_compute_dtype,
            )

            current_disc_loss = compute_loss(
                comp=comp_disc,
                target=self.target,
                focus_map=self.focus_map,
                alpha=self.alpha,
            ).item()
            from autoforge.Helper.PruningHelper import find_color_bands

            if current_disc_loss < self.best_discrete_loss:
                self.best_discrete_loss = current_disc_loss
                self.best_params = self.get_current_parameters()
                self.best_tau = tau_g
                self.best_seed = seed
                self.best_swaps = len(find_color_bands(disc_global)) - 1
                self.best_step = self.num_steps_done

    def rng_seed_search(
        self,
        start_loss: float,
        num_seeds: int,
        autoset_seed: bool = False,
        progress_callback=None,
    ):
        """
        Search for the best seed for the best discrete solution.

        The material assignment is read out of the logits with a *seeded*
        Gumbel draw, so a different seed can turn the same solution into a
        visibly better set of per-layer colors at no cost to anything else.
        The winner is re-scored through the exact composite path before being
        accepted, so this can only ever improve on ``start_loss``.

        Args:
            start_loss (float): Loss to beat. Pass a freshly measured
                full-resolution discrete loss — a number from a different
                resolution makes the "never regress" check meaningless.
            num_seeds (int): Number of seeds to search.
            autoset_seed (bool, optional): Whether to automatically set the seed. Defaults to False.
            progress_callback (callable, optional): Called with 0-100 as the
                search works through the seeds.

        Returns:
            tuple[int | None, float]: Best seed found (None if none beat
            ``start_loss``) and its loss.
        """
        # Only the material-selection RNG seed varies across candidates here
        # - the height map (best_params["pixel_height_logits"]/"height_offsets")
        # is fixed for the whole search, so the expensive [L,H,W] effective-
        # thickness pipeline (height offset gather + print mask + bleed) is
        # computed once and reused, instead of being recomputed from scratch
        # (inside composite_image_disc) for every one of num_seeds candidates.
        from autoforge.Helper.PruningHelper import (
            _make_shared_eff_thick,
            _material_select_batched_seeds,
            _compose_candidate,
        )

        best_seed = None
        best_loss = start_loss
        global_logits = self.best_params["global_logits"]
        # Batch the per-layer noise-generation + material-selection step
        # across a group of candidate seeds at once (still composing/scoring
        # one candidate at a time - only the L-iteration Python loop's
        # per-iteration cost is amortized across the batch, not the [H,W]
        # compositing, to avoid growing peak VRAM with num_seeds).
        seed_batch_size = 20
        all_seeds = np.random.randint(0, 1000000, size=num_seeds)
        with torch.no_grad():
            shared_eff = _make_shared_eff_thick(self)
            tbar = tqdm(range(0, num_seeds, seed_batch_size), desc="Searching for new best seed")
            for batch_start in tbar:
                if progress_callback is not None:
                    progress_callback(100.0 * batch_start / max(num_seeds, 1), best_loss)
                batch_seeds = all_seeds[batch_start : batch_start + seed_batch_size]
                seeds_t = torch.as_tensor(batch_seeds, device=global_logits.device, dtype=torch.int64)
                cols_b, tds_b = _material_select_batched_seeds(
                    global_logits,
                    self.material_colors,
                    self.material_TDs,
                    seeds_t,
                    tau=self.vis_tau,
                )
                for b, seed in enumerate(batch_seeds):
                    comp_disc = _compose_candidate(
                        shared_eff, cols_b[b], tds_b[b], self.background
                    )
                    current_disc_loss = compute_loss(
                        comp=comp_disc,
                        target=self.target,
                        focus_map=self.focus_map,
                        alpha=self.alpha,
                    ).item()
                    if current_disc_loss < best_loss:
                        best_loss = current_disc_loss
                        best_seed = int(seed)
                        tbar.set_postfix(best_loss=f"{best_loss:.4f}")

        if best_seed is not None:
            # `_material_select_batched_seeds`/`_compose_candidate` (the
            # batched fast-path used above) can disagree with the exact
            # per-layer `deterministic_gumbel_softmax` + `composite_image_disc`
            # path that actually produces the final output (e.g. differing
            # NaN-handling for extreme gumbel-softmax inputs at very low tau -
            # same root cause as the b158c32 discard in results.tsv). Verify
            # the winning candidate against the exact path before trusting it
            # - cheap (one extra composite) relative to the num_seeds batched
            # search, and prevents ever regressing below start_loss.
            with torch.no_grad():
                disc_global, _ = self.discretize_solution(
                    self.best_params, self.vis_tau, self.h, self.max_layers,
                    rng_seed=best_seed,
                )
                from autoforge.Helper.PruningHelper import disc_to_logits
                disc_global_logits = disc_to_logits(
                    disc_global, self.material_colors.shape[0], big_pos=1e5
                )
                effective_logits = self._apply_height_offset(
                    self.best_params["pixel_height_logits"],
                    self.best_params["height_offsets"],
                )
                verified_comp = composite_image_disc(
                    effective_logits,
                    disc_global_logits,
                    self.vis_tau,
                    self.vis_tau,
                    self.h,
                    self.max_layers,
                    self.material_colors,
                    self.material_TDs,
                    self.background,
                    rng_seed=best_seed,
                    compute_dtype=self.composite_compute_dtype,
                )
                verified_loss = compute_loss(
                    comp=verified_comp,
                    target=self.target,
                    focus_map=self.focus_map,
                    alpha=self.alpha,
                ).item()
            if verified_loss < start_loss:
                best_loss = verified_loss
            else:
                best_seed = None
                best_loss = start_loss

        if autoset_seed and best_seed is not None and best_loss < start_loss:
            self.best_seed = best_seed
        return best_seed, best_loss

    def __del__(self):
        """
        Clean up resources when the optimizer is destroyed.
        """
        if self.writer is not None:
            self.writer.close()
