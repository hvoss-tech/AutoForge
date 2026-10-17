from __future__ import annotations
import re
from pydantic import BaseModel, Field, ConfigDict, field_validator, model_validator
from typing import Optional, Any, Literal
from datetime import datetime


def _to_camel(s: str) -> str:
    s = re.sub(r'_[a-z]', lambda m: m.group(0)[1].upper(), s)
    return s


class CamelCaseModel(BaseModel):
    model_config = ConfigDict(alias_generator=_to_camel, populate_by_name=True)


_HEX_COLOR = re.compile(r"^#?([0-9a-fA-F]{3}|[0-9a-fA-F]{6}|[0-9a-fA-F]{8})$")


def normalize_hex_color(value: Any) -> str:
    """``#RRGGBB`` for any hex spelling a library file may use (``RRGGBB``
    without the hash, ``#RGB``, ``#RRGGBBAA``), or ``ValueError``.

    The pipeline parses colors with ``hex_to_rgb``, which slices fixed
    offsets: ``#fff`` or ``red`` got into the library unchecked and only
    failed once a run started ("invalid literal for int() with base 16"),
    and a color without ``#`` rendered as no color at all in the UI."""
    text = str(value).strip()
    match = _HEX_COLOR.match(text)
    if not match:
        raise ValueError(f"{text!r} is not a hex color like #RRGGBB")
    digits = match.group(1)
    if len(digits) == 3:
        digits = "".join(c * 2 for c in digits)
    return "#" + digits[:6]


class Filament(CamelCaseModel):
    brand: str = ""
    name: str = ""
    color: str = "#ffffff"
    td: float = 0.0
    owned: bool = False
    uuid: str = ""
    filament_type: str = ""
    source: str = "user"

    @field_validator("color", mode="before")
    @classmethod
    def _normalize_color(cls, value: Any) -> str:
        return normalize_hex_color(value)


class ColorSliderConfig(CamelCaseModel):
    td: float = 5.0
    layer: int = 0
    depth_mm: float = 0.0
    filament_uuid: str = ""
    enabled: bool = False



class HeightPin(CamelCaseModel):
    """A filament suggested at a height: ``position`` is a fraction of the
    print's height (0 = the base, 1 = the top layer)."""
    uuid: str
    position: float = Field(0.5, ge=0.0, le=1.0)


class HeightColorPick(CamelCaseModel):
    """An image colour raised toward the top or lowered toward the base."""
    color: str
    tolerance: float = Field(15.0, ge=1.0, le=100.0)
    direction: Literal["raise", "lower"] = "raise"

    @field_validator("color")
    @classmethod
    def _hex(cls, v: str) -> str:
        return normalize_hex_color(v)


class HeightGuidanceSettings(CamelCaseModel):
    """Height guidance (--height_guidance): soft suggestions where colours
    sit in the print, weighed against the colour error by ``strength``."""
    strength: float = Field(1.0, ge=0.0, le=100.0)
    order: Literal["auto", "dark_to_light", "light_to_dark"] = "auto"
    pins: list[HeightPin] = []
    depth: bool = False
    depth_flip: bool = False
    colors: list[HeightColorPick] = []


class BorderSettings(CamelCaseModel):
    """A border around the exported model, on the base layer in the
    background filament (Helper/BorderMesh). It doesn't change the
    optimization: the 3D view and the export add it to the finished model."""
    shape: Literal["none", "rectangle", "oval", "circle", "outline"] = "none"
    width: float = 5.0
    corner_radius: float = 3.0
    hanger: Literal["none", "top", "corners"] = "none"
    hanger_diameter: float = 4.0

    @model_validator(mode="before")
    @classmethod
    def _clamp(cls, data: Any) -> Any:
        # Clamped instead of rejected: the border rides along with every
        # settings post, and one odd value must not block a run or a save.
        from dataclasses import asdict

        from autoforge.Helper.BorderMesh import BorderSpec

        if isinstance(data, BaseModel):
            data = data.model_dump()
        if not isinstance(data, dict):
            return {}
        snake = {re.sub(r"[A-Z]", lambda m: "_" + m.group(0).lower(), k): v for k, v in data.items()}
        return asdict(BorderSpec.from_dict(snake))

    def spec(self):
        from autoforge.Helper.BorderMesh import BorderSpec

        return BorderSpec(**self.model_dump())


class OptimizationSettings(CamelCaseModel):
    # Ranges are enforced here so a bad value is rejected up front with a
    # message naming the field, instead of starting a job that dies with
    # "division by zero" (processing_reduction_factor=0) or a scikit-learn
    # KMeans error (max_layers=0).
    input_image: str = ""
    csv_file: str = ""
    json_file: str = ""
    output_folder: str = "output"
    iterations: int = Field(6000, ge=1)
    warmup_fraction: float = Field(1.0, ge=0, le=1)
    learning_rate_warmup_fraction: float = Field(0.01, ge=0, le=1)
    init_tau: float = Field(1.0, gt=0)
    final_tau: float = Field(0.01, gt=0)
    learning_rate: float = Field(0.015, gt=0)
    layer_height: float = Field(0.04, gt=0)
    max_layers: int = Field(75, ge=1)
    min_layers: int = Field(0, ge=0)
    background_height: float = Field(0.24, ge=0)
    background_color: str = "#000000"
    auto_background_color: bool = True

    @field_validator("background_color", mode="before")
    @classmethod
    def _normalize_background_color(cls, value: Any) -> str:
        # Unchecked, a value like "red" was accepted and only failed once a
        # run had started (hex_to_rgb). Empty means the default.
        if value is None or str(value).strip() == "":
            return "#000000"
        return normalize_hex_color(value)
    stl_output_size: int = Field(150, gt=0)
    processing_reduction_factor: int = Field(2, ge=1)
    nozzle_diameter: float = Field(0.4, gt=0)
    early_stopping: int = Field(3000, ge=1)
    perform_pruning: bool = False
    fast_pruning: bool = True
    fast_pruning_percent: float = Field(0.25, gt=0, le=1)
    spike_removal: bool = True
    # 0 or less silently turned spike removal off.
    spike_threshold_layers: int = Field(1, ge=1)
    # Pixel height refine: cost per layer of height difference to each
    # neighbour, traded against colour error (0 = colour only).
    pixel_height_smoothness: float = Field(1.0, ge=0)
    # Edge bleed strength (see OptimizerHelper.bleed_layer_effect).
    edge_bleed: float = Field(0.25, ge=0, le=1)
    pruning_max_colors: int = Field(100, ge=1)
    pruning_max_swaps: int = Field(100, ge=0)
    pruning_max_layer: int = Field(75, ge=1)
    # Limits the optimizer itself keeps to while it runs (--constrained_opt),
    # instead of pruning a result down afterwards. None = unlimited. Colors
    # count the base filament, like the pruning dialog's "Max colors".
    max_colors: Optional[int] = Field(None, ge=2)
    max_swaps: Optional[int] = Field(None, ge=0)
    # WebUI-only orchestration flag (the CLI/pipeline never reads it): once
    # an optimization job completes, the frontend automatically starts one
    # pruning pass at the result's own current counts (no forced reduction)
    # so it's print-ready without the user opening the Pruning dialog by
    # hand. See appStore.setCurrentJob.
    auto_initial_prune: bool = True
    random_seed: int = 0
    # Empty/None means auto-detect (CUDA/ROCm, then Apple Metal, then CPU).
    device: Optional[str] = None
    # Deprecated: Metal is auto-detected now. Kept so older clients that still
    # post `mps: true` keep working.
    mps: bool = False
    run_name: Optional[str] = None
    tensorboard: bool = False
    # Extra rounds give identical results (see auto_forge --num_init_rounds).
    num_init_rounds: int = Field(1, ge=1)
    num_init_cluster_layers: int = Field(-1, ge=-1)
    disable_visualization_for_gradio: int = 1
    # Runs the optimization this many times (like the CLI's --best_of), then
    # waits for the user to pick one of the runs (see JobStatus.best_of_choosing);
    # the frontend asks before starting more than one.
    best_of: int = Field(1, ge=1, le=20)
    discrete_check: int = Field(100, ge=1)
    flatforge: bool = False
    cap_layers: int = Field(0, ge=0)
    # Same choices as the CLI; anything else used to silently run as kmeans.
    init_heightmap_method: Literal["kmeans", "depth"] = "kmeans"
    priority_mask: str = ""
    # How many times more a painted focus-area pixel counts (--priority_mask_strength).
    priority_mask_strength: float = Field(10.0, ge=1.0, le=1000.0)
    # Height guidance and its painted raise/lower regions (an uploads/ file
    # name, like priority_mask).
    height_guidance: Optional[HeightGuidanceSettings] = None
    height_regions: str = ""
    border: BorderSettings = Field(default_factory=BorderSettings)
    visualize: bool = False

    @model_validator(mode="after")
    def _check_ranges_between_fields(self):
        # Rejected up front with a message, instead of a run that anneals the
        # temperature upwards or a pruning pass that can never meet its limits.
        if self.final_tau > self.init_tau:
            raise ValueError(f"final_tau ({self.final_tau:g}) must not be larger than init_tau ({self.init_tau:g}).")
        if self.min_layers > self.pruning_max_layer:
            raise ValueError(
                f"min_layers ({self.min_layers}) must not be larger than pruning_max_layer ({self.pruning_max_layer})."
            )
        return self

    def base_height_error(self) -> Optional[str]:
        """The CLI refuses a base that isn't a whole number of layers
        (perform_basic_check); the webui ran it anyway, so every color layer
        of the model sat off the printer's layer grid and the swap
        instructions named heights the slicer never prints. Checked with a
        tolerance: 0.28 / 0.04 is 7.000000000000001 in floating point."""
        ratio = self.background_height / self.layer_height
        if abs(ratio - round(ratio)) > 1e-6:
            return (
                f"Base height ({self.background_height:g} mm) must be a multiple of the "
                f"layer height ({self.layer_height:g} mm), e.g. "
                f"{max(1, round(ratio)) * self.layer_height:.2f} mm."
            )
        return None


class JobStatus(CamelCaseModel):
    job_id: str = ""
    status: str = "pending"
    progress: float = 0.0
    iteration: int = 0
    total_iterations: int = 0
    loss: Optional[float] = None
    error: Optional[str] = None
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    preview_image: Optional[str] = None
    phase: Optional[str] = None
    # The input image this job ran on (and whether it produced the
    # flatforge-style per-material STLs instead of a single final_model.stl).
    # Both are read back on a page reload, where "the latest job" has no
    # other way of being matched against the image currently on screen —
    # without them a completed job from a different image was restored (and
    # even exportable) right under another image.
    input_image: Optional[str] = None
    flatforge: Optional[bool] = None
    # Live counts of the solution as it stands, reported while pruning runs.
    # Pruning's whole purpose is to bring these down, so the dialog shows them
    # changing as it happens rather than only once the job is finished.
    result_colors: Optional[int] = None
    result_swaps: Optional[int] = None
    result_layers: Optional[int] = None
    # How the result follows the height guidance (see
    # sliders.guidance_report_from_optimizer); None without guidance.
    guidance_report: Optional[dict[str, Any]] = None
    # Best of N runs (OptimizationSettings.best_of): the losses of the runs
    # finished so far, and once completed, which of them (0-based) was kept.
    best_of_losses: Optional[list[float]] = None
    best_of_run: Optional[int] = None
    # True while every run of the series has finished and the job waits for
    # the user to pick the one to keep (POST /api/optimize/best-of/.../pick);
    # best_of_run is then the lowest-loss run.
    best_of_choosing: Optional[bool] = None
    # Which pruning pass is running, and how many are allowed, when pruning
    # repeats itself until it stops improving.
    pruning_pass: Optional[int] = None
    pruning_max_passes: Optional[int] = None
    # The loss pruning started from, so the UI can show what it has gained
    # (`loss` carries the live value). Lower is closer to the input image.
    pruning_start_loss: Optional[float] = None
    # Automatic pruning (see PruningSettings.mode): the loss it must stay
    # under, one entry per finished pass ({pass, colors, swaps, layers, loss,
    # kept}), and why it stopped (helpers/auto_prune.py outcomes, or
    # "start_above_limit" when the result was over the limit to begin with).
    auto_target_loss: Optional[float] = None
    auto_passes: Optional[list[dict]] = None
    auto_outcome: Optional[str] = None


class PruningSettings(CamelCaseModel):
    # Colors count the base; a negative swap limit reached pruning's final
    # check as a cryptic failure.
    pruning_max_colors: int = Field(100, ge=1)
    pruning_max_swaps: int = Field(100, ge=0)
    pruning_max_layer: int = Field(75, ge=1)
    job_id: Optional[str] = None
    # "auto" ignores the three limits above: it lowers them pass by pass
    # until the loss would go over `target_loss`, and keeps the last pass
    # that was still under it.
    mode: Literal["manual", "auto"] = "manual"
    target_loss: Optional[float] = Field(None, gt=0)
    # Keep pruning the same result until a pass stops finding an
    # improvement. Pruning is a greedy search that restarts from whatever it
    # is given, so repeated passes keep gaining — but only up to a point, and
    # nobody wants to sit there pressing the button.
    auto_repeat: bool = False
    # Safety net for auto_repeat: a pass is minutes of GPU work, and a metric
    # that oscillates by a hair could otherwise loop forever.
    max_passes: int = Field(25, ge=1, le=200)
    # Two optional polish passes that run *before* the reduction phases.
    # Every pruning phase is a greedy search scored against the current
    # solution, so a better starting point improves everything after it.
    # Both are strictly non-worsening (see FilamentOptimizer.rng_seed_search
    # and .polish_height_offsets); seed search is cheap enough to default on,
    # height fine-tuning is the slower of the two so it defaults off.
    seed_search: bool = True
    seed_search_count: int = Field(200, ge=1, le=20000)
    fine_tune_height: bool = False
    fine_tune_steps: int = Field(50, ge=1, le=2000)
    # Pruning the auto-preview ("__init__" as job_id) — a loaded project, or
    # an image colored by hand: its heightmap plus these color layers make
    # the result to prune.
    settings: Optional[OptimizationSettings] = None
    sliders: list[dict[str, Any]] = Field(default_factory=list)
    active_filaments: list[dict[str, Any]] = Field(default_factory=list)
    background_color: Optional[str] = None
    base: Optional[dict[str, Any]] = None

    @model_validator(mode="after")
    def _auto_needs_target(self):
        if self.mode == "auto" and self.target_loss is None:
            raise ValueError("Automatic pruning needs a target loss.")
        return self


class StateSnapshot(CamelCaseModel):
    timestamp: float = 0.0
    label: str = ""
    active_filaments: list[Filament] = Field(default_factory=list)
    color_sliders: list[ColorSliderConfig] = Field(default_factory=list)
    settings: OptimizationSettings = Field(default_factory=OptimizationSettings)
    input_image: Optional[str] = None
    current_job_id: Optional[str] = None
    optimization_result_id: Optional[str] = None
    job_status: Optional[str] = None
    # {"min": int, "max": int} — the slider track range belongs to a result
    # just like its sliders do; without it, undoing from one result to
    # another kept the newer result's layer range.
    slider_layer_range: Optional[dict[str, int]] = None


class ProjectState(CamelCaseModel):
    color_sliders: list[ColorSliderConfig] = Field(default_factory=list)
    settings: OptimizationSettings = Field(default_factory=OptimizationSettings)
    active_filaments: list[Filament] = Field(default_factory=list)
