from __future__ import annotations
import re
from pydantic import BaseModel, Field, ConfigDict
from typing import Optional, Any, Literal
from datetime import datetime


def _to_camel(s: str) -> str:
    s = re.sub(r'_[a-z]', lambda m: m.group(0)[1].upper(), s)
    return s


class CamelCaseModel(BaseModel):
    model_config = ConfigDict(alias_generator=_to_camel, populate_by_name=True)


class Filament(CamelCaseModel):
    brand: str = ""
    name: str = ""
    color: str = "#ffffff"
    td: float = 0.0
    owned: bool = False
    uuid: str = ""
    filament_type: str = ""
    source: str = "user"


class ColorSliderConfig(CamelCaseModel):
    td: float = 5.0
    layer: int = 0
    depth_mm: float = 0.0
    filament_uuid: str = ""
    enabled: bool = False



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
    stl_output_size: int = Field(150, gt=0)
    processing_reduction_factor: int = Field(2, ge=1)
    nozzle_diameter: float = Field(0.4, gt=0)
    early_stopping: int = Field(2000, ge=1)
    perform_pruning: bool = False
    fast_pruning: bool = True
    fast_pruning_percent: float = Field(0.25, gt=0, le=1)
    spike_removal: bool = True
    spike_threshold_layers: int = 1
    pruning_max_colors: int = Field(100, ge=1)
    pruning_max_swaps: int = Field(100, ge=0)
    pruning_max_layer: int = Field(75, ge=1)
    random_seed: int = 0
    # Empty/None means auto-detect (CUDA/ROCm, then Apple Metal, then CPU).
    device: Optional[str] = None
    # Deprecated: Metal is auto-detected now. Kept so older clients that still
    # post `mps: true` keep working.
    mps: bool = False
    run_name: Optional[str] = None
    tensorboard: bool = False
    num_init_rounds: int = Field(16, ge=1)
    num_init_cluster_layers: int = Field(-1, ge=-1)
    disable_visualization_for_gradio: int = 1
    best_of: int = Field(1, ge=1)
    discrete_check: int = Field(100, ge=1)
    flatforge: bool = False
    cap_layers: int = 0
    # Same choices as the CLI; anything else used to silently run as kmeans.
    init_heightmap_method: Literal["kmeans", "depth"] = "kmeans"
    priority_mask: str = ""
    visualize: bool = False


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


class PruningSettings(CamelCaseModel):
    pruning_max_colors: int = 100
    pruning_max_swaps: int = 100
    pruning_max_layer: int = 75
    job_id: Optional[str] = None


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
