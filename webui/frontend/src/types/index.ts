export interface Filament {
  brand: string
  name: string
  color: string
  td: number
  owned: boolean
  uuid: string
  filament_type: string
  source?: 'user' | 'swatch' | 'filamentcolors'
}

export interface ColorSliderConfig {
  td: number
  layer: number
  depth_mm: number
  filament_uuid: string
  enabled: boolean
}

export interface OptimizationSettings {
  input_image: string
  csv_file: string
  json_file: string
  output_folder: string
  iterations: number
  warmup_fraction: number
  learning_rate_warmup_fraction: number
  init_tau: number
  final_tau: number
  learning_rate: number
  layer_height: number
  max_layers: number
  min_layers: number
  background_height: number
  background_color: string
  auto_background_color: boolean
  stl_output_size: number
  processing_reduction_factor: number
  nozzle_diameter: number
  early_stopping: number
  perform_pruning: boolean
  fast_pruning: boolean
  fast_pruning_percent: number
  spike_removal: boolean
  spike_threshold_layers: number
  pruning_max_colors: number
  pruning_max_swaps: number
  pruning_max_layer: number
  /** Run one unlimited pruning pass automatically once an optimization
   * completes (see appStore.setCurrentJob). */
  auto_initial_prune: boolean
  random_seed: number
  /** Torch device override ('cuda', 'cuda:1', 'mps', 'cpu'); null = auto-detect. */
  device: string | null
  /** @deprecated Metal is auto-detected now; use `device: 'mps'` instead. */
  mps: boolean
  run_name: string | null
  tensorboard: boolean
  num_init_rounds: number
  num_init_cluster_layers: number
  disable_visualization_for_gradio: number
  best_of: number
  discrete_check: number
  flatforge: boolean
  cap_layers: number
  init_heightmap_method: string
  priority_mask: string
  /** How many times more a painted focus-area pixel counts (2–100 in the UI). */
  priority_mask_strength: number
  visualize: boolean
}

export interface JobStatus {
  job_id: string
  status: 'pending' | 'running' | 'completed' | 'failed' | 'cancelled' | 'paused'
  progress: number
  iteration: number
  total_iterations: number
  loss: number | null
  error: string | null
  started_at: string | null
  completed_at: string | null
  preview_image: string | null
  phase: string | null
  /** Live counts of the solution while pruning runs (see models.JobStatus).
   * Null until the job reports them — pruning is the only thing that does. */
  result_colors?: number | null
  result_swaps?: number | null
  result_layers?: number | null
  /** Which auto-repeat pruning pass is running, out of how many allowed. */
  pruning_pass?: number | null
  pruning_max_passes?: number | null
  /** The loss pruning started from; `loss` carries the live value. */
  pruning_start_loss?: number | null
  /** The input image this job ran on, and whether it's a FlatForge run (see
   * models.JobStatus) — both absent on jobs recorded before these fields
   * existed. Used to tell whether a restored/latest job actually belongs to
   * the image currently on screen. */
  input_image?: string | null
  flatforge?: boolean | null
}

export interface PruningSettings {
  pruning_max_colors: number
  pruning_max_swaps: number
  pruning_max_layer: number
  /** Keep pruning until a pass stops finding an improvement. */
  auto_repeat: boolean
  max_passes: number
  /** Look for a better material discretization before pruning. */
  seed_search: boolean
  seed_search_count: number
  /** Polish the per-pixel heights before pruning. */
  fine_tune_height: boolean
  fine_tune_steps: number
}

export interface InitState {
  status: 'idle' | 'initializing' | 'ready' | string
  preview_image: string | null
}

export interface ProjectState {
  color_sliders: ColorSliderConfig[]
  settings: OptimizationSettings
  active_filaments: Filament[]
}

export interface Snapshot {
  timestamp: number
  label: string
  activeFilaments?: Filament[]
  colorSliders?: ColorSliderConfig[]
  settings?: OptimizationSettings
  inputImage?: string | null
  currentJobId?: string | null
  optimizationResultId?: string | null
  jobStatus?: string | null
  sliderLayerRange?: { min: number; max: number } | null
}
