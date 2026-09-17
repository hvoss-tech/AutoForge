import { create } from 'zustand'
import type { Filament, ColorSliderConfig, OptimizationSettings, JobStatus, ProjectState, PruningSettings, InitState, Snapshot } from '../types'

// Module-level undo stack — NOT in Zustand store to avoid infinite loops via subscribe
const UNDO_STACK: Snapshot[] = []
let UNDO_INDEX = -1

export interface HistoryEntry {
  timestamp: number
  label: string
  jobId: string | null
  jobStatus: string | null
}

function snapshotToHistoryEntry(s: Snapshot): HistoryEntry {
  return {
    timestamp: s.timestamp,
    label: s.label || 'State change',
    jobId: s.currentJobId ?? null,
    jobStatus: s.jobStatus ?? null,
  }
}

let pruningPollTimer: ReturnType<typeof setTimeout> | null = null

export type ToastLevel = 'error' | 'warning' | 'info'

export interface Toast {
  id: number
  message: string
  level: ToastLevel
}

let nextToastId = 1

// /api/optimize/latest returns the single most recent job ever run against
// this backend — it has no concept of "belongs to this browser session".
// A 'running'/'paused' job is worth reconnecting to, and a 'completed' one
// still has a real result worth showing, but a 'failed' or 'cancelled' job
// is just history: resurrecting it on every fresh page load showed
// "Optimization failed: ..." before the user had uploaded anything or
// clicked Run, for a run that may be from a completely unrelated session.
export function shouldRestoreJobOnLoad(status: string): boolean {
  return status === 'running' || status === 'paused' || status === 'completed'
}

// `inputImage` holds a `blob:` object URL right after a local upload (valid
// only in this browser tab, until the page reloads or is closed) — it's
// swapped for the durable `/uploads/<filename>` server path on the next
// `loadProjectState()` call, but nothing re-captures a snapshot at that
// point. Anything persisted (undo/redo snapshots, "Save Project" files)
// must use the durable path outright, or restoring/loading it later shows a
// permanently broken image even though the server still has the real file.
export function durableInputImageUrl(inputImage: string | null, settings: OptimizationSettings): string | null {
  if (inputImage && inputImage.startsWith('blob:') && settings.input_image) {
    return `/uploads/${settings.input_image}`
  }
  return inputImage
}

// 10 columns before anything has been run — 4 pre-populated (so a first-time
// user sees something to drag filaments onto) plus 6 empty slots to grow
// into, rather than the full ~15-40 columns a real optimizer/pruner result
// can produce. Once a job actually completes, `applySliders` replaces this
// wholesale with however many bands the result really has.
const defaultSliders: ColorSliderConfig[] = [
  { td: 2.0, layer: 8, depth_mm: 0.72, filament_uuid: '', enabled: true },
  { td: 3.0, layer: 13, depth_mm: 1.12, filament_uuid: '', enabled: true },
  { td: 8.0, layer: 20, depth_mm: 1.68, filament_uuid: '', enabled: true },
  { td: 5.0, layer: 27, depth_mm: 2.24, filament_uuid: '', enabled: true },
  ...Array.from({ length: 6 }, () => ({ td: 5.0, layer: 0, depth_mm: 0.0, filament_uuid: '', enabled: false })),
]

const THEME_STORAGE_KEY = 'autoforge-theme'

function readStoredTheme(): 'dark' | 'light' {
  try {
    const stored = localStorage.getItem(THEME_STORAGE_KEY)
    if (stored === 'light' || stored === 'dark') return stored
  } catch (_) {
    // localStorage unavailable (private browsing, etc.) — fall back to dark
  }
  return 'dark'
}

function applyTheme(theme: 'dark' | 'light') {
  document.documentElement.setAttribute('data-theme', theme)
  try {
    localStorage.setItem(THEME_STORAGE_KEY, theme)
  } catch (_) {
    // Ignore — theme just won't persist across reloads
  }
}

// Applied immediately at module load (before the first render) so there's
// no flash of the wrong theme while the store initializes.
const initialTheme = readStoredTheme()
applyTheme(initialTheme)

const defaultSettings: OptimizationSettings = {
  input_image: '',
  csv_file: '',
  json_file: '',
  output_folder: 'output',
  iterations: 6000,
  warmup_fraction: 1.0,
  learning_rate_warmup_fraction: 0.01,
  init_tau: 1.0,
  final_tau: 0.01,
  learning_rate: 0.015,
  layer_height: 0.04,
  max_layers: 75,
  min_layers: 0,
  background_height: 0.24,
  background_color: '#000000',
  auto_background_color: true,
  stl_output_size: 150,
  processing_reduction_factor: 2,
  nozzle_diameter: 0.4,
  early_stopping: 2000,
  perform_pruning: false,
  fast_pruning: true,
  fast_pruning_percent: 0.25,
  spike_removal: true,
  spike_threshold_layers: 1,
  pruning_max_colors: 100,
  pruning_max_swaps: 100,
  pruning_max_layer: 75,
  random_seed: 0,
  device: null,
  mps: false,
  run_name: null,
  tensorboard: false,
  num_init_rounds: 16,
  num_init_cluster_layers: -1,
  disable_visualization_for_gradio: 1,
  best_of: 1,
  discrete_check: 100,
  flatforge: false,
  cap_layers: 0,
  init_heightmap_method: 'kmeans',
  priority_mask: '',
  visualize: true,
}

const defaultPruningSettings: PruningSettings = {
  pruning_max_colors: 100,
  pruning_max_swaps: 100,
  pruning_max_layer: 75,
}

interface AppState {
  filaments: Filament[]
  filamentTypes: string[]
  filamentBrands: string[]
  activeFilaments: Filament[]
  colorSliders: ColorSliderConfig[]
  sliderLayerRange: { min: number; max: number }
  settings: OptimizationSettings
  currentJob: JobStatus | null
  inputImage: string | null
  previewImage: string | null
  previewVersion: number
  stlFile: string | null
  settingsModalOpen: boolean
  activeTab: string
  filterQuery: string
  filterType: string
  filterBrand: string
  pruningModalOpen: boolean
  pruningSettings: PruningSettings
  pruningJob: JobStatus | null
  initState: InitState
  hasRenderedInitPreview: boolean
  newFilamentModalOpen: boolean
  importModalOpen: boolean
  customLibraryLoaded: boolean
  editFilamentModalOpen: boolean
  editingFilament: Filament | null
  theme: 'dark' | 'light'
  toasts: Toast[]

  setFilaments: (filaments: Filament[]) => void
  setFilamentTypes: (types: string[]) => void
  setFilamentBrands: (brands: string[]) => void
  setActiveFilaments: (filaments: Filament[]) => void
  addActiveFilament: (filament: Filament) => void
  removeActiveFilament: (uuid: string) => void
  setSliders: (sliders: ColorSliderConfig[]) => void
  applySliders: (sliders: ColorSliderConfig[], range?: { min: number; max: number }) => void
  updateSlider: (index: number, updates: Partial<ColorSliderConfig>) => void
  setSettings: (settings: OptimizationSettings) => void
  setCurrentJob: (job: JobStatus | null) => void
  setSliderLayerRange: (range: { min: number; max: number }) => void
  setInputImage: (image: string | null) => void
  setPreviewImage: (image: string | null) => void
  bumpPreviewVersion: () => void
  setStlFile: (file: string | null) => void
  setSettingsModalOpen: (open: boolean) => void
  setActiveTab: (tab: string) => void
  setFilterQuery: (query: string) => void
  setFilterType: (type: string) => void
  setFilterBrand: (brand: string) => void
  setPruningModalOpen: (open: boolean) => void
  setPruningSettings: (settings: PruningSettings) => void
  setPruningJob: (job: JobStatus | null) => void
  setInitState: (state: Partial<InitState>) => void
  setHasRenderedInitPreview: (val: boolean) => void
  setNewFilamentModalOpen: (open: boolean) => void
  setImportModalOpen: (open: boolean) => void
  setCustomLibraryLoaded: (loaded: boolean) => void
  setEditFilamentModalOpen: (open: boolean) => void
  setEditingFilament: (filament: Filament | null) => void
  toggleTheme: () => void
  pushToast: (message: string, level?: ToastLevel) => void
  dismissToast: (id: number) => void
  startOptimization: () => Promise<string>
  pauseOptimization: (jobId: string) => Promise<void>
  resumeOptimization: (jobId: string) => Promise<void>
  cancelOptimization: (jobId: string) => Promise<void>
  startPruning: () => Promise<string>
  pausePruning: (jobId: string) => Promise<void>
  resumePruning: (jobId: string) => Promise<void>
  cancelPruning: (jobId: string) => Promise<void>
  runInit: () => Promise<void>
  loadProjectState: () => Promise<void>
  loadActiveFilaments: () => Promise<void>
  loadCurrentJob: () => Promise<void>
  loadProjectFromFile: (data: unknown) => Promise<void>

  // Undo/redo
  historyLength: number
  historyIndex: number
  historyEntries: HistoryEntry[]
  undo: () => Promise<void>
  redo: () => Promise<void>
  restoreToIndex: (index: number) => Promise<void>
  captureSnapshot: (label?: string) => void
}

export const useAppStore = create<AppState>((set, get) => ({
  filaments: [],
  filamentTypes: [],
  filamentBrands: [],
  activeFilaments: [],
  colorSliders: defaultSliders,
  sliderLayerRange: { min: 0, max: 75 },
  settings: defaultSettings,
  currentJob: null,
  inputImage: null,
  previewImage: null,
  previewVersion: 0,
  stlFile: null,
  settingsModalOpen: false,
  activeTab: 'PLA',
  filterQuery: '',
  filterType: '',
  filterBrand: '',
  pruningModalOpen: false,
  pruningSettings: defaultPruningSettings,
  pruningJob: null,
  initState: { status: 'idle', preview_image: null },
  hasRenderedInitPreview: false,
  newFilamentModalOpen: false,
  importModalOpen: false,
  customLibraryLoaded: false,
  editFilamentModalOpen: false,
  editingFilament: null,
  theme: initialTheme,
  historyLength: 0,
  historyIndex: -1,
  historyEntries: [],
  toasts: [],

  pushToast: (message, level = 'error') => {
    const id = nextToastId++
    set((state) => ({ toasts: [...state.toasts, { id, message, level }] }))
    // Errors stay until dismissed (they may need to be read/copied); info
    // and warning toasts self-clear so they don't pile up.
    if (level !== 'error') {
      setTimeout(() => get().dismissToast(id), 6000)
    }
  },
  dismissToast: (id) => set((state) => ({ toasts: state.toasts.filter((t) => t.id !== id) })),

  setFilaments: (filaments) => { set({ filaments }); queueCaptureSnapshot('Filament library updated') },
  setFilamentTypes: (types) => set({ filamentTypes: types }),
  setFilamentBrands: (brands) => set({ filamentBrands: brands }),
  setActiveFilaments: (filaments) => { set({ activeFilaments: filaments }); queueCaptureSnapshot('Active filaments changed') },
  setSliders: (sliders) => { set({ colorSliders: sliders }); queueCaptureSnapshot('Color slider edit') },
  applySliders: (sliders, range) => {
    set((state) => {
      // The optimizer/pruner can legitimately produce more or fewer bands
      // than any fixed column count — a material can recur in several
      // non-contiguous layer bands, and pruning changes the band count
      // further. Show exactly what it produced; padding to (or truncating
      // at) a fixed number here previously threw away real segments, which
      // then made the *next* render-with-sliders reconstruction (driven by
      // this same list) visibly wrong versus the true discrete solution.
      const merged = sliders.map((s, i) => ({ ...(state.colorSliders[i] ?? {}), ...s }))
      return {
        colorSliders: merged,
        ...(range && Number.isFinite(range.min) && Number.isFinite(range.max)
          ? { sliderLayerRange: { min: range.min, max: range.max } }
          : {}),
      }
    })
    queueCaptureSnapshot('Color slider edit')
  },
  addActiveFilament: async (filament) => {
    set((state) => {
      const exists = state.activeFilaments.some((f) => f.uuid === filament.uuid)
      if (exists) return state
      return { activeFilaments: [...state.activeFilaments, filament] }
    })
    try {
      await fetch('/api/filaments/active', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(filament),
      })
    } catch (e) {
      console.error('Failed to add active filament:', e)
      get().pushToast(`Failed to add "${filament.name}" to active filaments — it may not be saved on the server.`)
    }
    queueCaptureSnapshot('Added filament')
  },
  removeActiveFilament: async (uuid) => {
    set((state) => ({
      activeFilaments: state.activeFilaments.filter((f) => f.uuid !== uuid),
    }))
    try {
      await fetch(`/api/filaments/active/${uuid}`, { method: 'DELETE' })
    } catch (e) {
      console.error('Failed to remove active filament:', e)
      get().pushToast('Failed to remove active filament on the server — it may reappear after a reload.')
    }
    queueCaptureSnapshot('Removed filament')
  },

  updateSlider: (index, updates) => {
    set((state) => {
      const newSliders = [...state.colorSliders]
      newSliders[index] = { ...newSliders[index], ...updates }
      if (updates.td !== undefined || updates.layer !== undefined) {
        const lh = state.settings.layer_height || 0.04
        const layer = updates.layer !== undefined ? updates.layer : newSliders[index].layer
        newSliders[index].depth_mm = parseFloat((layer * lh).toFixed(2))
      }
      return { colorSliders: newSliders }
    })
    queueCaptureSnapshot('Color slider edit')
  },

  setSettings: (settings) => {
    set({ settings })
    queueCaptureSnapshot('Settings changed')
  },
  setCurrentJob: (job) => {
    const prevStatus = get().currentJob?.status
    set({ currentJob: job });
    // When the job completes, mark the STL as available for the 3D preview
    if (job && job.status === 'completed') {
      set({ stlFile: job.job_id })
    }
    // Only worth a history entry on a real transition (job started, or
    // reached a terminal state) — not on every progress tick, which would
    // otherwise queue (and coalesce away) a snapshot write every callback
    // for the whole duration of a run.
    if (job && job.status !== prevStatus) {
      const label = job.status === 'completed' ? 'Optimization completed'
        : job.status === 'failed' ? 'Optimization failed'
        : job.status === 'cancelled' ? 'Optimization cancelled'
        : job.status === 'running' && prevStatus !== 'paused' ? 'Optimization started'
        : 'Job status changed'
      queueCaptureSnapshot(label)
    }
  },
  setSliderLayerRange: (range) => set({ sliderLayerRange: range }),
  setInputImage: (image) => { set({ inputImage: image }); queueCaptureSnapshot('Input image changed') },
  setPreviewImage: (image) => set({ previewImage: image }),
  bumpPreviewVersion: () => set((state) => ({ previewVersion: state.previewVersion + 1 })),
  setStlFile: (file) => set({ stlFile: file }),
  setSettingsModalOpen: (open) => set({ settingsModalOpen: open }),
  setActiveTab: (tab) => set({ activeTab: tab }),
  setFilterQuery: (query) => set({ filterQuery: query }),
  setFilterType: (type) => set({ filterType: type }),
  setFilterBrand: (brand) => set({ filterBrand: brand }),
  setPruningModalOpen: (open) => set({ pruningModalOpen: open }),
  setPruningSettings: (settings) => set({ pruningSettings: settings }),
  setPruningJob: (job) => set({ pruningJob: job }),
  setInitState: (state) => set((prev) => ({ initState: { ...prev.initState, ...state } })),
  setHasRenderedInitPreview: (val) => set({ hasRenderedInitPreview: val }),
  setNewFilamentModalOpen: (open) => set({ newFilamentModalOpen: open }),
  setImportModalOpen: (open) => set({ importModalOpen: open }),
  setCustomLibraryLoaded: (loaded) => set({ customLibraryLoaded: loaded }),
  setEditFilamentModalOpen: (open) => set({ editFilamentModalOpen: open }),
  setEditingFilament: (filament) => set({ editingFilament: filament }),
  toggleTheme: () => set((state) => {
    const next = state.theme === 'dark' ? 'light' : 'dark'
    applyTheme(next)
    return { theme: next }
  }),

  loadActiveFilaments: async () => {
    try {
      const response = await fetch('/api/filaments/active')
      const data = await response.json()
      if (Array.isArray(data)) {
        set({ activeFilaments: data })
      }
    } catch {
      // Use empty list
    }
  },

  loadCurrentJob: async () => {
    try {
      const response = await fetch('/api/optimize/latest')
      if (!response.ok) return
      const job: JobStatus = await response.json()
      if (!shouldRestoreJobOnLoad(job.status)) return
      set({ currentJob: job })
      if (job.status === 'completed') set({ stlFile: job.job_id })
    } catch {
      // No jobs yet, or backend unreachable — start with none
    }
  },

  loadProjectFromFile: async (data) => {
    if (!data || typeof data !== 'object') throw new Error('Invalid project file')
    const parsed = data as Partial<{
      colorSliders: ColorSliderConfig[]
      settings: OptimizationSettings
      activeFilaments: Filament[]
      inputImage: string | null
    }>

    const state = get()

    // Sync the backend's active-filament list to match the file — clear
    // what's active now, then re-add what the file specifies. A filament
    // referenced by the file but missing from the current library (e.g.
    // loaded on a different machine/profile) is created first so the
    // reference doesn't silently dangle.
    if (Array.isArray(parsed.activeFilaments)) {
      for (const f of state.activeFilaments) {
        try {
          await fetch(`/api/filaments/active/${f.uuid}`, { method: 'DELETE' })
        } catch (_) {}
      }
      const libraryRes = await fetch('/api/filaments')
      const library: Filament[] = await libraryRes.json().catch(() => [])
      const libraryUuids = new Set(library.map((f) => f.uuid))
      let libraryChanged = false
      for (const f of parsed.activeFilaments) {
        try {
          if (!libraryUuids.has(f.uuid)) {
            await fetch('/api/filaments', {
              method: 'POST',
              headers: { 'Content-Type': 'application/json' },
              body: JSON.stringify(f),
            })
            libraryChanged = true
          }
          await fetch('/api/filaments/active', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(f),
          })
        } catch (_) {}
      }
      if (libraryChanged) {
        const refreshed = await fetch('/api/filaments')
        set({ filaments: await refreshed.json().catch(() => library) })
      }
      get().setActiveFilaments(parsed.activeFilaments)
    }

    if (parsed.settings) get().setSettings({ ...state.settings, ...parsed.settings })
    if (Array.isArray(parsed.colorSliders)) get().setSliders(parsed.colorSliders)
    if (parsed.inputImage !== undefined) get().setInputImage(parsed.inputImage)
  },

  runInit: async () => {
    const state = get()
    set({ initState: { status: 'initializing', preview_image: null } })
    try {
      // /api/init/run used to take no body at all and read a settings
      // singleton the rest of the app never writes to (POST /api/optimize/
      // start is what actually carries live settings) — it had no way to
      // know which image or filaments to use for anything but a stub.
      const response = await fetch('/api/init/run', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(state.settings),
      })
      if (!response.ok) {
        const err = await response.json().catch(() => ({ detail: `HTTP ${response.status}` }))
        console.error('[store] Init rejected:', err.detail)
        set({ initState: { status: 'idle', preview_image: null } })
        // "Already initializing" is a benign race (see the comment on
        // ActiveFilamentsPanel's effect) — not worth alarming the user
        // about; anything else (including an OOM from heightmap init,
        // which runs real GPU work) is not.
        if (!String(err.detail ?? '').includes('Already initializing')) {
          get().pushToast(`Failed to prepare preview: ${err.detail ?? `HTTP ${response.status}`}`)
        }
        return
      }
      const result = await response.json()
      set({ initState: { status: 'ready', preview_image: result.preview_image ?? null } })
      if (Number.isFinite(result.min_layer) && Number.isFinite(result.max_layer)) {
        set({ sliderLayerRange: { min: result.min_layer, max: result.max_layer } })
      }
      get().bumpPreviewVersion()
    } catch (e) {
      console.error('[store] Failed to run init:', e)
      set({ initState: { status: 'idle', preview_image: null } })
      get().pushToast(`Failed to prepare preview: ${e instanceof Error ? e.message : String(e)}`)
    }
  },

  startOptimization: async () => {
    const state = get()
    const response = await fetch('/api/optimize/start', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(state.settings),
    })
    if (!response.ok) {
      const err = await response.json().catch(() => ({ detail: 'Request failed' }))
      throw new Error(err.detail ?? `HTTP ${response.status}`)
    }
    const data = await response.json()
    if (pruningPollTimer) clearTimeout(pruningPollTimer)
    set({ currentJob: { ...data, total_iterations: data.total_iterations || 0, progress: 0, iteration: 0, loss: null, error: null, started_at: new Date().toISOString(), completed_at: null, preview_image: null }, stlFile: null, pruningJob: null })
    return data.job_id
  },

  pauseOptimization: async (jobId) => {
    const response = await fetch(`/api/optimize/pause/${jobId}`, { method: 'POST' })
    if (!response.ok) throw new Error(`HTTP ${response.status}`)
    set((state) => ({
      currentJob: state.currentJob ? { ...state.currentJob, status: 'paused' } : null,
    }))
  },

  resumeOptimization: async (jobId) => {
    const response = await fetch(`/api/optimize/resume/${jobId}`, { method: 'POST' })
    if (!response.ok) throw new Error(`HTTP ${response.status}`)
    set((state) => ({
      currentJob: state.currentJob ? { ...state.currentJob, status: 'running' } : null,
    }))
  },

  cancelOptimization: async (jobId) => {
    const response = await fetch(`/api/optimize/cancel/${jobId}`, { method: 'POST' })
    if (!response.ok) throw new Error(`HTTP ${response.status}`)
    set((state) => ({
      currentJob: state.currentJob ? { ...state.currentJob, status: 'cancelled' } : null,
    }))
  },

  startPruning: async () => {
    const state = get()
    if (state.currentJob?.status !== 'completed') {
      throw new Error("Run an optimization first — there's no result to prune yet.")
    }
    const response = await fetch('/api/pruning/start', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ ...state.pruningSettings, job_id: state.currentJob.job_id }),
    })
    if (!response.ok) {
      const err = await response.json().catch(() => ({ detail: 'Request failed' }))
      throw new Error(err.detail ?? `HTTP ${response.status}`)
    }
    const data = await response.json()
    set({ pruningJob: { ...data, progress: 0, iteration: 0, loss: null, error: null, started_at: new Date().toISOString(), completed_at: null, preview_image: null } })

    const jobId = data.job_id
    if (pruningPollTimer) clearTimeout(pruningPollTimer)
    const poll = async () => {
      try {
        const res = await fetch(`/api/optimize/status/${jobId}`)
        if (res.ok) {
          const job = await res.json()
          set({ pruningJob: job })
          if (job.status === 'completed' || job.status === 'failed' || job.status === 'cancelled') return
        }
      } catch {
        // Keep polling; backend may briefly be unavailable between prune stages
      }
      pruningPollTimer = setTimeout(poll, 1000)
    }
    poll()
    return data.job_id
  },

  pausePruning: async (jobId) => {
    const response = await fetch(`/api/optimize/pause/${jobId}`, { method: 'POST' })
    if (!response.ok) throw new Error(`HTTP ${response.status}`)
    set((state) => ({
      pruningJob: state.pruningJob ? { ...state.pruningJob, status: 'paused' } : null,
    }))
  },

  resumePruning: async (jobId) => {
    const response = await fetch(`/api/optimize/resume/${jobId}`, { method: 'POST' })
    if (!response.ok) throw new Error(`HTTP ${response.status}`)
    set((state) => ({
      pruningJob: state.pruningJob ? { ...state.pruningJob, status: 'running' } : null,
    }))
  },

  cancelPruning: async (jobId) => {
    const response = await fetch(`/api/optimize/cancel/${jobId}`, { method: 'POST' })
    if (!response.ok) throw new Error(`HTTP ${response.status}`)
    set((state) => ({
      pruningJob: state.pruningJob ? { ...state.pruningJob, status: 'cancelled' } : null,
    }))
  },

  loadProjectState: async () => {
    try {
      const response = await fetch('/api/project/state')
      const data = await response.json()
      // The backend now serialises this endpoint with plain (snake_case)
      // field names throughout — matching the frontend's ColorSliderConfig
      // type exactly, including nested fields like depth_mm and
      // filament_uuid — so this can be applied directly.
      if (data.color_sliders && data.color_sliders.length > 0) set({ colorSliders: data.color_sliders })
      // `settings` (and therefore which input image is selected) was never
      // restored here — after a reload the store fell back to hardcoded
      // defaults even though the backend still had the real settings and
      // uploaded image on disk, which left `input_image` empty and the Run
      // button stuck disabled ("upload an image") until the user
      // re-uploaded, despite nothing actually being wrong.
      if (data.settings) set({ settings: data.settings })
      if (data.settings?.input_image) {
        set({ inputImage: `/uploads/${data.settings.input_image}` })
      }
    } catch {
      // Use defaults
    }

    // Hydrate the undo/redo + History stack from the backend's persisted
    // snapshots so both survive a page reload — without this, UNDO_STACK
    // starts empty every mount and Undo/Redo/History are all inert until
    // the user makes a fresh edit.
    try {
      const histResp = await fetch('/api/state/history')
      const snapshots: Snapshot[] = await histResp.json()
      if (Array.isArray(snapshots) && snapshots.length > 0) {
        const ascending = [...snapshots].sort((a, b) => a.timestamp - b.timestamp)
        UNDO_STACK.length = 0
        UNDO_STACK.push(...ascending)
        if (UNDO_STACK.length > 50) UNDO_STACK.splice(0, UNDO_STACK.length - 50)
        UNDO_INDEX = UNDO_STACK.length - 1
        set({
          historyIndex: UNDO_INDEX,
          historyLength: UNDO_STACK.length,
          historyEntries: UNDO_STACK.map(snapshotToHistoryEntry),
        })
      }
    } catch {
      // No persisted history yet — undo/redo start fresh
    }
  },

  captureSnapshot: (label) => {
    const state = get()
    const snapshot: Snapshot = {
      timestamp: Date.now() / 1000,
      label: label || 'State change',
      activeFilaments: state.activeFilaments,
      colorSliders: state.colorSliders,
      settings: state.settings,
      inputImage: durableInputImageUrl(state.inputImage, state.settings),
      currentJobId: state.currentJob?.job_id ?? null,
      optimizationResultId: state.currentJob?.status === 'completed' ? state.currentJob?.job_id : null,
      jobStatus: state.currentJob?.status ?? null,
    }

    fetch('/api/state/snapshot', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(snapshot),
    }).catch(() => {})

    // Keep the on-reload project state fresh — GET /api/project/state is
    // read on every app mount, so if this never fires the endpoint keeps
    // serving whatever was last written (or nothing), and a stale/empty
    // snapshot silently overrides live defaults on next load.
    fetch('/api/project/state', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        color_sliders: state.colorSliders,
        settings: state.settings,
        active_filaments: state.activeFilaments,
      }),
    }).catch(() => {})

    // A snapshot taken after undoing drops everything ahead of it (the old
    // "redo" branch) — standard undo-stack semantics, and it also keeps
    // this array in lockstep with what's actually persisted server-side.
    UNDO_STACK.splice(UNDO_INDEX + 1)
    UNDO_STACK.push(snapshot)
    if (UNDO_STACK.length > 50) UNDO_STACK.shift()
    UNDO_INDEX = UNDO_STACK.length - 1
    set({
      historyIndex: UNDO_INDEX,
      historyLength: UNDO_STACK.length,
      historyEntries: UNDO_STACK.map(snapshotToHistoryEntry),
    })
  },

  restoreToIndex: async (index) => {
    const state = get()
    if (index < 0 || index >= UNDO_STACK.length) return
    const target = UNDO_STACK[index]
    if (!target) return

    // Stop running job — the state we're jumping to shouldn't have to
    // race whatever's currently in flight.
    if (state.currentJob && ['running', 'paused', 'pending'].includes(state.currentJob.status)) {
      try {
        await fetch(`/api/optimize/cancel/${state.currentJob.job_id}`, { method: 'POST' })
      } catch (_) {}
    }

    try {
      const resp = await fetch('/api/state/restore', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ timestamp: target.timestamp }),
      })
      if (!resp.ok) return

      UNDO_INDEX = index

      // Restore whichever job was "current" at snapshot time — without
      // this, undo/redo/History all lost the completed job's 3D result
      // (currentJob was unconditionally nulled), even though the snapshot
      // itself remembers exactly which job that was.
      let restoredJob: JobStatus | null = null
      if (target.currentJobId) {
        try {
          const jobResp = await fetch(`/api/optimize/status/${target.currentJobId}`)
          if (jobResp.ok) restoredJob = await jobResp.json()
        } catch (_) {}
      }

      set((prev) => ({
        activeFilaments: target.activeFilaments ?? prev.activeFilaments,
        colorSliders: target.colorSliders ?? prev.colorSliders,
        settings: target.settings ?? prev.settings,
        inputImage: target.inputImage ?? prev.inputImage,
        historyIndex: UNDO_INDEX,
        currentJob: restoredJob,
        stlFile: restoredJob && restoredJob.status === 'completed' ? restoredJob.job_id : null,
      }))
      get().bumpPreviewVersion()
    } catch (_) {}
  },

  undo: async () => {
    const state = get()
    if (state.historyIndex <= 0) return
    await get().restoreToIndex(state.historyIndex - 1)
  },

  redo: async () => {
    const state = get()
    if (state.historyIndex >= UNDO_STACK.length - 1) return
    await get().restoreToIndex(state.historyIndex + 1)
  },
}))
// Debounced snapshot queue — safe from infinite loops because captureSnapshot
// only calls set({ historyIndex, historyLength, historyEntries }), none of
// which re-trigger this (they're written via plain `set`, not the setter
// actions that call queueCaptureSnapshot).
let snapshotTimer: ReturnType<typeof setTimeout> | null = null
let pendingLabel: string | undefined

function queueCaptureSnapshot(label?: string) {
  if (label) pendingLabel = label
  if (snapshotTimer) clearTimeout(snapshotTimer)
  snapshotTimer = setTimeout(() => {
    const label = pendingLabel
    pendingLabel = undefined
    useAppStore.getState().captureSnapshot(label)
  }, 500)
}


