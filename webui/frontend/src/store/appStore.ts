import { create } from 'zustand'
import type { Filament, ColorSliderConfig, OptimizationSettings, JobStatus, ProjectState, PruningSettings, InitState, Snapshot } from '../types'
import { jobStatusFromControlResponse } from '../lib/jobControl'
import {
  acceptsPreviewUpdate,
  describeSettingsChange,
  describeSliderEdit,
  jobBelongsToImage,
  mergeHistoryLabels,
  resolveRestoredJob,
  restoredImage,
} from '../lib/history'
import { describeApiError } from '../lib/apiError'
import * as bandOps from '../lib/bandOps'
import { inheritRunInputs, readStoredRunInputs, storeRunInputs, type RunInputs } from '../lib/staleResult'
import { appendLossPoint, type LossPoint } from '../lib/lossHistory'
import { suggestPruningLimits, resultCounts, type PruningCounts } from '../lib/pruning'
import { buildPrintPlan } from '../lib/printPlan'
import { projectFileName, projectFingerprint, projectNameFromFile } from '../lib/project'

// Module-level undo stack — NOT in Zustand store to avoid infinite loops via subscribe
const UNDO_STACK: Snapshot[] = []
let UNDO_INDEX = -1
const MAX_HISTORY = 50

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
// Bumped by stopPruningPoll(). clearTimeout alone can't stop a poll whose
// status fetch is already in flight: it re-armed itself afterwards and, once
// the prune finished, made that result the current job — under a new image
// or a new project the user had switched to in the meantime.
let pruningPollGeneration = 0
function stopPruningPoll() {
  pruningPollGeneration += 1
  if (pruningPollTimer) clearTimeout(pruningPollTimer)
  pruningPollTimer = null
}
const MAX_TOASTS = 5

// POST /api/init/run requests from this page still waiting for an answer.
// While one is out, the server's init status describes the *previous* image
// until the new init actually starts, so status polls must not act on it.
let initRequestsInFlight = 0
export function isInitRequestInFlight(): boolean {
  return initRequestsInFlight > 0
}

export type ToastLevel = 'error' | 'warning' | 'info'

export type ImageView = 'original' | 'result' | 'split' | 'compare' | 'difference'

export type ServerConnection = 'ok' | 'reconnecting' | 'lost'

export type InitOutcome = 'ready' | 'busy' | 'error'

/** GET /api/sliders/base — see derive_base_from_result (helpers/sliders.py). */
export interface ResolvedBase {
  color: string
  height_mm: number
  layers: number
  filament_uuid: string
  auto: boolean
}

const TUTORIAL_SEEN_KEY = 'autoforge-tutorial-seen'

export function hasSeenTutorial(): boolean {
  return readStorage(TUTORIAL_SEEN_KEY) === '1'
}

export function markTutorialSeen(): void {
  writeStorage(TUTORIAL_SEEN_KEY, '1')
}

export interface ConfirmRequest {
  id: number
  title: string
  message: string
  confirmLabel: string
  danger?: boolean
}

let confirmResolver: ((ok: boolean) => void) | null = null
let nextConfirmId = 1

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

// Has this tab ever shown a job of its own? See acceptsPreviewUpdate().
let hasTrackedJob = false
export function markJobTracked(): void {
  hasTrackedJob = true
}
export function acceptsPreviewFor(jobId: string | undefined, currentJobId: string | undefined): boolean {
  return acceptsPreviewUpdate(jobId, currentJobId, hasTrackedJob)
}

async function refreshCurrentJob(jobId: string): Promise<void> {
  try {
    const response = await fetch(`/api/optimize/status/${jobId}`)
    if (response.ok) useAppStore.getState().setCurrentJob(await response.json())
  } catch (_) {}
}

// Replace the backend's active-filament list wholesale. /api/optimize/start
// reads the backend's list, not the store's, so anything that swaps the
// store's list (undo/redo, loading a project file) must sync it or the next
// Run silently uses different filaments than the UI shows.
async function syncActiveFilamentsToServer(filaments: Filament[]): Promise<void> {
  const response = await fetch('/api/filaments/active', {
    method: 'PUT',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(filaments),
  })
  if (!response.ok) throw new Error(`HTTP ${response.status}`)
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


function readStorage(key: string): string | null {
  try {
    return localStorage.getItem(key)
  } catch {
    return null
  }
}

function writeStorage(key: string, value: string | null) {
  try {
    if (value === null) localStorage.removeItem(key)
    else localStorage.setItem(key, value)
  } catch {
    // not remembered — fine
  }
}

const PROJECT_NAME_KEY = 'autoforge-project-name'
const SAVED_FINGERPRINT_KEY = 'autoforge-saved-project-fingerprint'
const AUTO_SAVE_LIBRARY_KEY = 'autoforge-library-auto-save'

/** Filament edits are written as you make them unless this was turned off.
 * On by default: the library is a list of the filaments you own, not a
 * document you compose — losing an edit by closing the dialog is never what
 * anyone wanted. */
export function readAutoSaveLibrary(): boolean {
  return readStorage(AUTO_SAVE_LIBRARY_KEY) !== '0'
}

function downloadBlob(blob: Blob, filename: string) {
  const url = URL.createObjectURL(blob)
  const a = document.createElement('a')
  a.href = url
  a.download = filename
  document.body.appendChild(a)
  a.click()
  a.remove()
  URL.revokeObjectURL(url)
}

/** Fingerprint of everything a saved project file holds (see saveProjectToFile). */
export function currentProjectFingerprint(state: Pick<AppState, 'projectName' | 'inputImage' | 'settings' | 'colorSliders' | 'activeFilaments'>): string {
  return projectFingerprint({
    name: state.projectName,
    inputImage: durableInputImageUrl(state.inputImage, state.settings),
    settings: state.settings,
    colorSliders: state.colorSliders,
    activeFilamentUuids: state.activeFilaments.map((f) => f.uuid),
  })
}

// A fresh project starts without color layers. Four pre-positioned but
// unassigned "Empty" columns used to suggest something was already set up;
// the layers panel now explains how to get bands (run, or drag a filament).
const defaultSliders: ColorSliderConfig[] = []

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

export const defaultSettings: OptimizationSettings = {
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
  auto_initial_prune: true,
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
  priority_mask_strength: 10,
  visualize: true,
}

const defaultPruningSettings: PruningSettings = {
  pruning_max_colors: 100,
  pruning_max_swaps: 100,
  pruning_max_layer: 75,
  auto_repeat: false,
  max_passes: 25,
  seed_search: true,
  seed_search_count: 200,
  fine_tune_height: false,
  fine_tune_steps: 50,
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
  /** Bumped when the 3D mesh file must be fetched again. previewVersion
   * (the result image) moves with it, except for this tab's own slider
   * edits: those recolor the mesh in place (lib/meshColors). */
  meshVersion: number
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
  /** The import dialog was opened by the one-time first-start offer to
   * import HueForge's personal library, not by the Import button. */
  importModalHueforgeOffer: boolean
  customLibraryLoaded: boolean
  editFilamentModalOpen: boolean
  editingFilament: Filament | null
  theme: 'dark' | 'light'
  toasts: Toast[]
  /** The user changed sliders by hand since the last optimizer/pruner stack
   * arrived — a new run would replace those edits. */
  slidersEditedByHand: boolean
  confirmRequest: ConfirmRequest | null
  historyOpen: boolean
  bottomTab: 'layers' | 'plan'
  imageView: ImageView
  /** The band selected in the layer list / color core (a colorSliders index). */
  selectedBand: number | null
  /** The band under the pointer in either of them. */
  hoveredBand: number | null
  /** The image, layers and result were picked up from the previous session. */
  sessionRestored: boolean
  /** A restored image that already has its preview — no heightmap init needed. */
  initSkipImage: string | null
  /** Settings and active filaments each run started with, by job id. */
  runInputsByJob: Record<string, RunInputs>
  lossHistory: LossPoint[]
  lossJobId: string | null
  /** Result counts when the last pruning started, for a before/after summary. */
  pruningBaseline: PruningCounts | null
  projectName: string
  /** Fingerprint of the project as last saved to a file (null: never saved). */
  savedProjectFingerprint: string | null
  /** The base/background slab as the pipeline actually built it — with
   * `auto_background_color` on, the color used is picked from the active
   * filaments by the backend, so settings alone can't describe it. */
  resolvedBase: ResolvedBase | null
  /** The tutorial overlay (first visit, or the ? button in the top bar). */
  tutorialOpen: boolean
  /** Write filament edits to the library as they're made. */
  autoSaveLibrary: boolean
  /** Whether the running job's server can be reached: 'reconnecting' while
   * the progress socket is being re-established, 'lost' once that gave up
   * and status polls fail too (the server stopped or crashed). */
  serverConnection: ServerConnection
  /** Bumped by retryServerConnection(); the job socket reconnects from
   * scratch whenever it changes. */
  connectionRetryNonce: number

  setFilaments: (filaments: Filament[]) => void
  setFilamentTypes: (types: string[]) => void
  setFilamentBrands: (brands: string[]) => void
  setActiveFilaments: (filaments: Filament[]) => void
  addActiveFilament: (filament: Filament) => void
  removeActiveFilament: (uuid: string) => void
  setSliders: (sliders: ColorSliderConfig[], label?: string, opts?: { handEdited?: boolean }) => void
  addBand: (filament?: Filament) => void
  removeBand: (index: number) => void
  requestConfirm: (request: Omit<ConfirmRequest, 'id'>) => Promise<boolean>
  resolveConfirm: (ok: boolean) => void
  setHistoryOpen: (open: boolean) => void
  setBottomTab: (tab: 'layers' | 'plan') => void
  setImageView: (view: ImageView) => void
  setSelectedBand: (index: number | null) => void
  setHoveredBand: (index: number | null) => void
  moveBand: (from: number, to: number) => void
  insertBandAbove: (index: number) => boolean
  sortBandsByLayer: () => void
  restoreSessionImage: () => Promise<void>
  dismissSessionRestored: () => void
  startNewProject: () => void
  setProjectName: (name: string) => void
  saveProjectToFile: () => void
  applySliders: (sliders: ColorSliderConfig[], range?: { min: number; max: number }) => void
  updateSlider: (index: number, updates: Partial<ColorSliderConfig>) => void
  setSettings: (settings: OptimizationSettings) => void
  setCurrentJob: (job: JobStatus | null) => void
  setSliderLayerRange: (range: { min: number; max: number }) => void
  setInputImage: (image: string | null) => void
  setPreviewImage: (image: string | null) => void
  bumpPreviewVersion: () => void
  bumpImageVersion: () => void
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
  openHueforgeImportOffer: () => void
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
  startPruning: (overrides?: Partial<PruningSettings>) => Promise<string>
  pausePruning: (jobId: string) => Promise<void>
  resumePruning: (jobId: string) => Promise<void>
  cancelPruning: (jobId: string) => Promise<void>
  /** Builds the auto-preview. 'busy' means another init was already
   * running, which is a benign race and worth retrying — unlike 'error'. */
  runInit: () => Promise<InitOutcome>
  loadProjectState: () => Promise<void>
  loadActiveFilaments: () => Promise<void>
  loadCurrentJob: () => Promise<void>
  loadProjectFromFile: (data: unknown, fileName?: string) => Promise<void>
  loadBaseColor: (jobId?: string) => Promise<void>
  setResolvedBase: (base: ResolvedBase | null) => void
  setBaseFilament: (filament: Filament) => void
  setTutorialOpen: (open: boolean) => void
  setAutoSaveLibrary: (on: boolean) => void
  setServerConnection: (state: ServerConnection) => void
  retryServerConnection: () => void
  /** The uploaded focus-area mask (a file in uploads/), or '' for none. */
  setPriorityMask: (filename: string) => void
  applyUploadedImage: (filename: string, displayUrl: string) => Promise<void>

  // Undo/redo
  historyLength: number
  historyIndex: number
  historyEntries: HistoryEntry[]
  undo: () => Promise<void>
  redo: () => Promise<void>
  restoreToIndex: (index: number) => Promise<void>
  clearHistory: () => Promise<void>
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
  meshVersion: 0,
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
  importModalHueforgeOffer: false,
  customLibraryLoaded: false,
  editFilamentModalOpen: false,
  editingFilament: null,
  theme: initialTheme,
  historyLength: 0,
  historyIndex: -1,
  historyEntries: [],
  toasts: [],
  slidersEditedByHand: false,
  confirmRequest: null,
  historyOpen: false,
  bottomTab: 'layers',
  imageView: 'original',
  selectedBand: null,
  hoveredBand: null,
  sessionRestored: false,
  initSkipImage: null,
  runInputsByJob: readStoredRunInputs(),
  lossHistory: [],
  lossJobId: null,
  pruningBaseline: null,
  projectName: readStorage(PROJECT_NAME_KEY) ?? '',
  savedProjectFingerprint: readStorage(SAVED_FINGERPRINT_KEY),
  resolvedBase: null,
  // First visit opens it by itself; after that it's the ? button's job.
  tutorialOpen: !hasSeenTutorial(),
  autoSaveLibrary: readAutoSaveLibrary(),
  serverConnection: 'ok',
  connectionRetryNonce: 0,

  requestConfirm: (request) => {
    // Only one at a time: a newer request cancels an unanswered older one.
    confirmResolver?.(false)
    return new Promise<boolean>((resolve) => {
      confirmResolver = resolve
      set({ confirmRequest: { ...request, id: nextConfirmId++ } })
    })
  },
  resolveConfirm: (ok) => {
    const resolve = confirmResolver
    confirmResolver = null
    set({ confirmRequest: null })
    resolve?.(ok)
  },
  setHistoryOpen: (open) => set({ historyOpen: open }),
  setBottomTab: (tab) => set({ bottomTab: tab }),
  setImageView: (view) => set({ imageView: view }),
  setSelectedBand: (index) => set({ selectedBand: index }),
  setHoveredBand: (index) => set({ hoveredBand: index }),

  moveBand: (from, to) => {
    const { colorSliders, settings } = get()
    const edit = bandOps.moveBand(colorSliders, from, to, settings.layer_height || 0.04)
    set({ colorSliders: edit.sliders, slidersEditedByHand: true, selectedBand: edit.index, hoveredBand: null })
    queueCaptureSnapshot(`Moved band ${from + 1} to position ${edit.index + 1}`)
  },
  insertBandAbove: (index) => {
    const { colorSliders, settings, sliderLayerRange } = get()
    const edit = bandOps.insertBandAbove(
      colorSliders,
      index,
      { filament_uuid: '', td: 5 },
      sliderLayerRange.max || settings.max_layers || 75,
      settings.layer_height || 0.04,
    )
    if (!edit) return false
    set({ colorSliders: edit.sliders, slidersEditedByHand: true, selectedBand: edit.index, hoveredBand: null })
    queueCaptureSnapshot(`Inserted a band above band ${index + 1}`)
    return true
  },
  sortBandsByLayer: () => {
    const { colorSliders, selectedBand } = get()
    const edit = bandOps.sortBandsByLayer(colorSliders, selectedBand ?? 0)
    set({ colorSliders: edit.sliders, selectedBand: selectedBand === null ? null : edit.index, hoveredBand: null })
    queueCaptureSnapshot('Sorted bands by layer')
  },

  restoreSessionImage: async () => {
    const { settings, inputImage, currentJob } = get()
    if (inputImage || !settings.input_image) return
    const url = `/uploads/${settings.input_image}`
    let exists = false
    try {
      exists = (await fetch(url, { method: 'HEAD' })).ok
    } catch (_) {}
    if (get().inputImage) return
    if (!exists) {
      // The uploaded file is gone (uploads folder cleared): start without it
      // rather than pointing Run at a missing file.
      set((state) => ({ settings: { ...state.settings, input_image: '' } }))
      return
    }
    // A restored result, or a preview the server still holds, doesn't need
    // the (slow, GPU) heightmap init again just because the page reloaded.
    let skipInit = !!currentJob
    if (!skipInit) {
      try {
        const status = await (await fetch('/api/init/status')).json()
        skipInit = status.status === 'ready' || status.status === 'initializing'
      } catch (_) {}
    }
    if (get().inputImage) return
    set({ inputImage: url, sessionRestored: true, initSkipImage: skipInit ? url : null })
  },
  dismissSessionRestored: () => set({ sessionRestored: false }),
  startNewProject: () => {
    stopPruningPoll()
    set((state) => ({
      inputImage: null,
      settings: { ...state.settings, input_image: '', priority_mask: '' },
      colorSliders: [],
      sliderLayerRange: { min: 0, max: state.settings.max_layers || 75 },
      currentJob: null,
      stlFile: null,
      previewImage: null,
      initState: { status: 'idle', preview_image: null },
      initSkipImage: null,
      hasRenderedInitPreview: false,
      pruningJob: null,
      pruningBaseline: null,
      resolvedBase: null,
      sessionRestored: false,
      slidersEditedByHand: false,
      selectedBand: null,
      hoveredBand: null,
      imageView: 'original',
      lossHistory: [],
      lossJobId: null,
      projectName: '',
      savedProjectFingerprint: null,
    }))
    writeStorage(PROJECT_NAME_KEY, null)
    writeStorage(SAVED_FINGERPRINT_KEY, null)
    // The server still holds the previous image's auto-preview (mesh file,
    // "ready" status and its optimizer's GPU memory); without this the 3D
    // panel would serve that mesh again the moment a new image is added.
    fetch('/api/init/reset', { method: 'POST' }).catch(() => {})
    queueCaptureSnapshot('Started a new project')
  },
  setProjectName: (name) => {
    set({ projectName: name })
    writeStorage(PROJECT_NAME_KEY, name || null)
  },
  saveProjectToFile: () => {
    flushPendingSnapshot()
    const state = get()
    const data = {
      version: 1,
      name: state.projectName,
      savedAt: new Date().toISOString(),
      colorSliders: state.colorSliders,
      settings: state.settings,
      activeFilaments: state.activeFilaments,
      // A blob: URL only works in this tab; store the durable server path.
      inputImage: durableInputImageUrl(state.inputImage, state.settings),
    }
    downloadBlob(new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' }), projectFileName(state.projectName))
    const fingerprint = currentProjectFingerprint(state)
    set({ savedProjectFingerprint: fingerprint })
    writeStorage(SAVED_FINGERPRINT_KEY, fingerprint)
  },

  addBand: (filament) => {
    set((state) => {
      const top = Math.max(0, ...state.colorSliders.filter((c) => c.enabled).map((c) => c.layer))
      const layer = Math.min(state.sliderLayerRange.max || 75, top + 5) || 5
      const lh = state.settings.layer_height || 0.04
      const band: ColorSliderConfig = {
        td: filament?.td ?? 5,
        layer,
        depth_mm: parseFloat((layer * lh).toFixed(2)),
        filament_uuid: filament?.uuid ?? '',
        enabled: true,
      }
      // Reuse a disabled, never-positioned column before growing the list.
      const free = state.colorSliders.findIndex((c) => !c.enabled && c.layer === 0)
      const colorSliders = [...state.colorSliders]
      if (free >= 0) colorSliders[free] = band
      else colorSliders.push(band)
      return { colorSliders, slidersEditedByHand: true, selectedBand: free >= 0 ? free : colorSliders.length - 1 }
    })
    queueCaptureSnapshot(filament ? `Added a band of ${filament.name}` : 'Added a band')
  },
  removeBand: (index) => {
    set((state) => ({ colorSliders: state.colorSliders.filter((_, i) => i !== index), slidersEditedByHand: true, selectedBand: null, hoveredBand: null }))
    queueCaptureSnapshot(`Removed band ${index + 1}`)
  },

  pushToast: (message, level = 'error') => {
    const id = nextToastId++
    set((state) => {
      const toasts = [...state.toasts, { id, message, level }]
      // A repeated error source (a flaky connection, a retried failing
      // request) used to stack toasts forever since only info/warning ones
      // self-cleared — cap how many stay on screen at once.
      return { toasts: toasts.length > MAX_TOASTS ? toasts.slice(toasts.length - MAX_TOASTS) : toasts }
    })
    // Errors stay longer (they may need to be read/copied) but still clear
    // eventually; info and warning toasts self-clear quickly.
    setTimeout(() => get().dismissToast(id), level === 'error' ? 20000 : 6000)
  },
  dismissToast: (id) => set((state) => ({ toasts: state.toasts.filter((t) => t.id !== id) })),

  // Not an undo step: the library list is refreshed by searches/filters and
  // after imports, and isn't part of a snapshot anyway.
  setFilaments: (filaments) => set({ filaments }),
  setFilamentTypes: (types) => set({ filamentTypes: types }),
  setFilamentBrands: (brands) => set({ filamentBrands: brands }),
  setActiveFilaments: (filaments) => { set({ activeFilaments: filaments }); queueCaptureSnapshot('Active filaments changed') },
  setSliders: (sliders, label, opts) => {
    // A slider's TD following a library filament's edited TD (see
    // EditFilamentModal) is a sync, not something the user laid out by
    // hand — marking it hand-edited made an unrelated library edit trigger
    // "Replace your color layers?" on the next Run, and queued a
    // misleading "Edited color layers" undo entry for a change the user
    // never made to the sliders themselves.
    const handEdited = opts?.handEdited ?? true
    set({ colorSliders: sliders, ...(handEdited ? { slidersEditedByHand: true } : {}) })
    queueCaptureSnapshot(label ?? 'Edited color layers')
  },
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
        slidersEditedByHand: false,
        ...(range && Number.isFinite(range.min) && Number.isFinite(range.max)
          ? { sliderLayerRange: { min: range.min, max: range.max } }
          : {}),
      }
    })
    // No snapshot: this is the optimizer/pruner pushing its stack, not a user
    // edit. Recording each live preview update as "Color slider edit" filled
    // the history with noise during a run and — once an undo landed in the
    // pre-run phase, where the init poll kept re-applying a stack — created
    // new entries every second that wiped the redo branch.
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
    queueCaptureSnapshot(`Added ${filament.name}`)
  },
  removeActiveFilament: async (uuid) => {
    const removedName = get().activeFilaments.find((f) => f.uuid === uuid)?.name ?? 'filament'
    set((state) => ({
      activeFilaments: state.activeFilaments.filter((f) => f.uuid !== uuid),
    }))
    try {
      await fetch(`/api/filaments/active/${uuid}`, { method: 'DELETE' })
    } catch (e) {
      console.error('Failed to remove active filament:', e)
      get().pushToast('Failed to remove active filament on the server — it may reappear after a reload.')
    }
    queueCaptureSnapshot(`Removed ${removedName}`)
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
      return { colorSliders: newSliders, slidersEditedByHand: true }
    })
    const filamentName = updates.filament_uuid
      ? [...get().activeFilaments, ...get().filaments].find((f) => f.uuid === updates.filament_uuid)?.name
      : undefined
    queueCaptureSnapshot(describeSliderEdit(index, updates, filamentName))
  },

  setSettings: (settings) => {
    const before = get().settings
    set({ settings })
    queueCaptureSnapshot(describeSettingsChange(before as unknown as Record<string, unknown>, settings as unknown as Record<string, unknown>))
  },
  setCurrentJob: (job) => {
    const prevStatus = get().currentJob?.status
    if (job) markJobTracked()
    set({ currentJob: job });
    if (job && (job.status === 'running' || job.status === 'paused') && job.loss !== null && job.loss !== undefined) {
      const { lossJobId, lossHistory } = get()
      set({ lossHistory: appendLossPoint(lossJobId === job.job_id ? lossHistory : [], job.iteration, job.loss), lossJobId: job.job_id })
    }
    // When the job completes, mark the STL as available for the 3D preview
    if (job && job.status === 'completed') {
      set({ stlFile: job.job_id })
    }
    // The finished result is the one job transition worth an undo step (it's
    // a state you can return to); started/failed/cancelled change nothing
    // the user can restore.
    if (job && job.status === 'completed' && prevStatus !== 'completed') {
      // The run is what resolves the base color (auto-selection happens
      // inside the pipeline), so pick it up now that there's a result.
      get().loadBaseColor(job.job_id)
      queueCaptureSnapshot('Optimization completed')
      // Auto-clean the fresh result once, so it's print-ready without the
      // user opening the Pruning dialog by hand: the limits are set to the
      // result's own current counts, so nothing is forced out, but the
      // color-seed search and swap-position pass still merge/move anything
      // that doesn't cost accuracy. Never re-fires for a pruning job's own
      // completion — that path sets currentJob directly (see startPruning's
      // poll loop) rather than through this action; the job_id check is
      // cheap extra insurance against that assumption changing later.
      if (get().settings.auto_initial_prune && !job.job_id.startsWith('prune-')) {
        const current = resultCounts(buildPrintPlan(get().colorSliders, [], get().settings))
        const limits = suggestPruningLimits(current)
        set({ pruningBaseline: current })
        get().startPruning({
          pruning_max_colors: limits.colors,
          pruning_max_swaps: limits.swaps,
          pruning_max_layer: limits.layers,
          auto_repeat: false,
          seed_search: true,
          fine_tune_height: false,
        }).catch((e) => {
          console.error('Automatic initial pruning failed:', e)
        })
      }
    }
  },
  setSliderLayerRange: (range) => set({ sliderLayerRange: range }),
  setInputImage: (image) => { set({ inputImage: image }); queueCaptureSnapshot('Input image changed') },
  setPreviewImage: (image) => set({ previewImage: image }),
  bumpPreviewVersion: () => set((state) => ({ previewVersion: state.previewVersion + 1, meshVersion: state.meshVersion + 1 })),
  bumpImageVersion: () => set((state) => ({ previewVersion: state.previewVersion + 1 })),
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
  setImportModalOpen: (open) => set({ importModalOpen: open, importModalHueforgeOffer: false }),
  openHueforgeImportOffer: () => set({ importModalOpen: true, importModalHueforgeOffer: true }),
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
      // Called after loadProjectState() (App.tsx), so settings.input_image
      // already reflects the persisted project's image, if any.
      if (job.status === 'completed' && !jobBelongsToImage(job.input_image, get().settings.input_image)) {
        return
      }
      markJobTracked()
      set({ currentJob: job })
      if (job.status === 'completed') {
        set({ stlFile: job.job_id })
        get().loadBaseColor(job.job_id)
      }
    } catch {
      // No jobs yet, or backend unreachable — start with none
    }
  },

  loadProjectFromFile: async (data, fileName) => {
    if (!data || typeof data !== 'object') throw new Error('Invalid project file')
    const parsed = data as Partial<{
      colorSliders: ColorSliderConfig[]
      settings: OptimizationSettings
      activeFilaments: Filament[]
      inputImage: string | null
    }>

    const state = get()

    // A loaded project's image is a different one than whatever was on
    // screen (definitely — it's a different project), so every result
    // derived from the *previous* image must go, the same way
    // applyUploadedImage() clears it for a fresh upload. Leaving any of it
    // in place is what showed the previous project's PLY under the loaded
    // one, and re-rendered the new project's colors onto the old heightmap.
    stopPruningPoll()
    await fetch('/api/init/reset', { method: 'POST' }).catch(() => {})
    set({
      currentJob: null,
      stlFile: null,
      previewImage: null,
      initState: { status: 'idle', preview_image: null },
      initSkipImage: null,
      hasRenderedInitPreview: false,
      pruningJob: null,
      pruningBaseline: null,
      resolvedBase: null,
      lossHistory: [],
      lossJobId: null,
      imageView: 'original',
      hoveredBand: null,
      // The loaded sliders may themselves reflect someone's earlier hand
      // work, but nothing has been hand-edited in *this* session yet.
      slidersEditedByHand: false,
    })

    // Sync the backend's active-filament list to match the file — clear
    // what's active now, then re-add what the file specifies. A filament
    // referenced by the file but missing from the current library (e.g.
    // loaded on a different machine/profile) is created first so the
    // reference doesn't silently dangle.
    if (Array.isArray(parsed.activeFilaments)) {
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
        } catch (_) {}
      }
      try {
        await syncActiveFilamentsToServer(parsed.activeFilaments)
      } catch (e) {
        get().pushToast(`Failed to save the project's active filaments on the server: ${e instanceof Error ? e.message : String(e)}`)
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
    get().setProjectName(projectNameFromFile((data as { name?: unknown }).name, fileName ?? ''))
    const fingerprint = currentProjectFingerprint(get())
    set({ savedProjectFingerprint: fingerprint, sessionRestored: false, selectedBand: null })
    writeStorage(SAVED_FINGERPRINT_KEY, fingerprint)
  },

  setResolvedBase: (base) => set({ resolvedBase: base }),

  loadBaseColor: async (jobId) => {
    try {
      // Pinned to a specific job when the caller knows which result is on
      // screen — without it this answered for "the newest completed job",
      // which showed the wrong base color after undoing to an older result.
      const url = jobId ? `/api/sliders/base?job_id=${encodeURIComponent(jobId)}` : '/api/sliders/base'
      const response = await fetch(url)
      if (!response.ok) return
      const data = await response.json()
      if (data && data.base) set({ resolvedBase: data.base as ResolvedBase })
    } catch (_) {
      // The base row falls back to `settings.background_color`.
    }
  },

  setBaseFilament: (filament) => {
    // Picking the base by hand turns auto-selection off — otherwise the next
    // run would quietly overwrite the choice with the closest match again.
    const { settings, activeFilaments } = get()
    if (!activeFilaments.some((f) => f.uuid === filament.uuid)) get().addActiveFilament(filament)
    set({
      resolvedBase: {
        color: filament.color,
        height_mm: settings.background_height,
        layers: Math.round((settings.background_height || 0) / (settings.layer_height || 0.04)),
        filament_uuid: filament.uuid,
        auto: false,
      },
    })
    get().setSettings({ ...settings, background_color: filament.color, auto_background_color: false })
  },

  setTutorialOpen: (open) => {
    if (!open) markTutorialSeen()
    set({ tutorialOpen: open })
  },

  setAutoSaveLibrary: (on) => {
    writeStorage(AUTO_SAVE_LIBRARY_KEY, on ? '1' : '0')
    set({ autoSaveLibrary: on })
  },

  setServerConnection: (state) => {
    if (get().serverConnection !== state) set({ serverConnection: state })
  },
  retryServerConnection: () => set((s) => ({ serverConnection: 'reconnecting', connectionRetryNonce: s.connectionRetryNonce + 1 })),

  setPriorityMask: (filename) => {
    const before = get().settings
    if (before.priority_mask === filename) return
    set({ settings: { ...before, priority_mask: filename } })
    queueCaptureSnapshot(filename ? 'Painted focus areas' : 'Cleared focus areas')
  },

  applyUploadedImage: async (filename, displayUrl) => {
    // A new photo invalidates everything derived from the old one. Leaving
    // any of it in place is what made the 3D panel show the previous image:
    // `stlFile`/`currentJob` kept the old result's mesh on screen, and the
    // backend kept serving the old auto-preview mesh (its state was still
    // "ready") until the new heightmap init finished.
    stopPruningPoll()
    // Before the store changes, not after: setting `inputImage` is what makes
    // useAutoPreviewInit start the new heightmap init, and a reset landing
    // after that had started would wipe the init it had just kicked off.
    await fetch('/api/init/reset', { method: 'POST' }).catch(() => {})
    // The focus-area mask was painted over the old picture's shapes.
    const settings = { ...get().settings, input_image: filename, priority_mask: '' }
    set({
      settings,
      inputImage: displayUrl,
      // The previous image's heightmap bounds must not keep clamping band
      // edits until the new init lands and reports its own range.
      sliderLayerRange: { min: 0, max: settings.max_layers || 75 },
      initState: { status: 'idle', preview_image: null },
      initSkipImage: null,
      hasRenderedInitPreview: false,
      previewImage: null,
      currentJob: null,
      stlFile: null,
      pruningJob: null,
      pruningBaseline: null,
      resolvedBase: null,
      lossHistory: [],
      lossJobId: null,
      imageView: 'original',
      sessionRestored: false,
      selectedBand: null,
      hoveredBand: null,
    })
    queueCaptureSnapshot('Input image changed')
  },

  runInit: async () => {
    const state = get()
    // Captured now, not read again after the await: settings.input_image
    // can change while this request is in flight (a new upload, a history
    // step), and this response must not be applied under a different photo
    // than the one it was built for.
    const forImage = state.settings.input_image
    set({ initState: { status: 'initializing', preview_image: null } })
    initRequestsInFlight += 1
    // Whether *this* call is the only one out. If a sibling init request is
    // also in flight, it owns resolving `initState` when it settles — this
    // call forcing it back to 'idle' on its own rejection raced the sibling,
    // flapped the status, and let useAutoPreviewInit's effect (which only
    // remembers 'error' outcomes in `failedFor`) re-fire into a request loop.
    const isOnlyRequest = () => initRequestsInFlight <= 1
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
        const detail = describeApiError(await response.json().catch(() => null), response.status)
        console.error('[store] Init rejected:', detail)
        if (isOnlyRequest()) set({ initState: { status: 'idle', preview_image: null } })
        // "Already initializing" is a benign race (see the comment on
        // ActiveFilamentsPanel's effect); a 409 for a build superseded by a
        // reset (image changed mid-build) is exactly as benign — the image
        // that mattered just changed out from under this call, and
        // useAutoPreviewInit will already have kicked off (or will kick
        // off) the build that actually matters. Neither is worth alarming
        // the user about; anything else (including an OOM from heightmap
        // init, which runs real GPU work) is not.
        if (detail.includes('Already initializing') || detail.includes('reset while it was being prepared')) {
          return 'busy'
        }
        get().pushToast(`Failed to prepare preview: ${detail}`)
        return 'error'
      }
      const result = await response.json()
      if (get().settings.input_image !== forImage) {
        // The image changed while this request was out — this result is
        // for a photo that's no longer on screen. Applying it would show
        // one image's heightmap/base color under a different one.
        return 'busy'
      }
      set({ initState: { status: 'ready', preview_image: result.preview_image ?? null } })
      if (Number.isFinite(result.min_layer) && Number.isFinite(result.max_layer)) {
        set({ sliderLayerRange: { min: result.min_layer, max: result.max_layer } })
      }
      // Which filament the pipeline chose for the base is only known here.
      if (result.base) set({ resolvedBase: result.base as ResolvedBase })
      get().bumpPreviewVersion()
      return 'ready'
    } catch (e) {
      console.error('[store] Failed to run init:', e)
      if (isOnlyRequest()) set({ initState: { status: 'idle', preview_image: null } })
      get().pushToast(`Failed to prepare preview: ${e instanceof Error ? e.message : String(e)}`)
      return 'error'
    } finally {
      initRequestsInFlight -= 1
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
      throw new Error(describeApiError(await response.json().catch(() => null), response.status))
    }
    const data = await response.json()
    markJobTracked()
    stopPruningPoll()
    const runInputsByJob = storeRunInputs(get().runInputsByJob, data.job_id, {
      settings: { ...state.settings },
      filamentUuids: state.activeFilaments.map((f) => f.uuid),
    })
    set({ runInputsByJob, lossHistory: [], lossJobId: data.job_id, pruningBaseline: null })
    set({ currentJob: { ...data, total_iterations: data.total_iterations || 0, progress: 0, iteration: 0, loss: null, error: null, started_at: new Date().toISOString(), completed_at: null, preview_image: null, phase: 'Preparing' }, stlFile: null, pruningJob: null, previewImage: null, imageView: 'original' })
    return data.job_id
  },

  pauseOptimization: async (jobId) => {
    const response = await fetch(`/api/optimize/pause/${jobId}`, { method: 'POST' })
    if (!response.ok) throw new Error(`HTTP ${response.status}`)
    const status = jobStatusFromControlResponse(await response.json().catch(() => null), 'paused')
    if (status !== 'paused') {
      // The job had already finished — pick up its final state (and, for a
      // completed job, its result) instead of the one we asked for.
      await refreshCurrentJob(jobId)
      return
    }
    set((state) => ({
      currentJob: state.currentJob ? { ...state.currentJob, status } : null,
    }))
  },

  resumeOptimization: async (jobId) => {
    const response = await fetch(`/api/optimize/resume/${jobId}`, { method: 'POST' })
    if (!response.ok) throw new Error(`HTTP ${response.status}`)
    const status = jobStatusFromControlResponse(await response.json().catch(() => null), 'running')
    if (status !== 'running') {
      // The job had already finished — pick up its final state (and, for a
      // completed job, its result) instead of the one we asked for.
      await refreshCurrentJob(jobId)
      return
    }
    set((state) => ({
      currentJob: state.currentJob ? { ...state.currentJob, status } : null,
    }))
  },

  cancelOptimization: async (jobId) => {
    const response = await fetch(`/api/optimize/cancel/${jobId}`, { method: 'POST' })
    if (!response.ok) throw new Error(`HTTP ${response.status}`)
    const status = jobStatusFromControlResponse(await response.json().catch(() => null), 'cancelled')
    if (status !== 'cancelled') {
      // The job had already finished — pick up its final state (and, for a
      // completed job, its result) instead of the one we asked for.
      await refreshCurrentJob(jobId)
      return
    }
    set((state) => ({
      currentJob: state.currentJob ? { ...state.currentJob, status } : null,
    }))
  },

  startPruning: async (overrides) => {
    const state = get()
    if (state.currentJob?.status !== 'completed') {
      throw new Error("Run an optimization first — there's no result to prune yet.")
    }
    const pruningSettings = { ...state.pruningSettings, ...overrides }
    const response = await fetch('/api/pruning/start', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ ...pruningSettings, job_id: state.currentJob.job_id }),
    })
    if (!response.ok) {
      throw new Error(describeApiError(await response.json().catch(() => null), response.status))
    }
    const data = await response.json()
    set({ pruningJob: { ...data, progress: 0, iteration: 0, loss: null, error: null, started_at: new Date().toISOString(), completed_at: null, preview_image: null } })

    // The pruning job's own id (`data.job_id`, e.g. "prune-xxxx") gets its
    // own real, independent output directory and pipeline result once it
    // completes (see api/pruning.py) — it is no longer just a progress
    // tracker for the original optimization job. Once that happens, this
    // becomes the active job: every subsequent action (viewing the result,
    // a follow-up prune, a history snapshot captured from here on) must
    // point at this id, not the one pruning started from — that job's own
    // files stop changing once this prune begins, so anything still
    // pointing at it would keep showing the *pre-this-prune* result.
    const jobId = data.job_id
    const sourceJobId = state.currentJob.job_id
    stopPruningPoll()
    const generation = pruningPollGeneration
    const poll = async () => {
      try {
        const res = await fetch(`/api/optimize/status/${jobId}`)
        if (generation !== pruningPollGeneration) return
        if (res.ok) {
          const job = await res.json()
          set({ pruningJob: job })
          if (job.status === 'completed') {
            // The pruned result is built from the same run's settings and
            // filaments. Without its own entry here it had none, so the
            // "Out of date" badge could never show for a pruned result.
            const runInputsByJob = inheritRunInputs(get().runInputsByJob, sourceJobId, job.job_id)
            set({ currentJob: job, stlFile: job.job_id, runInputsByJob })
            get().loadBaseColor(job.job_id)
            queueCaptureSnapshot('Pruning completed')
          }
          if (job.status === 'completed' || job.status === 'failed' || job.status === 'cancelled') return
        }
      } catch {
        // Keep polling; backend may briefly be unavailable between prune stages
      }
      if (generation !== pruningPollGeneration) return
      pruningPollTimer = setTimeout(poll, 1000)
    }
    poll()
    return data.job_id
  },

  pausePruning: async (jobId) => {
    const response = await fetch(`/api/optimize/pause/${jobId}`, { method: 'POST' })
    if (!response.ok) throw new Error(`HTTP ${response.status}`)
    const status = jobStatusFromControlResponse(await response.json().catch(() => null), 'paused')
    set((state) => ({
      pruningJob: state.pruningJob ? { ...state.pruningJob, status } : null,
    }))
  },

  resumePruning: async (jobId) => {
    const response = await fetch(`/api/optimize/resume/${jobId}`, { method: 'POST' })
    if (!response.ok) throw new Error(`HTTP ${response.status}`)
    const status = jobStatusFromControlResponse(await response.json().catch(() => null), 'running')
    set((state) => ({
      pruningJob: state.pruningJob ? { ...state.pruningJob, status } : null,
    }))
  },

  cancelPruning: async (jobId) => {
    const response = await fetch(`/api/optimize/cancel/${jobId}`, { method: 'POST' })
    if (!response.ok) throw new Error(`HTTP ${response.status}`)
    const status = jobStatusFromControlResponse(await response.json().catch(() => null), 'cancelled')
    set((state) => ({
      pruningJob: state.pruningJob ? { ...state.pruningJob, status } : null,
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
      // Settings restore including `input_image`; the image itself is shown
      // again by restoreSessionImage() once the job is known. Blanking it
      // here used to leave a half-restored page: the previous result, layers
      // and "Done" on screen, but an empty image panel saying "Upload".
      if (data.settings) set({ settings: data.settings })
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
        if (UNDO_STACK.length > MAX_HISTORY) UNDO_STACK.splice(0, UNDO_STACK.length - MAX_HISTORY)
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
      sliderLayerRange: state.sliderLayerRange,
    }

    // An edit made after undoing abandons the redo branch; the server must
    // drop it too or it reappears (interleaved by time) on the next reload.
    const abandonsRedo = UNDO_INDEX >= 0 && UNDO_INDEX < UNDO_STACK.length - 1
    const query = abandonsRedo ? `?discard_after=${UNDO_STACK[UNDO_INDEX].timestamp}` : ''
    fetch(`/api/state/snapshot${query}`, {
      method: 'POST',
      keepalive: true,
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(snapshot),
    }).catch(() => {})

    persistProjectState(state)

    UNDO_STACK.splice(UNDO_INDEX + 1)
    UNDO_STACK.push(snapshot)
    if (UNDO_STACK.length > MAX_HISTORY) UNDO_STACK.shift()
    UNDO_INDEX = UNDO_STACK.length - 1
    set({
      historyIndex: UNDO_INDEX,
      historyLength: UNDO_STACK.length,
      historyEntries: UNDO_STACK.map(snapshotToHistoryEntry),
    })
  },

  clearHistory: () => runHistoryExclusive(async () => {
    if (snapshotTimer) {
      clearTimeout(snapshotTimer)
      snapshotTimer = null
      pendingLabel = undefined
    }
    await fetch('/api/state/history', { method: 'DELETE' }).catch(() => {})
    UNDO_STACK.length = 0
    UNDO_INDEX = -1
    // The current state becomes the new starting point.
    get().captureSnapshot('History cleared')
  }),

  restoreToIndex: (index) => runHistoryExclusive(async () => {
    flushPendingSnapshot()
    await applySnapshot(index)
  }),

  undo: () => runHistoryExclusive(async () => {
    flushPendingSnapshot()
    if (UNDO_INDEX <= 0) return
    const undone = UNDO_STACK[UNDO_INDEX].label
    await applySnapshot(UNDO_INDEX - 1)
    get().pushToast(`Undid: ${undone}`, 'info')
  }),

  redo: () => runHistoryExclusive(async () => {
    flushPendingSnapshot()
    if (UNDO_INDEX >= UNDO_STACK.length - 1) return
    await applySnapshot(UNDO_INDEX + 1)
    get().pushToast(`Redid: ${UNDO_STACK[UNDO_INDEX].label}`, 'info')
  }),
}))
// Debounced snapshot queue — safe from infinite loops because captureSnapshot
// only calls set({ historyIndex, historyLength, historyEntries }), none of
// which re-trigger this (they're written via plain `set`, not the setter
// actions that call queueCaptureSnapshot).
let snapshotTimer: ReturnType<typeof setTimeout> | null = null
let pendingLabel: string | undefined
// Set for the whole duration of an undo/redo/restoreToIndex — see
// applySnapshot(). An edit's debounced snapshot must not fire while a
// restore is in flight: applySnapshot's own final setState would land after
// it and silently overwrite the edit right back out of the visible state,
// leaving a history entry that doesn't match what's on screen.
let applyInFlight = false
let deferredCaptureLabel: string | undefined

function queueCaptureSnapshot(label?: string) {
  if (applyInFlight) {
    if (label) deferredCaptureLabel = mergeHistoryLabels(deferredCaptureLabel, label)
    return
  }
  if (label) pendingLabel = mergeHistoryLabels(pendingLabel, label)
  if (snapshotTimer) clearTimeout(snapshotTimer)
  snapshotTimer = setTimeout(() => {
    snapshotTimer = null
    const label = pendingLabel
    pendingLabel = undefined
    useAppStore.getState().captureSnapshot(label)
  }, 500)
}

// An edit made moments before Undo is still waiting in the debounce; record
// it first so Undo steps back *from* it rather than silently discarding it.
function flushPendingSnapshot() {
  if (!snapshotTimer) return
  clearTimeout(snapshotTimer)
  snapshotTimer = null
  const label = pendingLabel
  pendingLabel = undefined
  useAppStore.getState().captureSnapshot(label)
}

// Closing or reloading the tab within the debounce window used to drop the
// last edit entirely (it's also what persists the project state). The
// requests use `keepalive`, so they still go out while the page unloads.
if (typeof window !== 'undefined') {
  window.addEventListener('pagehide', flushPendingSnapshot)
}

// Undo/redo/History clicks run strictly one after another. Each one awaits
// network calls, and they used to read `historyIndex` before the previous
// one had updated it — pressing Ctrl+Z twice quickly restored the same step
// twice instead of going back two.
let historyQueue: Promise<void> = Promise.resolve()
function runHistoryExclusive(fn: () => Promise<void>): Promise<void> {
  const run = historyQueue.then(fn)
  historyQueue = run.catch(() => {})
  return run
}

function persistProjectState(state: Pick<AppState, 'colorSliders' | 'settings' | 'activeFilaments'>) {
  // GET /api/project/state is what a reload starts from.
  fetch('/api/project/state', {
    method: 'POST',
    keepalive: true,
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      color_sliders: state.colorSliders,
      settings: state.settings,
      active_filaments: state.activeFilaments,
    }),
  }).catch(() => {})
}

async function applySnapshot(index: number) {
  const target = UNDO_STACK[index]
  if (!target) return
  const store = useAppStore.getState()
  // UNDO_INDEX (and the store's historyIndex) move only right before the
  // final setState below, after every await here has settled — not here.
  // A captureSnapshot debounce that fires during one of those awaits reads
  // UNDO_INDEX to decide where to splice/append; moving it this early made
  // it act as if the restore had already landed while the visible state
  // (and the network calls it depends on) hadn't, so a concurrent edit's
  // snapshot was spliced in at the wrong spot and then clobbered by this
  // function's own restore a moment later — the edit vanished from the
  // screen even though a (mismatched) history entry for it remained.
  //
  // Belt and suspenders: also hold off the debounce itself for the whole
  // restore (not just the index update), so an edit mid-restore can't be
  // captured — and clobbered by this function's own final setState — at
  // all. It replays once this restore has fully landed.
  applyInFlight = true
  try {
    return await applySnapshotBody(index, target, store)
  } finally {
    applyInFlight = false
    if (deferredCaptureLabel !== undefined) {
      const label = deferredCaptureLabel
      deferredCaptureLabel = undefined
      queueCaptureSnapshot(label)
    }
  }
}

async function applySnapshotBody(index: number, target: Snapshot, store: AppState) {
  if (target.activeFilaments) {
    try {
      await syncActiveFilamentsToServer(target.activeFilaments)
    } catch (_) {
      store.pushToast('Failed to restore active filaments on the server — the next run may use a different set than shown.')
    }
  }

  const jobPlan = resolveRestoredJob(store.currentJob, target)
  let jobUpdate: Partial<AppState> = {}
  if (jobPlan.kind === 'none') {
    jobUpdate = { currentJob: null, stlFile: null }
  } else if (jobPlan.kind === 'fetch') {
    let job: JobStatus | null = null
    try {
      const resp = await fetch(`/api/optimize/status/${jobPlan.jobId}`)
      if (resp.ok) job = await resp.json()
    } catch (_) {}
    jobUpdate = job?.status === 'completed' ? { currentJob: job, stlFile: job.job_id } : { currentJob: null, stlFile: null }
  }

  // A step without an uploaded image (e.g. "Started a new project") has no
  // image to show, whatever the snapshot's inputImage field says.
  const { url: restoredImageUrl, changed: imageChanged } = restoredImage(target, {
    inputImage: store.inputImage,
    inputFile: store.settings.input_image,
  })

  // Landing on a different image means the server's auto-preview belongs to
  // the wrong photo — and it is still marked "ready", so the status poll
  // would hand the 3D panel the previous image's heightmap. Dropped before
  // the store changes, because that is what starts the new init.
  //
  // Skipped when the destination already has its own completed job result:
  // that step doesn't use the auto-preview at all (useAutoPreviewInit bails
  // out whenever hasResult is true), so resetting it here only cost time —
  // and, worse, could *block* on releasing/rebuilding some *other* image's
  // auto-preview if one happened to be mid-build (the concurrent-init fix
  // makes /api/init/reset wait out an in-flight build before it returns),
  // stalling a plain history jump for as long as an unrelated heightmap
  // init took to finish. The stale auto-preview is harmless left alone: the
  // next *real* init request still releases it before building anew.
  const destinationHasResult = jobUpdate.currentJob?.status === 'completed'
  if (imageChanged && !destinationHasResult) await fetch('/api/init/reset', { method: 'POST' }).catch(() => {})

  UNDO_INDEX = index
  useAppStore.setState((prev) => ({
    historyIndex: index,
    activeFilaments: target.activeFilaments ?? prev.activeFilaments,
    colorSliders: target.colorSliders ?? prev.colorSliders,
    settings: target.settings ?? prev.settings,
    inputImage: restoredImageUrl,
    ...(target.sliderLayerRange ? { sliderLayerRange: target.sliderLayerRange } : {}),
    ...jobUpdate,
    // Everything below is *derived from the image*, so stepping to a
    // snapshot of a different one has to drop it. Keeping it is what made
    // History show one project's picture with another's layers: the live
    // preview PNG stayed on screen, and `initState: ready` kept the 3D panel
    // pointed at /api/init/mesh — which still held the previous image's
    // heightmap until a new init finished.
    ...(imageChanged
      ? {
          previewImage: null,
          initState: { status: 'idle' as const, preview_image: null },
          initSkipImage: null,
          hasRenderedInitPreview: false,
          resolvedBase: null,
          imageView: 'original' as const,
        }
      : {}),
    selectedBand: null,
    hoveredBand: null,
    // Nothing has been hand-edited *since arriving at this point in
    // history* — leaving the old value in place kept the "Replace your
    // color layers?" confirmation showing on the next Run even after
    // undoing away from the hand edit that had set it.
    slidersEditedByHand: false,
  }))
  useAppStore.getState().bumpPreviewVersion()
  if (jobUpdate.currentJob) useAppStore.getState().loadBaseColor(jobUpdate.currentJob.job_id)
  persistProjectState(useAppStore.getState())
}


