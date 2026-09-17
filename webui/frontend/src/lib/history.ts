import type { JobStatus, Snapshot } from '../types'

export const ACTIVE_JOB_STATUSES: ReadonlyArray<string> = ['pending', 'running', 'paused']

/** The image a history snapshot belongs to.
 *
 * `settings.input_image` is the authority — it is the file the server would
 * actually run, and it is what everything else in the snapshot (sliders,
 * result, layer range) was captured against. The snapshot's own `inputImage`
 * field is only the URL used to display it, and it can be missing (older
 * snapshots) or unusable (a `blob:` URL from a tab that has since reloaded).
 *
 * Falling back to "whatever is on screen right now", as this used to, is the
 * one thing it must never do: stepping to a snapshot of a *different* image
 * then left the previous picture in the panel beside the restored image's
 * color layers — one photo shown for two different projects. */
export function snapshotInputImageUrl(target: Snapshot, fallback: string | null = null): string | null {
  const filename = target.settings?.input_image
  if (!filename) return target.settings ? null : (target.inputImage ?? fallback)
  const stored = target.inputImage
  if (stored && !stored.startsWith('blob:') && stored.includes(filename)) return stored
  return `/uploads/${filename}`
}

/** The image to show after jumping to a snapshot, and whether it's a
 * different photo than the one on screen.
 *
 * "Different" has to mean *a different file*, not a different URL string.
 * A freshly uploaded image is displayed from a `blob:` URL while the
 * snapshot records the durable `/uploads/<name>` path for the same file, so
 * comparing URLs called every undo right after an upload an image change —
 * which threw away the live preview and re-ran the (slow, GPU) heightmap
 * init for a photo that hadn't changed at all. Same file: keep the URL that
 * is already loaded, and report no change. */
export function restoredImage(
  target: Snapshot,
  current: { inputImage: string | null; inputFile: string | undefined },
): { url: string | null; changed: boolean } {
  const targetFile = target.settings ? target.settings.input_image || '' : null
  if (targetFile !== null && targetFile === (current.inputFile || '') && current.inputImage) {
    return { url: current.inputImage, changed: false }
  }
  const url = snapshotInputImageUrl(target, current.inputImage)
  return { url, changed: url !== current.inputImage }
}

/** Whether a /ws/preview broadcast is meant for this tab.
 *
 * The channel is shared by every connected tab and carries no per-connection
 * subscription. Accepting everything whenever the tab had no job of its own
 * is what let a pruning run's final broadcast (sent under the *optimization*
 * job's id) rewrite the color layers of a history step the user had just
 * returned to, and what let one image's colors land on another's.
 *
 * A tab that has never tracked a job is the one case where an unknown job is
 * worth taking: that's how a run started from the API, or in another tab,
 * gets noticed at all. */
export function acceptsPreviewUpdate(
  broadcastJobId: string | undefined,
  currentJobId: string | undefined,
  hasTrackedAnyJob: boolean,
): boolean {
  if (currentJobId) return broadcastJobId === currentJobId
  return !hasTrackedAnyJob
}

export type RestoredJob =
  | { kind: 'keep' }
  | { kind: 'none' }
  | { kind: 'fetch'; jobId: string }

/** Which job the app should show after jumping to a history snapshot.
 *
 * - A job that's still running is never touched: undoing a slider tweak used
 *   to cancel a long optimization outright.
 * - Otherwise the snapshot's *completed result* (optimizationResultId) wins.
 *   The old code used currentJobId, so a snapshot taken while a job was
 *   still running restored that job — by then completed — with the pre-result
 *   slider stack, re-rendering the result with the wrong sliders.
 * - No result at snapshot time means none now (back to the pre-run preview). */
export function resolveRestoredJob(currentJob: JobStatus | null, target: Snapshot): RestoredJob {
  if (currentJob && ACTIVE_JOB_STATUSES.includes(currentJob.status)) return { kind: 'keep' }
  const resultId = target.optimizationResultId ?? null
  if (!resultId) return { kind: 'none' }
  if (currentJob?.job_id === resultId && currentJob.status === 'completed') return { kind: 'keep' }
  return { kind: 'fetch', jobId: resultId }
}

interface ShortcutEvent {
  key: string
  ctrlKey: boolean
  metaKey: boolean
  shiftKey: boolean
  altKey?: boolean
  target?: unknown
}

function isTextEditingTarget(target: unknown): boolean {
  const el = target as { tagName?: string; isContentEditable?: boolean; type?: string } | null
  if (!el || typeof el.tagName !== 'string') return false
  if (el.isContentEditable) return true
  const tag = el.tagName.toUpperCase()
  if (tag === 'TEXTAREA' || tag === 'SELECT') return true
  if (tag !== 'INPUT') return false
  // Sliders/checkboxes/color pickers have no native text undo to protect.
  return !['range', 'checkbox', 'radio', 'button', 'color', 'file'].includes((el.type ?? 'text').toLowerCase())
}

/** Global undo/redo shortcut for a keydown event, or null.
 *
 * `e.key` is 'Z' (upper case) while Shift is held, so the old
 * `e.shiftKey && e.key === 'z'` check meant Ctrl+Shift+Z never redid
 * anything. Also accepts Ctrl+Y and Cmd on macOS, and leaves the shortcut to
 * the browser while the user is typing in a text field. */
export function historyShortcut(e: ShortcutEvent): 'undo' | 'redo' | null {
  if (!(e.ctrlKey || e.metaKey) || e.altKey) return null
  if (isTextEditingTarget(e.target)) return null
  const key = e.key.toLowerCase()
  if (key === 'z') return e.shiftKey ? 'redo' : 'undo'
  if (key === 'y' && !e.shiftKey) return 'redo'
  return null
}

/** Two edits that land in the same debounce window become one undo step;
 * keep its label informative instead of just "the last one wins". Adding
 * four filaments in a row used to show up as a single "Added filament". */
export function mergeHistoryLabels(previous: string | undefined, next: string): string {
  if (!previous || previous === next) return next
  for (const verb of ['Added', 'Removed']) {
    const prefix = `${verb} `
    if (previous.startsWith(prefix) && next.startsWith(prefix)) {
      const names = [...previous.slice(prefix.length).split(', '), next.slice(prefix.length)]
      return prefix + [...new Set(names)].join(', ')
    }
  }
  return next
}

/** "Moved band 2 to layer 14" instead of the old catch-all "Color slider edit". */
export function describeSliderEdit(
  index: number,
  updates: { td?: number; layer?: number; enabled?: boolean; filament_uuid?: string },
  filamentName?: string,
): string {
  const band = `band ${index + 1}`
  if (updates.filament_uuid !== undefined) return `Assigned ${filamentName ?? 'filament'} to ${band}`
  if (updates.enabled !== undefined) return `${updates.enabled ? 'Enabled' : 'Disabled'} ${band}`
  if (updates.layer !== undefined) return `Moved ${band} to layer ${updates.layer}`
  if (updates.td !== undefined) return `Changed TD of ${band} to ${updates.td}`
  return `Edited ${band}`
}

const SETTING_NAMES: Record<string, string> = {
  input_image: 'input image',
  stl_output_size: 'dimension',
  background_height: 'base height',
  learning_rate_warmup_fraction: 'LR warmup fraction',
  init_tau: 'initial temperature',
  final_tau: 'final temperature',
  discrete_check: 'result check interval',
  num_init_rounds: 'init rounds',
  num_init_cluster_layers: 'cluster layers',
  init_heightmap_method: 'heightmap method',
  processing_reduction_factor: 'processing reduction',
}

/** Plain-language name of a settings key ("layer height"). */
export function settingName(key: string): string {
  return SETTING_NAMES[key] ?? key.replace(/_/g, ' ')
}

export function describeSettingsChange(before: Record<string, unknown>, after: Record<string, unknown>): string {
  const changed = Object.keys(after).filter((k) => before[k] !== after[k])
  if (changed.length !== 1) return 'Changed settings'
  return `Changed ${settingName(changed[0])}`
}
