/** What a run was started with, to tell when the result on screen no longer
 * matches the current settings. */
export interface RunInputs {
  settings: Record<string, unknown>
  filamentUuids: string[]
}

// Settings that don't change what the optimizer produces.
const IGNORED = new Set([
  'pruning_max_colors',
  'pruning_max_swaps',
  'pruning_max_layer',
  'visualize',
  'disable_visualization_for_gradio',
  'tensorboard',
  'run_name',
  'output_folder',
])

/** Settings keys that changed since the run, plus 'active_filaments' when
 * the filament set did (empty when nothing did, or when it's unknown what the
 * run used). Show them with history.ts's settingName(). */
export function staleReasons(
  run: RunInputs | null | undefined,
  settings: Record<string, unknown>,
  activeFilamentUuids: string[],
): string[] {
  if (!run) return []
  // The focus-area strength only matters when there are focus areas, in
  // the run or now.
  const noMask = !run.settings.priority_mask && !settings.priority_mask
  const reasons = Object.keys(settings)
    .filter((key) => !IGNORED.has(key) && key in run.settings && run.settings[key] !== settings[key])
    .filter((key) => !(noMask && key === 'priority_mask_strength'))
  const before = [...run.filamentUuids].sort().join(',')
  const after = [...activeFilamentUuids].sort().join(',')
  if (before !== after) reasons.push('active_filaments')
  return reasons
}

const STORAGE_KEY = 'autoforge-run-inputs'
const KEEP = 20

export function readStoredRunInputs(): Record<string, RunInputs> {
  try {
    const parsed = JSON.parse(localStorage.getItem(STORAGE_KEY) ?? '{}')
    return parsed && typeof parsed === 'object' ? parsed : {}
  } catch {
    return {}
  }
}

export function storeRunInputs(all: Record<string, RunInputs>, jobId: string, inputs: RunInputs): Record<string, RunInputs> {
  const next = { ...all, [jobId]: inputs }
  const ids = Object.keys(next)
  for (const id of ids.slice(0, Math.max(0, ids.length - KEEP))) delete next[id]
  try {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(next))
  } catch {
    // not remembered — the result just can't be flagged as out of date later
  }
  return next
}

/** A pruned result is built from its source run's settings and filaments, so
 * it inherits that run's inputs. Without an entry of its own, staleReasons()
 * got `undefined` for it and a pruned result could never be "Out of date". */
export function inheritRunInputs(all: Record<string, RunInputs>, fromJobId: string, toJobId: string): Record<string, RunInputs> {
  const source = all[fromJobId]
  if (!source || fromJobId === toJobId) return all
  return storeRunInputs(all, toJobId, source)
}
