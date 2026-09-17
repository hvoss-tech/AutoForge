import type { JobStatus, Snapshot } from '../types'

export const ACTIVE_JOB_STATUSES: ReadonlyArray<string> = ['pending', 'running', 'paused']

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
