/** Color / swap limits the optimizer keeps to while it runs
 * (settings.max_colors / settings.max_swaps; null = unlimited).
 *
 * Colors count the base filament too, exactly like the pruning dialog's
 * "Max colors" — so the smallest useful limit is 2 (the base plus one color).
 * Kept import-free so the node unit tests can load it directly. */

export type RunLimitKey = 'max_colors' | 'max_swaps'

export interface RunLimits {
  max_colors?: number | null
  max_swaps?: number | null
}

export const RUN_LIMIT_MIN: Record<RunLimitKey, number> = { max_colors: 2, max_swaps: 0 }

/** What "At most" starts at when a limit is switched on. */
export const RUN_LIMIT_DEFAULT: Record<RunLimitKey, number> = { max_colors: 4, max_swaps: 20 }

export function isLimited(value: number | null | undefined): value is number {
  return typeof value === 'number' && Number.isFinite(value)
}

/** A limit value as the settings store it: a whole number at or above the
 * minimum, or null for unlimited. */
export function normalizeLimit(key: RunLimitKey, value: number | null | undefined): number | null {
  if (!isLimited(value)) return null
  return Math.max(RUN_LIMIT_MIN[key], Math.round(value))
}

/** "at most 4 colors · 20 swaps", or null when nothing is limited. */
export function describeRunLimits(limits: RunLimits): string | null {
  const parts: string[] = []
  if (isLimited(limits.max_colors)) parts.push(`${limits.max_colors} colors`)
  if (isLimited(limits.max_swaps)) parts.push(`${limits.max_swaps} ${limits.max_swaps === 1 ? 'swap' : 'swaps'}`)
  return parts.length ? `at most ${parts.join(' · ')}` : null
}

/** A note when a color limit can't bite: with N filaments active the result
 * never has more than N colors (the base is one of them). */
export function colorLimitNote(maxColors: number | null | undefined, activeFilaments: number): string | null {
  if (!isLimited(maxColors) || activeFilaments <= 0) return null
  if (maxColors >= activeFilaments + 1) {
    return `You have ${activeFilaments} filament${activeFilaments === 1 ? '' : 's'} active, so this limit has no effect.`
  }
  return null
}

/** Does a result's counts (colors including the base, swaps) keep to the limits? */
export function withinRunLimits(limits: RunLimits, counts: { colors: number; swaps: number }): boolean {
  if (isLimited(limits.max_colors) && counts.colors > limits.max_colors) return false
  if (isLimited(limits.max_swaps) && counts.swaps > limits.max_swaps) return false
  return true
}
