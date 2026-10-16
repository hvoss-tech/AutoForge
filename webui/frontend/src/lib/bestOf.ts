/** "Best of" runs (settings.best_of): the optimization runs that many times
 * and the result with the lowest loss is kept, like the CLI's --best_of. */

export const BEST_OF_MAX = 20

/** The setting as a whole number of runs, at least one. */
export function bestOfRuns(value: unknown): number {
  const n = Math.floor(Number(value))
  return Number.isFinite(n) && n > 1 ? Math.min(n, BEST_OF_MAX) : 1
}

/** The warning to confirm before a best-of run starts, or null for a
 * single run. */
export function bestOfWarning(value: unknown): { title: string; message: string; confirmLabel: string } | null {
  const runs = bestOfRuns(value)
  if (runs === 1) return null
  return {
    title: `Run the optimization ${runs} times?`,
    message:
      `"Best of" is set to ${runs}: the whole optimization runs ${runs} times and the result with the lowest loss is kept. ` +
      `This takes about ${runs}× as long as a single run.\n` +
      'Cancelling stops all runs, and keeps none of them.',
    confirmLabel: `Run ${runs} times`,
  }
}

/** The runs of a best-of job for the UI: one line per finished run, the
 * kept one marked. Null for a job of a single run. */
export function describeBestOfRuns(losses: number[] | null | undefined, kept?: number | null): string[] | null {
  if (!losses || losses.length === 0) return null
  return losses.map((loss, i) => `Run ${i + 1}: ${loss.toFixed(4)}${i === kept ? ' (kept)' : ''}`)
}
