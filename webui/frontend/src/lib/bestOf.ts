/** "Best of" runs (settings.best_of): the optimization runs that many times,
 * then the user looks at every run in a gallery and picks the one to keep. */

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
      `"Best of" is set to ${runs}: the whole optimization runs ${runs} times, then you compare the results side by side and pick the one to keep. ` +
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

/** Enough of a job to tell whether it waits for a run to be picked. */
interface BestOfJob {
  status: string
  best_of_choosing?: boolean | null
  best_of_losses?: number[] | null
}

/** True while every run of the series is done and the job waits for the
 * user to pick one (the server keeps the flag on a job cancelled meanwhile). */
export function isChoosingBestOf(job: BestOfJob | null | undefined): boolean {
  return !!job && !!job.best_of_choosing && (job.status === 'running' || job.status === 'paused') && (job.best_of_losses?.length ?? 0) > 1
}

/** Columns and rows of the gallery: square-ish, and a single row for up to
 * three runs (they're compared side by side). */
export function galleryGrid(count: number): { columns: number; rows: number } {
  const n = Math.max(1, Math.floor(count))
  const columns = n <= 3 ? n : Math.ceil(Math.sqrt(n))
  return { columns, rows: Math.ceil(n / columns) }
}

/** The lowest-loss run's index (the first of equals), or -1 without runs. */
export function lowestLossRun(losses: number[]): number {
  let best = -1
  losses.forEach((loss, i) => {
    if (best < 0 || loss < losses[best]) best = i
  })
  return best
}

/** How much higher a run's loss is than the lowest, for its card: "Lowest
 * loss", or e.g. "+3.2%". */
export function describeLossGap(losses: number[], run: number): string {
  const best = lowestLossRun(losses)
  if (run === best || losses[best] === losses[run]) return 'Lowest loss'
  const base = losses[best]
  if (!(base > 0)) return `+${(losses[run] - base).toFixed(4)}`
  const pct = ((losses[run] - base) / base) * 100
  return `+${pct < 0.1 ? pct.toFixed(2) : pct.toFixed(1)}%`
}

/** The two runs marked for an A/B comparison after clicking `run`: clicking
 * a marked run unmarks it, a third run replaces the one marked first. */
export function toggleCompareRun(marked: number[], run: number): number[] {
  if (marked.includes(run)) return marked.filter((r) => r !== run)
  return [...marked, run].slice(-2)
}

/** The image of a run of the series still waiting for a choice. */
export function bestOfImageUrl(jobId: string, run: number): string {
  return `/api/optimize/best-of/${encodeURIComponent(jobId)}/image/${run}`
}

/** The run shown after stepping `delta` from `run`, wrapping around. */
export function stepRun(run: number, delta: number, count: number): number {
  if (count <= 0) return 0
  return (((run + delta) % count) + count) % count
}
