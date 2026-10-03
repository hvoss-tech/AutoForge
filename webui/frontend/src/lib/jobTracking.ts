import type { JobStatus } from '../types'
import type { PruningCounts } from './pruning'

/** Whether a status report may replace the job currently on screen.
 *
 * Status reports arrive asynchronously (socket messages, polls, refreshes);
 * one that resolves after the UI moved on to another job used to replace
 * that job with the old one — and, for a completed old job, re-run its
 * "Optimization completed" side effects (snapshot, automatic prune). Only a
 * report for the job being shown (or one when nothing is shown) is applied;
 * switching to a different job is done explicitly, never by a report. */
export function acceptsStatusFor(current: Pick<JobStatus, 'job_id'> | null, incoming: Pick<JobStatus, 'job_id'> | null): boolean {
  if (!incoming || !current) return true
  return incoming.job_id === current.job_id
}

/** Whether a status update is the job's transition into "failed" — the
 * moment to tell the user, whichever channel (socket or poll) delivered it. */
export function isFailureTransition(previousStatus: string | undefined, job: Pick<JobStatus, 'status'> | null): boolean {
  return !!job && job.status === 'failed' && previousStatus !== 'failed'
}

/** The counts the automatic first prune keeps: the finished result's own
 * counts as the server reports them (result_colors / result_swaps /
 * result_layers) — the color sliders at that moment still show the last
 * training preview, from before the final base and limit searches changed
 * the stack. ``fallback`` (the slider plan's counts) only when the server
 * sent none. */
export function finishedResultCounts(
  job: Pick<JobStatus, 'result_colors' | 'result_swaps' | 'result_layers'>,
  fallback: PruningCounts,
): PruningCounts {
  const { result_colors: colors, result_swaps: swaps, result_layers: layers } = job
  if (typeof colors === 'number' && typeof swaps === 'number' && typeof layers === 'number') {
    return { colors, swaps, layers }
  }
  return fallback
}

export const PRUNING_POLL_MAX_NETWORK_ERRORS = 30

/** What the pruning poll does with one response: keep polling, or stop
 * because the job ended or can't be followed any more (the server forgot it
 * — a restart — or has been unreachable for a long stretch). */
export function pruningPollStep(
  response: { status: number; jobStatus?: string } | null,
  consecutiveNetworkErrors: number,
): 'continue' | 'done' | 'lost' {
  if (response === null) {
    return consecutiveNetworkErrors >= PRUNING_POLL_MAX_NETWORK_ERRORS ? 'lost' : 'continue'
  }
  if (response.status === 404) return 'lost'
  if (response.jobStatus && ['completed', 'failed', 'cancelled'].includes(response.jobStatus)) return 'done'
  return 'continue'
}

/** Whether a finished prune may become the current job: only while the
 * result it started from is still the one on screen. Undoing (or jumping in
 * History) during a prune used to be overridden when the prune finished. */
export function adoptsPrunedResult(currentJobId: string | undefined | null, sourceJobId: string): boolean {
  return currentJobId === sourceJobId
}
