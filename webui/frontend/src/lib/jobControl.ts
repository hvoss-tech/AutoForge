import type { JobStatus } from '../types'

const JOB_STATUSES: ReadonlyArray<JobStatus['status']> = ['pending', 'running', 'completed', 'failed', 'cancelled', 'paused']

// Pause/resume/cancel respond with the job's real status. A job that already
// finished (e.g. Cancel clicked just as it completed) is left untouched by the
// backend, so assuming the requested status showed a completed result as
// "cancelled", or a finished job as "paused" with a Resume button forever.
export function jobStatusFromControlResponse(body: unknown, fallback: JobStatus['status']): JobStatus['status'] {
  const status = (body as { status?: unknown } | null)?.status
  return JOB_STATUSES.includes(status as JobStatus['status']) ? (status as JobStatus['status']) : fallback
}
