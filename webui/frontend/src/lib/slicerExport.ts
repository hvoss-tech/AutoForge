/** Runs a slicer project export with progress: the server builds the file
 * in the background (POST /api/outputs/3mf/{job}/start, polled at
 * /tasks/{id}), then the browser fetches it, counting the bytes. */

import { describeApiError } from './apiError'

export interface ExportProgress {
  /** 0..1 over the whole export, download included. */
  fraction: number
  stage: string
}

/** The server's share of the bar; the download takes the rest. */
const BUILD_SHARE = 0.9
const POLL_MS = 200

interface TaskStatus {
  task_id: string
  progress: number
  stage: string
  done: boolean
  error: string | null
  cancelled: boolean
  printer?: string | null
  notes?: string[]
  warnings?: string[]
}

export interface SlicerExportResult {
  blob: Blob
  /** The printer preset the project was made for (the one for the result's nozzle). */
  printer: string | null
  notes: string[]
  /** Things the user has to fix in the slicer, e.g. a nozzle without a preset. */
  warnings: string[]
}

const sleep = (ms: number) => new Promise((resolve) => setTimeout(resolve, ms))

function abortError(): Error {
  const e = new Error('Export cancelled')
  e.name = 'AbortError'
  return e
}

async function errorOf(response: Response): Promise<Error> {
  return new Error(describeApiError(await response.json().catch(() => null), response.status))
}

export async function runSlicerExport(
  jobId: string,
  body: Record<string, unknown>,
  onProgress: (p: ExportProgress) => void,
  signal?: AbortSignal,
  fetchImpl: typeof fetch = fetch,
): Promise<SlicerExportResult> {
  onProgress({ fraction: 0, stage: 'Starting' })
  const started = await fetchImpl(`/api/outputs/3mf/${jobId}/start`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  })
  if (!started.ok) throw await errorOf(started)
  const taskId = ((await started.json()) as TaskStatus).task_id

  let finished: TaskStatus | null = null
  for (;;) {
    if (signal?.aborted) {
      await fetchImpl(`/api/outputs/tasks/${taskId}`, { method: 'DELETE' }).catch(() => undefined)
      throw abortError()
    }
    const r = await fetchImpl(`/api/outputs/tasks/${taskId}`)
    if (!r.ok) throw await errorOf(r)
    const status = (await r.json()) as TaskStatus
    if (status.error) throw new Error(status.error)
    if (status.cancelled) throw abortError()
    onProgress({ fraction: BUILD_SHARE * status.progress, stage: status.stage })
    if (status.done) {
      finished = status
      break
    }
    await sleep(POLL_MS)
  }

  onProgress({ fraction: BUILD_SHARE, stage: 'Downloading' })
  const file = await fetchImpl(`/api/outputs/tasks/${taskId}/file`, { signal })
  if (!file.ok) throw await errorOf(file)
  const info = { printer: finished?.printer ?? null, notes: finished?.notes ?? [], warnings: finished?.warnings ?? [] }
  const total = Number(file.headers.get('content-length')) || 0
  if (!file.body || !total) {
    const blob = await file.blob()
    onProgress({ fraction: 1, stage: 'Done' })
    return { blob, ...info }
  }
  const reader = file.body.getReader()
  const chunks: Uint8Array[] = []
  let received = 0
  for (;;) {
    const { done, value } = await reader.read()
    if (done) break
    chunks.push(value)
    received += value.length
    onProgress({ fraction: BUILD_SHARE + (1 - BUILD_SHARE) * Math.min(1, received / total), stage: 'Downloading' })
  }
  onProgress({ fraction: 1, stage: 'Done' })
  return { blob: new Blob(chunks as BlobPart[], { type: 'application/vnd.ms-package.3dmanufacturing-3dmodel+xml' }), ...info }
}

/** Ask the server to load a printer's profiles now (downloading them the
 * first time), while the user is still choosing. Fire and forget. */
const prepared = new Set<string>()
export function preparePrinter(name: string, nozzle?: number | null, fetchImpl: typeof fetch = fetch): void {
  const key = `${name}|${nozzle ?? ''}`
  if (!name || prepared.has(key)) return
  prepared.add(key)
  fetchImpl('/api/outputs/3mf/printers/prepare', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ printer: name, nozzle: nozzle ?? null }),
  }).catch(() => prepared.delete(key))
}
