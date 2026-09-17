import { parseColoredMesh, type ParsedMesh } from './plyParser'

/** Parse a colored PLY off the main thread.
 *
 * Parsing walks every vertex and every face one property at a time; on a
 * preview mesh that is hundreds of thousands of iterations, and on the main
 * thread it froze the whole page — no scrolling, no slider, no spinner —
 * for the duration, on every edit that re-rendered the model. A worker turns
 * that into background work, so the UI stays live and the old mesh keeps
 * rendering until the new one is ready.
 *
 * The parsed arrays come back transferred (not copied), so handing a mesh
 * over costs nothing on either side.
 */

let worker: Worker | null = null
let nextId = 1
const pending = new Map<number, { resolve: (mesh: ParsedMesh) => void; reject: (err: Error) => void }>()

function ensureWorker(): Worker | null {
  if (worker) return worker
  // No Worker (tests under jsdom/node, or a locked-down embedding): the
  // caller falls back to parsing inline, which is correct, just blocking.
  if (typeof Worker === 'undefined') return null
  try {
    worker = new Worker(new URL('./plyParser.worker.ts', import.meta.url), { type: 'module' })
  } catch {
    return null
  }
  worker.onmessage = (event: MessageEvent) => {
    const { id, positions, colors, indices, positionsHash, error } = event.data ?? {}
    const entry = pending.get(id)
    if (!entry) return
    pending.delete(id)
    if (error) entry.reject(new Error(error))
    else entry.resolve({ positions, colors, indices, positionsHash })
  }
  worker.onerror = () => {
    // One broken worker must not leave every caller hanging for ever.
    for (const entry of pending.values()) entry.reject(new Error('PLY worker failed'))
    pending.clear()
    worker?.terminate()
    worker = null
  }
  return worker
}

export function parseColoredMeshAsync(buffer: ArrayBuffer): Promise<ParsedMesh> {
  const w = ensureWorker()
  if (!w) {
    try {
      return Promise.resolve(parseColoredMesh(buffer))
    } catch (err) {
      return Promise.reject(err)
    }
  }
  const id = nextId++
  return new Promise<ParsedMesh>((resolve, reject) => {
    pending.set(id, { resolve, reject })
    // `buffer` is transferred, so nothing here may read it afterwards.
    w.postMessage({ id, buffer }, [buffer])
  })
}
