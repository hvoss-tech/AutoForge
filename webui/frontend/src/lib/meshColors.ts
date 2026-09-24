/** Live recoloring of the 3D mesh on a slider edit.
 *
 * A slider edit changes which filament sits on which layer — never a height —
 * so the only thing about the mesh that changes is its vertex colors. The
 * render endpoint returns them (`vertex_colors`: RGB bytes for the mesh's
 * top vertices, in vertex order; the bottom vertices never change), and the
 * 3D view writes them straight into the geometry it already has. That
 * replaces rebuilding, exporting, downloading and parsing a ~25MB PLY per
 * edit. The PLY on disk catches up in the background for reloads.
 *
 * Kept free of imports so node's test runner can load it directly.
 */

// Same sRGB -> linear conversion plyParser applies to PLY colors: the lit
// mesh must look identical whether its colors came from a PLY or from here.
const SRGB_TO_LINEAR_LUT = new Float32Array(256)
for (let b = 0; b < 256; b++) {
  const c = b / 255
  SRGB_TO_LINEAR_LUT[b] = c <= 0.04045 ? c / 12.92 : Math.pow((c + 0.055) / 1.055, 2.4)
}

// Mirrors api/preview.py's INIT_JOB_SENTINEL.
const INIT_JOB_SENTINEL = '__init__'

/** The mesh URL (without its `?v=` suffix) the 3D view loads for a render
 * target — also how an update finds the mesh it belongs to. */
export function meshKeyForJob(jobId: string): string {
  return jobId === INIT_JOB_SENTINEL ? '/api/init/mesh' : `/api/outputs/colored-ply/${jobId}`
}

export function decodeBase64(b64: string): Uint8Array {
  const binary = atob(b64)
  const out = new Uint8Array(binary.length)
  for (let i = 0; i < binary.length; i++) out[i] = binary.charCodeAt(i)
  return out
}

/** Writes `rgb` (top-vertex colors, 0-255) into a mesh color buffer whose
 * first half are the top vertices. Returns false — leaving `target` alone —
 * when the buffer isn't the mesh these colors were computed for. */
export function writeTopVertexColors(target: Float32Array, rgb: Uint8Array): boolean {
  if (rgb.length === 0 || rgb.length % 3 !== 0 || target.length !== rgb.length * 2) return false
  for (let i = 0; i < rgb.length; i++) target[i] = SRGB_TO_LINEAR_LUT[rgb[i]]
  return true
}

export interface VertexColorUpdate {
  meshKey: string
  rgb: Uint8Array
  /** Monotonic, so a mesh that finishes loading later can tell whether this
   * update is newer than the file it loaded. */
  seq: number
}

type Listener = (update: VertexColorUpdate) => boolean
const listeners = new Set<Listener>()
const latestByKey = new Map<string, VertexColorUpdate>()
let nextSeq = 1

export function currentColorSeq(): number {
  return nextSeq
}

export function subscribeVertexColors(listener: Listener): () => void {
  listeners.add(listener)
  return () => {
    listeners.delete(listener)
  }
}

/** Hands new colors to whichever mesh view shows `meshKey`. Returns whether
 * one applied them; if not (no mesh loaded yet, or a different one), the
 * caller falls back to reloading the mesh file. */
export function publishVertexColors(meshKey: string, rgb: Uint8Array): boolean {
  const update = { meshKey, rgb, seq: nextSeq++ }
  latestByKey.set(meshKey, update)
  let applied = false
  for (const listener of listeners) applied = listener(update) || applied
  return applied
}

/** The newest colors published for `meshKey` since sequence `sinceSeq` —
 * a mesh file whose download started before them is older than they are. */
export function colorsNewerThan(meshKey: string, sinceSeq: number): VertexColorUpdate | null {
  const latest = latestByKey.get(meshKey)
  return latest && latest.seq >= sinceSeq ? latest : null
}

// Render requests this tab sent. Their /ws/preview broadcast must not make
// this tab reload the mesh it has just recolored.
const ownRenderIds = new Set<string>()
let renderCounter = 0

export function newRenderId(): string {
  renderCounter += 1
  const id = `${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 8)}-${renderCounter}`
  ownRenderIds.add(id)
  if (ownRenderIds.size > 200) ownRenderIds.delete(ownRenderIds.values().next().value as string)
  return id
}

export function isOwnRender(renderId: unknown): boolean {
  return typeof renderId === 'string' && ownRenderIds.has(renderId)
}
