/** A border around the exported model, on the base layer in the base
 * filament (backend: Helper/BorderMesh). It is not an optimizer setting: the
 * 3D view and the export add it to the finished model, so it can be changed
 * at any time without a new run.
 *
 * Kept free of imports so node's test runner can load it directly. */

export type BorderShape = 'none' | 'rectangle' | 'oval' | 'circle' | 'outline'
export type BorderHanger = 'none' | 'top' | 'corners'

export interface BorderSettings {
  shape: BorderShape
  /** mm of base added around the picture. */
  width: number
  /** mm, rectangle only. */
  corner_radius: number
  hanger: BorderHanger
  hanger_diameter: number
}

export const DEFAULT_BORDER: BorderSettings = {
  shape: 'none',
  width: 5,
  corner_radius: 3,
  hanger: 'none',
  hanger_diameter: 4,
}

// Same ranges as the backend (BorderMesh.WIDTH_RANGE etc.).
export const BORDER_WIDTH_RANGE = { min: 1, max: 100 }
export const HANGER_DIAMETER_RANGE = { min: 2, max: 20 }
/** A rounded corner wider than this many border widths would leave less
 * than half the width around the picture's corner; the backend caps it. */
export const MAX_CORNER_RADIUS_PER_WIDTH = 2.2

export const BORDER_SHAPES: { value: BorderShape; label: string; hint: string }[] = [
  { value: 'none', label: 'None', hint: 'No border: the base ends at the edge of the picture' },
  { value: 'rectangle', label: 'Rectangle', hint: 'A frame around the picture, with optional rounded corners' },
  { value: 'oval', label: 'Oval', hint: 'An oval around the picture, in its proportions' },
  { value: 'circle', label: 'Circle', hint: 'A circle around the picture — tight around a round motif on a transparent background' },
  { value: 'outline', label: 'Outline', hint: 'Follows the outline of the picture — for motifs on a transparent background' },
]

export const BORDER_HANGERS: { value: BorderHanger; label: string; hint: string }[] = [
  { value: 'none', label: 'None', hint: 'No hanging hole' },
  { value: 'top', label: 'Top', hint: 'One hole above the picture, in the middle' },
  { value: 'corners', label: 'Two', hint: 'Two holes above the picture, towards its corners' },
]

const SHAPES = BORDER_SHAPES.map((s) => s.value)
const HANGERS = BORDER_HANGERS.map((s) => s.value)

function clampNumber(value: unknown, fallback: number, min: number, max: number): number {
  const n = typeof value === 'number' ? value : Number(value)
  if (!Number.isFinite(n)) return fallback
  return Math.min(max, Math.max(min, n))
}

/** A complete, in-range border from whatever a settings object, an older
 * project file or local storage holds (missing means no border). */
export function normalizeBorder(raw: unknown): BorderSettings {
  if (!raw || typeof raw !== 'object') return { ...DEFAULT_BORDER }
  const r = raw as Record<string, unknown>
  return {
    shape: SHAPES.includes(r.shape as BorderShape) ? (r.shape as BorderShape) : 'none',
    width: clampNumber(r.width, DEFAULT_BORDER.width, BORDER_WIDTH_RANGE.min, BORDER_WIDTH_RANGE.max),
    corner_radius: clampNumber(r.corner_radius, DEFAULT_BORDER.corner_radius, 0, 100),
    hanger: HANGERS.includes(r.hanger as BorderHanger) ? (r.hanger as BorderHanger) : 'none',
    hanger_diameter: clampNumber(r.hanger_diameter, DEFAULT_BORDER.hanger_diameter, HANGER_DIAMETER_RANGE.min, HANGER_DIAMETER_RANGE.max),
  }
}

export function borderActive(border: BorderSettings | null | undefined): boolean {
  return !!border && border.shape !== 'none'
}

/** The largest corner radius that takes effect at this width. */
export function maxCornerRadius(width: number): number {
  return Math.round(MAX_CORNER_RADIUS_PER_WIDTH * width * 10) / 10
}

function mm(value: number): string {
  return `${Math.round(value * 10) / 10} mm`
}

/** "Rectangle 5 mm, rounded 3 mm, hanging hole" — or null without a border. */
export function describeBorder(border: BorderSettings | null | undefined): string | null {
  if (!border || !borderActive(border)) return null
  const label = BORDER_SHAPES.find((s) => s.value === border.shape)?.label ?? border.shape
  const parts = [`${label} ${mm(border.width)}`]
  const radius = Math.min(border.corner_radius, maxCornerRadius(border.width))
  if (border.shape === 'rectangle' && radius > 0) parts.push(`rounded ${mm(radius)}`)
  if (border.hanger === 'top') parts.push('hanging hole')
  if (border.hanger === 'corners') parts.push('2 hanging holes')
  return parts.join(', ')
}

/** "160 × 312 mm" */
export function formatPrintSize(size: [number, number] | null | undefined): string | null {
  if (!size || !size.every(Number.isFinite)) return null
  return `${Math.round(size[0])} × ${Math.round(size[1])} mm`
}

/** What a border preview depends on; equal keys give the same mesh. */
export function borderPreviewKey(source: string, border: BorderSettings, backgroundHeight: number): string {
  return JSON.stringify([source, border, backgroundHeight])
}

export interface BorderMeshData {
  positions: Float32Array
  indices: Uint32Array
}

function base64Bytes(text: string): Uint8Array {
  const binary = atob(text)
  const bytes = new Uint8Array(binary.length)
  for (let i = 0; i < binary.length; i++) bytes[i] = binary.charCodeAt(i)
  return bytes
}

/** The border-preview response's mesh (base64 float32 xyz / uint32
 * triangles), or null when there is no border or the data doesn't add up. */
export function decodeBorderMesh(data: { positions?: string; indices?: string } | null | undefined): BorderMeshData | null {
  if (!data?.positions || !data?.indices) return null
  const p = base64Bytes(data.positions)
  const i = base64Bytes(data.indices)
  if (p.length === 0 || p.length % 12 !== 0 || i.length === 0 || i.length % 12 !== 0) return null
  const positions = new Float32Array(p.buffer, p.byteOffset, p.length / 4)
  const indices = new Uint32Array(i.buffer, i.byteOffset, i.length / 4)
  const vertexCount = positions.length / 3
  for (let k = 0; k < indices.length; k++) if (indices[k] >= vertexCount) return null
  return { positions, indices }
}
