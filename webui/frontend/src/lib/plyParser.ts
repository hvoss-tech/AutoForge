// The composited preview image (and therefore the colored PLY's vertex
// colors) is encoded in sRGB, like any normal 8-bit image — that's also
// what the 2D preview <img> displays, since browsers treat image pixels as
// sRGB automatically. Three.js's PBR lighting (meshStandardMaterial) does
// its math in linear color space, so feeding it raw sRGB bytes as if they
// were already linear makes the lit 3D mesh look measurably different from
// the flat 2D preview (typically darker/less saturated). Converting each
// byte through the sRGB transfer function up front — a 256-entry lookup
// table, since the input is always a uint8 — keeps the two in visual
// agreement without paying a per-vertex Math.pow cost.
const SRGB_TO_LINEAR_LUT = new Float32Array(256)
for (let b = 0; b < 256; b++) {
  const c = b / 255
  SRGB_TO_LINEAR_LUT[b] = c <= 0.04045 ? c / 12.92 : Math.pow((c + 0.055) / 1.055, 2.4)
}

const PLY_TYPE_SIZES: Record<string, number> = {
  float: 4,
  double: 8,
  uchar: 1,
  char: 1,
  ushort: 2,
  short: 2,
  int: 4,
  uint: 4,
}

export type ParsedHeader = {
  headerLen: number
  vertexCount: number
  faceCount: number
  vertexStride: number
  colorStride: number
}

export function parsePLYHeader(buffer: ArrayBuffer): ParsedHeader {
  const decoder = new TextDecoder('ascii')
  const header = decoder.decode(new Uint8Array(buffer, 0, Math.min(buffer.byteLength, 4096)))
  const end = header.indexOf('end_header')
  if (end === -1) throw new Error('Invalid PLY: no end_header')
  const headerLines = header.substring(0, end).split('\n').filter(l => l.trim())

  let vertexCount = 0
  let faceCount = 0
  let vertexStride = 0
  let colorStride = 0
  let hasColors = false

  for (const line of headerLines) {
    const parts = line.trim().split(/\s+/)
    if (parts[0] === 'element') {
      if (parts[1] === 'vertex') vertexCount = parseInt(parts[2])
      if (parts[1] === 'face') faceCount = parseInt(parts[2])
    } else if (parts[0] === 'property') {
      const typeName = parts[1] as keyof typeof PLY_TYPE_SIZES
      const byteSize = PLY_TYPE_SIZES[typeName]
      if (byteSize === undefined) continue
      vertexStride += byteSize
      const name = parts[2].toLowerCase()
      if (name === 'red' || name === 'green' || name === 'blue' || name === 'r' || name === 'g' || name === 'b') {
        colorStride += byteSize
        hasColors = true
      }
    }
  }

  const headerLen = end + 'end_header'.length + 1
  return { headerLen, vertexCount, faceCount, vertexStride, colorStride }
}

export interface ParsedMesh {
  positions: Float32Array
  colors: Float32Array | null
  indices: Uint32Array
  /** Cheap identity of the geometry (see `hashFloats`). */
  positionsHash: number
}

/** FNV-1a over the raw bits of a Float32Array.
 *
 * Used to tell "the same mesh, recolored" from "a different mesh": the 3D
 * view reuses the geometry already on the GPU in the first case and only
 * re-uploads colors, which is what keeps slider edits instant. Vertex count
 * alone isn't enough to tell them apart — pruning rewrites every height on
 * an unchanged grid, so the count (and the byte length) match while the
 * shape is completely different. Over a few hundred thousand values this
 * costs well under a millisecond, and it runs in the parser's worker. */
export function hashFloats(values: Float32Array): number {
  const words = new Uint32Array(values.buffer, values.byteOffset, values.length)
  let hash = 0x811c9dc5
  for (let i = 0; i < words.length; i++) {
    hash ^= words[i]
    hash = Math.imul(hash, 0x01000193)
  }
  return hash >>> 0
}

export function parseColoredMesh(buffer: ArrayBuffer): ParsedMesh {
  const dv = new DataView(buffer)
  const { headerLen, vertexCount, faceCount, vertexStride, colorStride } = parsePLYHeader(buffer)
  if (vertexCount === 0 || faceCount === 0) {
    throw new Error('Empty PLY mesh')
  }

  let offset = headerLen
  const positions = new Float32Array(vertexCount * 3)
  const hasColors = colorStride > 0
  const colors = hasColors ? new Float32Array(vertexCount * 3) : null

  const positionBytes = 12
  const colorReadBytes = Math.min(colorStride, 3)
  const skipBytes = Math.max(0, vertexStride - positionBytes - colorReadBytes)

  for (let i = 0; i < vertexCount; i++) {
    positions[i * 3] = dv.getFloat32(offset, true); offset += 4
    positions[i * 3 + 1] = dv.getFloat32(offset, true); offset += 4
    positions[i * 3 + 2] = dv.getFloat32(offset, true); offset += 4
    if (hasColors && colors) {
      colors[i * 3] = SRGB_TO_LINEAR_LUT[dv.getUint8(offset)]; offset += 1
      colors[i * 3 + 1] = SRGB_TO_LINEAR_LUT[dv.getUint8(offset)]; offset += 1
      colors[i * 3 + 2] = SRGB_TO_LINEAR_LUT[dv.getUint8(offset)]; offset += 1
      offset += skipBytes
    }
  }

  const indices = new Uint32Array(faceCount * 3)
  for (let i = 0; i < faceCount; i++) {
    const count = dv.getUint8(offset); offset += 1
    if (count === 3) {
      indices[i * 3] = dv.getUint32(offset, true); offset += 4
      indices[i * 3 + 1] = dv.getUint32(offset, true); offset += 4
      indices[i * 3 + 2] = dv.getUint32(offset, true); offset += 4
    } else {
      offset += count * 4
    }
  }

  return { positions, colors, indices, positionsHash: hashFloats(positions) }
}