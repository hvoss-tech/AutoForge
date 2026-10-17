/** Selections for the image editor: one byte per pixel (0 = not selected,
 * 255 = selected), made with a lasso, rectangle, ellipse, brush or magic
 * wand and combined with what was selected before. */

import { clamp, gaussianBlur } from './imageMath.ts'

export type SelectMode = 'replace' | 'add' | 'subtract'
export type Point = [number, number]

/** `shape` combined into `target`, in place. */
export function combine(target: Uint8Array, shape: Uint8Array, mode: SelectMode): void {
  if (mode === 'replace') {
    target.set(shape)
  } else if (mode === 'add') {
    for (let i = 0; i < target.length; i++) if (shape[i] > target[i]) target[i] = shape[i]
  } else {
    for (let i = 0; i < target.length; i++) {
      const keep = 255 - shape[i]
      if (keep < target[i]) target[i] = keep
    }
  }
}

/** A closed polygon (lasso, rectangle, ellipse) filled even-odd, sampled at
 * pixel centres. */
export function polygonMask(w: number, h: number, points: Point[]): Uint8Array {
  const mask = new Uint8Array(w * h)
  if (points.length < 3) return mask
  const xs: number[] = []
  for (let y = 0; y < h; y++) {
    const cy = y + 0.5
    xs.length = 0
    for (let i = 0, j = points.length - 1; i < points.length; j = i++) {
      const [x0, y0] = points[j]
      const [x1, y1] = points[i]
      if ((y0 <= cy && y1 > cy) || (y1 <= cy && y0 > cy)) xs.push(x0 + ((cy - y0) / (y1 - y0)) * (x1 - x0))
    }
    xs.sort((a, b) => a - b)
    for (let k = 0; k + 1 < xs.length; k += 2) {
      const from = Math.max(0, Math.ceil(xs[k] - 0.5))
      const to = Math.min(w - 1, Math.floor(xs[k + 1] - 0.5))
      for (let x = from; x <= to; x++) mask[y * w + x] = 255
    }
  }
  return mask
}

export function rectPoints(x0: number, y0: number, x1: number, y1: number): Point[] {
  return [[x0, y0], [x1, y0], [x1, y1], [x0, y1]]
}

export function ellipsePoints(x0: number, y0: number, x1: number, y1: number, segments = 96): Point[] {
  const cx = (x0 + x1) / 2
  const cy = (y0 + y1) / 2
  const rx = Math.abs(x1 - x0) / 2
  const ry = Math.abs(y1 - y0) / 2
  return Array.from({ length: segments }, (_, i) => {
    const t = (i / segments) * Math.PI * 2
    return [cx + rx * Math.cos(t), cy + ry * Math.sin(t)] as Point
  })
}

/** A round brush dragged from (x0, y0) to (x1, y1), in place: paints
 * (raises the selection) or erases (lowers it). `hardness` 0..1 is how much
 * of the radius is fully on before it fades out. */
export function brushStroke(
  mask: Uint8Array,
  w: number,
  h: number,
  x0: number,
  y0: number,
  x1: number,
  y1: number,
  radius: number,
  hardness: number,
  erase: boolean,
): void {
  const r = Math.max(0.5, radius)
  const inner = r * clamp(hardness, 0, 1)
  const minX = Math.max(0, Math.floor(Math.min(x0, x1) - r))
  const maxX = Math.min(w - 1, Math.ceil(Math.max(x0, x1) + r))
  const minY = Math.max(0, Math.floor(Math.min(y0, y1) - r))
  const maxY = Math.min(h - 1, Math.ceil(Math.max(y0, y1) + r))
  const dx = x1 - x0
  const dy = y1 - y0
  const len2 = dx * dx + dy * dy
  for (let y = minY; y <= maxY; y++) {
    for (let x = minX; x <= maxX; x++) {
      const px = x + 0.5
      const py = y + 0.5
      // Distance to the segment.
      const t = len2 > 0 ? clamp(((px - x0) * dx + (py - y0) * dy) / len2, 0, 1) : 0
      const d = Math.hypot(px - (x0 + t * dx), py - (y0 + t * dy))
      if (d >= r) continue
      const a = d <= inner ? 1 : 1 - (d - inner) / Math.max(1e-6, r - inner)
      const v = Math.round(a * 255)
      const i = y * w + x
      if (erase) {
        const keep = 255 - v
        if (keep < mask[i]) mask[i] = keep
      } else if (v > mask[i]) {
        mask[i] = v
      }
    }
  }
}

/** Pixels of a colour like the one at (x, y): within `tolerance` (ΔE, in
 * Lab) of it, and joined to it unless `contiguous` is off. */
export function magicWand(lab: Float32Array, w: number, h: number, x: number, y: number, tolerance: number, contiguous: boolean): Uint8Array {
  const mask = new Uint8Array(w * h)
  const sx = clamp(Math.floor(x), 0, w - 1)
  const sy = clamp(Math.floor(y), 0, h - 1)
  // The clicked colour, averaged over 3×3 so one noisy pixel isn't it.
  let l = 0
  let a = 0
  let b = 0
  let n = 0
  for (let yy = Math.max(0, sy - 1); yy <= Math.min(h - 1, sy + 1); yy++) {
    for (let xx = Math.max(0, sx - 1); xx <= Math.min(w - 1, sx + 1); xx++) {
      const i = (yy * w + xx) * 3
      l += lab[i]
      a += lab[i + 1]
      b += lab[i + 2]
      n++
    }
  }
  l /= n
  a /= n
  b /= n
  const tol2 = tolerance * tolerance
  const near = (i: number) => {
    const dl = lab[i * 3] - l
    const da = lab[i * 3 + 1] - a
    const db = lab[i * 3 + 2] - b
    return dl * dl + da * da + db * db <= tol2
  }
  if (!contiguous) {
    for (let i = 0; i < w * h; i++) if (near(i)) mask[i] = 255
    return mask
  }
  const stack = new Int32Array(w * h)
  let top = 0
  const start = sy * w + sx
  mask[start] = 255
  stack[top++] = start
  while (top > 0) {
    const i = stack[--top]
    const px = i % w
    const py = (i - px) / w
    for (let k = 0; k < 4; k++) {
      const j = k === 0 ? (px > 0 ? i - 1 : -1) : k === 1 ? (px < w - 1 ? i + 1 : -1) : k === 2 ? (py > 0 ? i - w : -1) : py < h - 1 ? i + w : -1
      if (j < 0 || mask[j]) continue
      if (near(j)) {
        mask[j] = 255
        stack[top++] = j
      }
    }
  }
  return mask
}

export function invertMask(mask: Uint8Array): void {
  for (let i = 0; i < mask.length; i++) mask[i] = 255 - mask[i]
}

/** How much of the picture is selected, 0..1. */
export function maskShare(mask: Uint8Array): number {
  if (mask.length === 0) return 0
  let sum = 0
  for (let i = 0; i < mask.length; i++) sum += mask[i]
  return sum / (mask.length * 255)
}

/** The selection with soft edges: blurred by `sigma` pixels. */
export function featherMask(mask: Uint8Array, w: number, h: number, sigma: number): Uint8Array {
  if (!(sigma > 0.3)) return mask
  const f = new Float32Array(mask.length)
  for (let i = 0; i < mask.length; i++) f[i] = mask[i]
  gaussianBlur(f, w, h, sigma, 1)
  const out = new Uint8Array(mask.length)
  for (let i = 0; i < mask.length; i++) out[i] = Math.round(clamp(f[i], 0, 255))
  return out
}

/** Grow (positive) or shrink (negative) the selection by about `pixels`. */
export function growMask(mask: Uint8Array, w: number, h: number, pixels: number): Uint8Array {
  if (pixels === 0) return mask.slice()
  const f = new Float32Array(mask.length)
  for (let i = 0; i < mask.length; i++) f[i] = mask[i] / 255
  gaussianBlur(f, w, h, Math.abs(pixels) / 2, 1)
  // A blurred edge passes 0.5 at the old border; moving the threshold moves it.
  const threshold = pixels > 0 ? 0.08 : 0.92
  const out = new Uint8Array(mask.length)
  for (let i = 0; i < mask.length; i++) out[i] = f[i] > threshold ? 255 : 0
  return out
}

/** The selection's outline as an SVG path in pixel coordinates: every edge
 * between a selected pixel (at least half selected) and one that isn't,
 * straight runs merged into one segment. '' for nothing selected. */
export function maskOutline(mask: Uint8Array, w: number, h: number): { path: string; segments: number } {
  const inside = (x: number, y: number) => x >= 0 && y >= 0 && x < w && y < h && mask[y * w + x] >= 128
  const parts: string[] = []
  // Horizontal edges, on the line between rows y - 1 and y.
  for (let y = 0; y <= h; y++) {
    let start = -1
    for (let x = 0; x <= w; x++) {
      const edge = x < w && inside(x, y - 1) !== inside(x, y)
      if (edge && start < 0) start = x
      else if (!edge && start >= 0) {
        parts.push(`M${start} ${y}H${x}`)
        start = -1
      }
    }
  }
  // Vertical edges, on the line between columns x - 1 and x.
  for (let x = 0; x <= w; x++) {
    let start = -1
    for (let y = 0; y <= h; y++) {
      const edge = y < h && inside(x - 1, y) !== inside(x, y)
      if (edge && start < 0) start = y
      else if (!edge && start >= 0) {
        parts.push(`M${x} ${start}V${y}`)
        start = -1
      }
    }
  }
  return { path: parts.join(''), segments: parts.length }
}
