/** The image editor's adjustments: light, colour, levels, curves and
 * detail. Everything tonal folds into one lookup table per channel (fast at
 * any size); hue/saturation work per pixel; soften/sharpen are blurs whose
 * radius follows the picture's size, so a preview matches the full image. */

import { clamp, gaussianBlur, linearToSrgb, srgbToLinear } from './imageMath.ts'

export type CurvePoint = [number, number]

export interface Levels {
  inBlack: number
  inWhite: number
  gamma: number
  outBlack: number
  outWhite: number
}

export interface Curves {
  master: CurvePoint[]
  r: CurvePoint[]
  g: CurvePoint[]
  b: CurvePoint[]
}

export interface Adjustments {
  /** Stops, -3..3. */
  exposure: number
  /** The rest of the sliders run -100..100 (hue: degrees, -180..180). */
  brightness: number
  contrast: number
  highlights: number
  shadows: number
  whites: number
  blacks: number
  temperature: number
  tint: number
  hue: number
  saturation: number
  vibrance: number
  levels: Levels
  curves: Curves
  /** 0..100. */
  soften: number
  sharpen: number
  /** Fewer, flatter colours (see colorReduce). */
  reduce: { enabled: boolean; colors: number }
}

export type SliderKey =
  | 'exposure' | 'brightness' | 'contrast' | 'highlights' | 'shadows' | 'whites' | 'blacks'
  | 'temperature' | 'tint' | 'hue' | 'saturation' | 'vibrance' | 'soften' | 'sharpen'

export const IDENTITY_CURVE: CurvePoint[] = [[0, 0], [255, 255]]
export const DEFAULT_LEVELS: Levels = { inBlack: 0, inWhite: 255, gamma: 1, outBlack: 0, outWhite: 255 }

export function defaultAdjustments(): Adjustments {
  return {
    exposure: 0, brightness: 0, contrast: 0, highlights: 0, shadows: 0, whites: 0, blacks: 0,
    temperature: 0, tint: 0, hue: 0, saturation: 0, vibrance: 0,
    levels: { ...DEFAULT_LEVELS },
    curves: { master: IDENTITY_CURVE.map((p) => [...p] as CurvePoint), r: IDENTITY_CURVE.map((p) => [...p] as CurvePoint), g: IDENTITY_CURVE.map((p) => [...p] as CurvePoint), b: IDENTITY_CURVE.map((p) => [...p] as CurvePoint) },
    soften: 0, sharpen: 0,
    reduce: { enabled: false, colors: 16 },
  }
}

/** The sliders: range, step and what they do, in the order shown. */
export const SLIDERS: Record<SliderKey, { label: string; min: number; max: number; step: number; hint: string; unit?: string }> = {
  exposure: { label: 'Exposure', min: -3, max: 3, step: 0.05, unit: ' EV', hint: 'Like more or less light on the photo: everything brighter or darker, in stops.' },
  brightness: { label: 'Brightness', min: -100, max: 100, step: 1, hint: 'Lifts or lowers the midtones; the darkest and brightest parts stay put.' },
  contrast: { label: 'Contrast', min: -100, max: 100, step: 1, hint: 'Pushes darks and lights apart (or closer together).' },
  highlights: { label: 'Highlights', min: -100, max: 100, step: 1, hint: 'Only the bright parts: lower it to bring back detail in a bright sky.' },
  shadows: { label: 'Shadows', min: -100, max: 100, step: 1, hint: 'Only the dark parts: raise it to open up detail in shadows.' },
  whites: { label: 'Whites', min: -100, max: 100, step: 1, hint: 'Where white starts: raise it for cleaner whites, lower it to tone them down.' },
  blacks: { label: 'Blacks', min: -100, max: 100, step: 1, hint: 'The black level: lower it for deeper blacks, raise it to lift them.' },
  temperature: { label: 'Temperature', min: -100, max: 100, step: 1, hint: 'Cooler (blue) or warmer (yellow).' },
  tint: { label: 'Tint', min: -100, max: 100, step: 1, hint: 'Greener or more magenta.' },
  hue: { label: 'Hue', min: -180, max: 180, step: 1, unit: '°', hint: 'Turns every colour around the colour wheel.' },
  saturation: { label: 'Saturation', min: -100, max: 100, step: 1, hint: 'All colours stronger or weaker; -100 is black and white.' },
  vibrance: { label: 'Vibrance', min: -100, max: 100, step: 1, hint: 'Like saturation, but mostly for dull colours; strong ones change less.' },
  soften: { label: 'Soften', min: 0, max: 100, step: 1, hint: 'Smooths noise and fine texture. Small specks rarely print well, so a little often helps.' },
  sharpen: { label: 'Sharpen', min: 0, max: 100, step: 1, hint: 'Crisper edges and fine detail.' },
}

export const LIGHT_SLIDERS: SliderKey[] = ['exposure', 'brightness', 'contrast', 'highlights', 'shadows', 'whites', 'blacks']
export const COLOR_SLIDERS: SliderKey[] = ['temperature', 'tint', 'hue', 'saturation', 'vibrance']
export const DETAIL_SLIDERS: SliderKey[] = ['soften', 'sharpen']

const sameCurve = (c: CurvePoint[]) => c.length === 2 && c[0][0] === 0 && c[0][1] === 0 && c[1][0] === 255 && c[1][1] === 255
const sameLevels = (l: Levels) =>
  l.inBlack === 0 && l.inWhite === 255 && l.gamma === 1 && l.outBlack === 0 && l.outWhite === 255

export function curvesAreIdentity(c: Curves): boolean {
  return sameCurve(c.master) && sameCurve(c.r) && sameCurve(c.g) && sameCurve(c.b)
}

export function levelsAreIdentity(l: Levels): boolean {
  return sameLevels(l)
}

/** True when the adjustments change nothing. */
export function isIdentity(a: Adjustments): boolean {
  return (Object.keys(SLIDERS) as SliderKey[]).every((k) => a[k] === 0)
    && levelsAreIdentity(a.levels) && curvesAreIdentity(a.curves) && !a.reduce.enabled
}

/** How many settings of a group differ from their defaults (for the
 * sidebar's section badges). */
export function changedCount(a: Adjustments, keys: SliderKey[]): number {
  return keys.filter((k) => a[k] !== 0).length
}

/** A smooth, monotone-where-the-points-are curve through `points` (any
 * order, x 0..255), sampled at 0..255. Flat beyond the outermost points. */
export function curveLut(points: CurvePoint[]): Float32Array {
  const pts = [...points].sort((p, q) => p[0] - q[0]).filter((p, i, all) => i === 0 || p[0] !== all[i - 1][0])
  const lut = new Float32Array(256)
  if (pts.length === 0) {
    for (let i = 0; i < 256; i++) lut[i] = i
    return lut
  }
  if (pts.length === 1) return lut.fill(pts[0][1])
  const n = pts.length
  const dx: number[] = []
  const slope: number[] = []
  for (let i = 0; i < n - 1; i++) {
    dx.push(pts[i + 1][0] - pts[i][0])
    slope.push((pts[i + 1][1] - pts[i][1]) / dx[i])
  }
  // Fritsch-Carlson tangents: no overshoot between points.
  const m = new Array<number>(n)
  m[0] = slope[0]
  m[n - 1] = slope[n - 2]
  for (let i = 1; i < n - 1; i++) {
    m[i] = slope[i - 1] * slope[i] <= 0 ? 0 : (slope[i - 1] + slope[i]) / 2
  }
  for (let i = 0; i < n - 1; i++) {
    if (slope[i] === 0) {
      m[i] = 0
      m[i + 1] = 0
      continue
    }
    const a = m[i] / slope[i]
    const b = m[i + 1] / slope[i]
    const s = a * a + b * b
    if (s > 9) {
      const t = 3 / Math.sqrt(s)
      m[i] = t * a * slope[i]
      m[i + 1] = t * b * slope[i]
    }
  }
  let seg = 0
  for (let x = 0; x < 256; x++) {
    if (x <= pts[0][0]) {
      lut[x] = pts[0][1]
      continue
    }
    if (x >= pts[n - 1][0]) {
      lut[x] = pts[n - 1][1]
      continue
    }
    while (x > pts[seg + 1][0]) seg++
    const h = dx[seg]
    const t = (x - pts[seg][0]) / h
    const t2 = t * t
    const t3 = t2 * t
    lut[x] =
      (2 * t3 - 3 * t2 + 1) * pts[seg][1] + (t3 - 2 * t2 + t) * h * m[seg] + (-2 * t3 + 3 * t2) * pts[seg + 1][1] + (t3 - t2) * h * m[seg + 1]
  }
  for (let x = 0; x < 256; x++) lut[x] = clamp(lut[x], 0, 255)
  return lut
}

/** Levels as a function of a 0..255 value. */
export function applyLevels(v: number, l: Levels): number {
  const span = Math.max(1, l.inWhite - l.inBlack)
  const t = clamp((v - l.inBlack) / span, 0, 1)
  const g = Math.pow(t, 1 / clamp(l.gamma, 0.1, 10))
  return l.outBlack + g * (l.outWhite - l.outBlack)
}

const smooth01 = (t: number) => {
  const c = clamp(t, 0, 1)
  return c * c
}

/** The tone of one channel value (0..1, after white balance and exposure)
 * through blacks/whites, brightness, contrast and shadows/highlights. */
function tone(v: number, a: Adjustments): number {
  // Black and white points.
  const black = -a.blacks / 100 * 0.2
  const white = 1 - a.whites / 100 * 0.25
  v = (v - black) / Math.max(0.05, white - black)
  v = clamp(v, 0, 1)
  // Brightness: a midtone gamma.
  if (a.brightness !== 0) v = Math.pow(v, Math.pow(2, -a.brightness / 100))
  // Contrast: towards an S-curve, or towards mid grey.
  if (a.contrast > 0) {
    const s = v < 0.5 ? 2 * v * v : 1 - 2 * (1 - v) * (1 - v)
    v = v + (s - v) * (a.contrast / 100)
  } else if (a.contrast < 0) {
    v = v + (0.5 + (v - 0.5) * 0.4 - v) * (-a.contrast / 100)
  }
  // Shadows and highlights only move their own end of the range.
  if (a.shadows !== 0) {
    const w = smooth01((0.65 - v) / 0.65)
    v += (a.shadows / 100) * 0.4 * w * (a.shadows > 0 ? 1 - v : v)
  }
  if (a.highlights !== 0) {
    const w = smooth01((v - 0.35) / 0.65)
    v += (a.highlights / 100) * 0.4 * w * (a.highlights > 0 ? 1 - v : v)
  }
  return clamp(v, 0, 1)
}

/** The lookup tables (0..255 in, 0..255 out) for red, green and blue: white
 * balance, exposure, tone, levels and curves in one step each. */
export function buildLuts(a: Adjustments): [Uint8Array, Uint8Array, Uint8Array] {
  const exposure = Math.pow(2, a.exposure)
  const gains = [
    exposure * Math.pow(2, (a.temperature / 100) * 0.45 + (a.tint / 100) * 0.15),
    exposure * Math.pow(2, -(a.tint / 100) * 0.3),
    exposure * Math.pow(2, -(a.temperature / 100) * 0.45 + (a.tint / 100) * 0.15),
  ]
  const master = curveLut(a.curves.master)
  const channelCurves = [curveLut(a.curves.r), curveLut(a.curves.g), curveLut(a.curves.b)]
  const luts: [Uint8Array, Uint8Array, Uint8Array] = [new Uint8Array(256), new Uint8Array(256), new Uint8Array(256)]
  for (let c = 0; c < 3; c++) {
    // Never let a slider turn darker into lighter (a curve or inverted
    // output levels may, on purpose).
    const mustRise = sameCurve(a.curves.master) && sameCurve(a.curves[(['r', 'g', 'b'] as const)[c]]) && a.levels.outBlack <= a.levels.outWhite
    let previous = 0
    for (let i = 0; i < 256; i++) {
      let v = linearToSrgb(srgbToLinear(i) * gains[c])
      v = tone(v, a) * 255
      v = applyLevels(v, a.levels)
      v = master[Math.round(clamp(v, 0, 255))]
      v = channelCurves[c][Math.round(clamp(v, 0, 255))]
      let out = Math.round(clamp(v, 0, 255))
      if (mustRise && out < previous) out = previous
      previous = out
      luts[c][i] = out
    }
  }
  return luts
}

/** The adjustments applied to an RGBA buffer, into `out` (may be `src`).
 * `scale` is the picture's size relative to a 1000 px one (soften/sharpen
 * radii follow it). Alpha is kept. */
export function applyAdjustments(
  src: Uint8ClampedArray,
  w: number,
  h: number,
  a: Adjustments,
  out: Uint8ClampedArray = new Uint8ClampedArray(src.length),
  scale = 1,
): Uint8ClampedArray {
  const n = w * h
  const [lr, lg, lb] = buildLuts(a)
  const colour = a.hue !== 0 || a.saturation !== 0 || a.vibrance !== 0
  const angle = (a.hue * Math.PI) / 180
  const cos = Math.cos(angle)
  const sin = Math.sin(angle)
  const sat = 1 + a.saturation / 100
  const vib = a.vibrance / 100
  for (let i = 0; i < n; i++) {
    const p = i * 4
    let r = lr[src[p]]
    let g = lg[src[p + 1]]
    let b = lb[src[p + 2]]
    if (colour) {
      // YIQ: rotate the hue, scale the chroma.
      const y = 0.299 * r + 0.587 * g + 0.114 * b
      let ci = 0.596 * r - 0.274 * g - 0.322 * b
      let cq = 0.211 * r - 0.523 * g + 0.312 * b
      let k = sat
      if (vib !== 0) {
        const mx = Math.max(r, g, b)
        const mn = Math.min(r, g, b)
        const s = mx === 0 ? 0 : (mx - mn) / mx
        k *= 1 + vib * (1 - s)
      }
      const ri = (ci * cos - cq * sin) * k
      const rq = (ci * sin + cq * cos) * k
      ci = ri
      cq = rq
      r = y + 0.956 * ci + 0.621 * cq
      g = y - 0.272 * ci - 0.647 * cq
      b = y - 1.106 * ci + 1.703 * cq
    }
    out[p] = r
    out[p + 1] = g
    out[p + 2] = b
    out[p + 3] = src[p + 3]
  }
  if (a.soften > 0 || a.sharpen > 0) detail(out, w, h, a, scale)
  return out
}

/** Soften (a blur) and sharpen (an unsharp mask), in place. */
function detail(buf: Uint8ClampedArray, w: number, h: number, a: Adjustments, scale: number): void {
  const n = w * h
  const rgb = new Float32Array(n * 3)
  for (let i = 0; i < n; i++) {
    rgb[i * 3] = buf[i * 4]
    rgb[i * 3 + 1] = buf[i * 4 + 1]
    rgb[i * 3 + 2] = buf[i * 4 + 2]
  }
  if (a.soften > 0) gaussianBlur(rgb, w, h, (a.soften / 100) * 4 * scale, 3)
  if (a.sharpen > 0) {
    const blurred = rgb.slice()
    gaussianBlur(blurred, w, h, Math.max(0.5, 1.2 * scale), 3)
    const amount = (a.sharpen / 100) * 1.5
    for (let i = 0; i < rgb.length; i++) rgb[i] += (rgb[i] - blurred[i]) * amount
  }
  for (let i = 0; i < n; i++) {
    buf[i * 4] = rgb[i * 3]
    buf[i * 4 + 1] = rgb[i * 3 + 1]
    buf[i * 4 + 2] = rgb[i * 3 + 2]
  }
}

/** Per-channel and luminance histograms (256 bins each) of an RGBA buffer,
 * counting only opaque pixels. */
export function histogram(rgba: Uint8ClampedArray | Uint8Array): { r: Uint32Array; g: Uint32Array; b: Uint32Array; l: Uint32Array } {
  const r = new Uint32Array(256)
  const g = new Uint32Array(256)
  const b = new Uint32Array(256)
  const l = new Uint32Array(256)
  for (let p = 0; p < rgba.length; p += 4) {
    if (rgba[p + 3] < 128) continue
    r[rgba[p]]++
    g[rgba[p + 1]]++
    b[rgba[p + 2]]++
    l[Math.round(0.2126 * rgba[p] + 0.7152 * rgba[p + 1] + 0.0722 * rgba[p + 2])]++
  }
  return { r, g, b, l }
}

/** Levels that stretch the picture's own darkest and brightest 0.5% to
 * full black and white (the "Auto" button). */
export function autoLevels(hist: Uint32Array): Levels {
  let total = 0
  for (let i = 0; i < 256; i++) total += hist[i]
  if (total === 0) return { ...DEFAULT_LEVELS }
  const cut = total * 0.005
  let lo = 0
  let acc = 0
  while (lo < 255 && acc + hist[lo] <= cut) acc += hist[lo++]
  let hi = 255
  acc = 0
  while (hi > 0 && acc + hist[hi] <= cut) acc += hist[hi--]
  if (hi - lo < 10) return { ...DEFAULT_LEVELS }
  return { ...DEFAULT_LEVELS, inBlack: lo, inWhite: hi }
}
