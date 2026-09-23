/** The "Differences" image view: where the printed result strays from the
 * picture, as a heatmap over a dimmed grey copy of the picture.
 *
 * The distance is CIE76 ΔE in Lab — the same color space the optimizer's
 * loss works in, so what lights up here is what the optimizer was trying to
 * shrink. Rough reading: below ~2 nobody can tell, ~2–10 a close look shows
 * it, above ~10 it's plainly a different color.
 */

/** ΔE at which the heatmap reaches its brightest color. */
export const DIFF_SCALE_MAX = 30
/** ΔE from which a pixel counts as "clearly different". */
export const CLEARLY_DIFFERENT = 10

const SRGB_TO_LINEAR = (() => {
  const lut = new Float32Array(256)
  for (let i = 0; i < 256; i++) {
    const c = i / 255
    lut[i] = c <= 0.04045 ? c / 12.92 : Math.pow((c + 0.055) / 1.055, 2.4)
  }
  return lut
})()

function labF(t: number): number {
  return t > 0.008856 ? Math.cbrt(t) : 7.787 * t + 16 / 116
}

/** sRGB (0–255) to CIE Lab (D65). */
export function rgbToLab(r: number, g: number, b: number): [number, number, number] {
  const lr = SRGB_TO_LINEAR[r], lg = SRGB_TO_LINEAR[g], lb = SRGB_TO_LINEAR[b]
  const x = (lr * 0.4124564 + lg * 0.3575761 + lb * 0.1804375) / 0.95047
  const y = lr * 0.2126729 + lg * 0.7151522 + lb * 0.072175
  const z = (lr * 0.0193339 + lg * 0.119192 + lb * 0.9503041) / 1.08883
  const fx = labF(x), fy = labF(y), fz = labF(z)
  return [116 * fy - 16, 500 * (fx - fy), 200 * (fy - fz)]
}

export function deltaE(a: [number, number, number], b: [number, number, number]): number {
  return Math.hypot(a[0] - b[0], a[1] - b[1], a[2] - b[2])
}

// Inferno-like ramp: dark purple (tiny difference) → red → orange → pale
// yellow (large). Perceptually ordered, and readable for the common kinds of
// color blindness, unlike a green→red ramp.
const RAMP: [number, number, number][] = [
  [40, 11, 84],
  [120, 28, 109],
  [187, 55, 84],
  [237, 105, 37],
  [251, 180, 26],
  [252, 255, 164],
]

/** Heatmap color for t in [0, 1]. */
export function heatColor(t: number): [number, number, number] {
  const x = Math.max(0, Math.min(1, t)) * (RAMP.length - 1)
  const i = Math.min(RAMP.length - 2, Math.floor(x))
  const f = x - i
  const a = RAMP[i], b = RAMP[i + 1]
  return [Math.round(a[0] + (b[0] - a[0]) * f), Math.round(a[1] + (b[1] - a[1]) * f), Math.round(a[2] + (b[2] - a[2]) * f)]
}

/** CSS gradient of the same ramp, for the legend. */
export function heatGradientCss(): string {
  return `linear-gradient(to right, ${RAMP.map(([r, g, b], i) => `rgb(${r},${g},${b}) ${Math.round((i / (RAMP.length - 1)) * 100)}%`).join(', ')})`
}

export interface DiffStats {
  /** Average ΔE over the compared (opaque) pixels. */
  mean: number
  /** Share (0–1) of compared pixels at or above CLEARLY_DIFFERENT. */
  clearlyDifferent: number
  /** ΔE shown as the brightest color. */
  scaleMax: number
}

/** The heatmap's brightest color is this image's own 95th-percentile
 * difference, so it shows where the result is worst *relative to the rest*
 * — which is what you need to decide where to paint focus areas. A fixed
 * scale saturated the whole picture whenever a result was off overall. The
 * floor keeps a near-perfect result from having its tiny differences blown
 * up into alarming colors. */
export const DIFF_SCALE_FLOOR = 8

/** Separable box blur (in place), so the map shows regions rather than
 * single-pixel noise from fine texture. Transparent pixels (NaN) are left
 * out of their neighbours' averages. */
export function boxBlur(values: Float32Array, width: number, height: number, radius: number): Float32Array {
  if (radius < 1) return values
  const tmp = new Float32Array(values.length)
  const pass = (src: Float32Array, dst: Float32Array, horizontal: boolean) => {
    const outer = horizontal ? height : width
    const inner = horizontal ? width : height
    for (let o = 0; o < outer; o++) {
      for (let i = 0; i < inner; i++) {
        let sum = 0
        let count = 0
        for (let k = Math.max(0, i - radius); k <= Math.min(inner - 1, i + radius); k++) {
          const v = src[horizontal ? o * width + k : k * width + o]
          if (v === v) {
            sum += v
            count++
          }
        }
        const idx = horizontal ? o * width + i : i * width + o
        dst[idx] = src[idx] === src[idx] && count ? sum / count : NaN
      }
    }
  }
  pass(values, tmp, true)
  pass(tmp, values, false)
  return values
}

/** The value below which `share` of the (non-NaN) values fall; a histogram
 * rather than a sort, since this runs over the whole picture. */
export function percentile(values: Float32Array, share: number, max = 100, bins = 400): number {
  const hist = new Uint32Array(bins)
  let n = 0
  for (const v of values) {
    if (v !== v) continue
    hist[Math.min(bins - 1, Math.floor((v / max) * bins))]++
    n++
  }
  if (!n) return 0
  const target = share * n
  let seen = 0
  for (let b = 0; b < bins; b++) {
    seen += hist[b]
    if (seen >= target) return ((b + 1) / bins) * max
  }
  return max
}

/** Compares two same-sized RGBA buffers and returns the heatmap (RGBA, same
 * size) with summary numbers. Pixels transparent in the original are not
 * part of the print and are left transparent and uncounted. */
export function diffHeatmap(
  original: Uint8ClampedArray,
  result: Uint8ClampedArray,
  width: number,
  height: number,
): { rgba: Uint8ClampedArray; stats: DiffStats } {
  const n = width * height
  const diff = new Float32Array(n)
  const grey = new Float32Array(n)
  let sum = 0
  let counted = 0
  let over = 0
  for (let p = 0; p < n; p++) {
    const i = p * 4
    if (original[i + 3] < 128) {
      diff[p] = NaN
      continue
    }
    const o = rgbToLab(original[i], original[i + 1], original[i + 2])
    const d = deltaE(o, rgbToLab(result[i], result[i + 1], result[i + 2]))
    diff[p] = d
    grey[p] = (o[0] / 100) * 255 * 0.4
    sum += d
    counted++
    if (d >= CLEARLY_DIFFERENT) over++
  }

  boxBlur(diff, width, height, Math.round(Math.max(width, height) / 300))
  const scaleMax = Math.max(DIFF_SCALE_FLOOR, Math.min(DIFF_SCALE_MAX * 2, percentile(diff, 0.95)))

  // Dimmed grey picture underneath for orientation; the heat color on top,
  // more opaque the bigger the difference, so matching areas stay quiet and
  // problem areas jump out.
  const out = new Uint8ClampedArray(n * 4)
  for (let p = 0; p < n; p++) {
    const d = diff[p]
    if (d !== d) continue
    const i = p * 4
    // A contrast curve: mid-sized differences stay dim, so the worst areas
    // stand out instead of the whole picture glowing when a result is off
    // everywhere.
    const t = Math.pow(Math.min(1, d / scaleMax), 1.6)
    const [hr, hg, hb] = heatColor(t)
    const alpha = 0.1 + 0.9 * t
    const g = grey[p]
    out[i] = g + (hr - g) * alpha
    out[i + 1] = g + (hg - g) * alpha
    out[i + 2] = g + (hb - g) * alpha
    out[i + 3] = 255
  }
  return {
    rgba: out,
    stats: { mean: counted ? sum / counted : 0, clearlyDifferent: counted ? over / counted : 0, scaleMax },
  }
}

/** Plain-language verdict for the average ΔE. */
export function describeMeanDifference(mean: number): string {
  if (mean < 3) return 'Very close match'
  if (mean < 6) return 'Close match'
  if (mean < 10) return 'Noticeable differences'
  return 'Large differences'
}

/** Size to compare at: the result's own size, but no bigger than `maxSide`
 * on its longest side (the comparison runs on the main thread). */
export function comparisonSize(width: number, height: number, maxSide = 900): { width: number; height: number } {
  const scale = Math.min(1, maxSide / Math.max(width, height, 1))
  return { width: Math.max(1, Math.round(width * scale)), height: Math.max(1, Math.round(height * scale)) }
}
