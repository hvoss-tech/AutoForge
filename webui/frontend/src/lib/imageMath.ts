/** Colour spaces and blurs shared by the image editor's adjustments,
 * selections and colour reduction. Pure functions on typed arrays, so they
 * run the same in the page, in the editor's worker and in the unit tests. */

export const clamp = (v: number, lo: number, hi: number) => (v < lo ? lo : v > hi ? hi : v)

/** sRGB 0-255 to linear 0-1. */
const SRGB_TO_LINEAR = new Float32Array(256)
for (let i = 0; i < 256; i++) {
  const c = i / 255
  SRGB_TO_LINEAR[i] = c <= 0.04045 ? c / 12.92 : Math.pow((c + 0.055) / 1.055, 2.4)
}
export const srgbToLinear = (v: number) => SRGB_TO_LINEAR[v | 0]

/** Linear 0-1 (any value) to sRGB 0-1, clamped. */
export function linearToSrgb(c: number): number {
  if (c <= 0) return 0
  if (c >= 1) return 1
  return c <= 0.0031308 ? c * 12.92 : 1.055 * Math.pow(c, 1 / 2.4) - 0.055
}

// D65 white.
const XN = 0.95047
const YN = 1.0
const ZN = 1.08883
const labF = (t: number) => (t > 216 / 24389 ? Math.cbrt(t) : (24389 / 27 * t + 16) / 116)
const labFInv = (t: number) => (t * t * t > 216 / 24389 ? t * t * t : (116 * t - 16) / (24389 / 27))

/** sRGB 0-255 to CIELAB (D65), written to out[o..o+2]. */
export function rgbToLab(r: number, g: number, b: number, out: Float32Array | number[], o = 0): void {
  const rl = SRGB_TO_LINEAR[r | 0]
  const gl = SRGB_TO_LINEAR[g | 0]
  const bl = SRGB_TO_LINEAR[b | 0]
  const x = (rl * 0.4124564 + gl * 0.3575761 + bl * 0.1804375) / XN
  const y = (rl * 0.2126729 + gl * 0.7151522 + bl * 0.072175) / YN
  const z = (rl * 0.0193339 + gl * 0.119192 + bl * 0.9503041) / ZN
  const fx = labF(x)
  const fy = labF(y)
  const fz = labF(z)
  out[o] = 116 * fy - 16
  out[o + 1] = 500 * (fx - fy)
  out[o + 2] = 200 * (fy - fz)
}

/** CIELAB (D65) to sRGB 0-255 (rounded, clamped), written to out[o..o+2]. */
export function labToRgb(l: number, a: number, b: number, out: Uint8Array | Uint8ClampedArray | number[], o = 0): void {
  const fy = (l + 16) / 116
  const fx = fy + a / 500
  const fz = fy - b / 200
  const x = labFInv(fx) * XN
  const y = labFInv(fy) * YN
  const z = labFInv(fz) * ZN
  const rl = x * 3.2404542 - y * 1.5371385 - z * 0.4985314
  const gl = -x * 0.969266 + y * 1.8760108 + z * 0.041556
  const bl = x * 0.0556434 - y * 0.2040259 + z * 1.0572252
  out[o] = Math.round(linearToSrgb(rl) * 255)
  out[o + 1] = Math.round(linearToSrgb(gl) * 255)
  out[o + 2] = Math.round(linearToSrgb(bl) * 255)
}

/** Every pixel of an RGBA buffer in Lab: 3 floats per pixel. */
export function rgbaToLab(rgba: Uint8Array | Uint8ClampedArray, count: number): Float32Array {
  const lab = new Float32Array(count * 3)
  for (let i = 0; i < count; i++) rgbToLab(rgba[i * 4], rgba[i * 4 + 1], rgba[i * 4 + 2], lab, i * 3)
  return lab
}

/** Rec. 709 luma of sRGB 0-255 values, 0-255. */
export const luma = (r: number, g: number, b: number) => 0.2126 * r + 0.7152 * g + 0.0722 * b

/** Gaussian blur of `channels` interleaved float channels, in place,
 * separable with clamped edges. sigma <= 0.3 is a no-op. */
export function gaussianBlur(data: Float32Array, w: number, h: number, sigma: number, channels = 1): void {
  if (!(sigma > 0.3) || w === 0 || h === 0) return
  const radius = Math.min(Math.ceil(sigma * 3), Math.max(w, h))
  const kernel = new Float32Array(radius * 2 + 1)
  let sum = 0
  for (let i = -radius; i <= radius; i++) {
    const v = Math.exp(-(i * i) / (2 * sigma * sigma))
    kernel[i + radius] = v
    sum += v
  }
  for (let i = 0; i < kernel.length; i++) kernel[i] /= sum
  const line = new Float32Array(Math.max(w, h) * channels)
  // Rows.
  for (let y = 0; y < h; y++) {
    const row = y * w * channels
    line.set(data.subarray(row, row + w * channels))
    for (let x = 0; x < w; x++) {
      for (let c = 0; c < channels; c++) {
        let acc = 0
        for (let k = -radius; k <= radius; k++) {
          const xx = x + k < 0 ? 0 : x + k >= w ? w - 1 : x + k
          acc += line[xx * channels + c] * kernel[k + radius]
        }
        data[row + x * channels + c] = acc
      }
    }
  }
  // Columns.
  for (let x = 0; x < w; x++) {
    for (let y = 0; y < h; y++) for (let c = 0; c < channels; c++) line[y * channels + c] = data[(y * w + x) * channels + c]
    for (let y = 0; y < h; y++) {
      for (let c = 0; c < channels; c++) {
        let acc = 0
        for (let k = -radius; k <= radius; k++) {
          const yy = y + k < 0 ? 0 : y + k >= h ? h - 1 : y + k
          acc += line[yy * channels + c] * kernel[k + radius]
        }
        data[(y * w + x) * channels + c] = acc
      }
    }
  }
}

/** A single channel resized with bilinear filtering (masks between the
 * editor's working size and the full picture). */
export function resizeChannel(src: Uint8Array, sw: number, sh: number, dw: number, dh: number): Uint8Array {
  if (sw === dw && sh === dh) return src.slice()
  const out = new Uint8Array(dw * dh)
  const sx = sw / dw
  const sy = sh / dh
  for (let y = 0; y < dh; y++) {
    const fy = clamp((y + 0.5) * sy - 0.5, 0, sh - 1)
    const y0 = Math.floor(fy)
    const y1 = Math.min(sh - 1, y0 + 1)
    const ty = fy - y0
    for (let x = 0; x < dw; x++) {
      const fx = clamp((x + 0.5) * sx - 0.5, 0, sw - 1)
      const x0 = Math.floor(fx)
      const x1 = Math.min(sw - 1, x0 + 1)
      const tx = fx - x0
      const top = src[y0 * sw + x0] * (1 - tx) + src[y0 * sw + x1] * tx
      const bottom = src[y1 * sw + x0] * (1 - tx) + src[y1 * sw + x1] * tx
      out[y * dw + x] = Math.round(top * (1 - ty) + bottom * ty)
    }
  }
  return out
}
