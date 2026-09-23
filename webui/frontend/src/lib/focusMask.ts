/** Focus areas ("priority mask"): parts of the picture painted as more
 * important, so the optimizer spends its effort there.
 *
 * The backend (auto_forge._load_priority_mask) reads the mask as a
 * greyscale image — white = focus, black = everything else — and weights
 * each pixel's color error by 0.1 + 0.9 × mask. Painted areas therefore
 * count ten times as much as unpainted ones. It drops an alpha channel, so
 * the exported PNG must be opaque: painted strokes flattened onto black.
 *
 * In the browser the mask lives in a canvas whose *alpha* is the focus
 * strength (white strokes on a transparent canvas), which is what the
 * overlay and the eraser need.
 */

/** How much more a painted pixel counts than an unpainted one: the
 * `priority_mask_strength` setting (the backend scales the mask so the
 * weighting gives exactly this ratio). */
export const DEFAULT_FOCUS_STRENGTH = 10
export const MIN_FOCUS_STRENGTH = 2
export const MAX_FOCUS_STRENGTH = 100

/** The strength slider is logarithmic (0–100): 2× to 10× gets as much of
 * its travel as 10× to 100×, since the step from 2× to 4× matters as much
 * as the one from 50× to 100×. Whole numbers, rounded to 5 above 20. */
export function strengthFromSlider(position: number): number {
  const t = Math.max(0, Math.min(100, position)) / 100
  const raw = MIN_FOCUS_STRENGTH * Math.pow(MAX_FOCUS_STRENGTH / MIN_FOCUS_STRENGTH, t)
  return raw > 20 ? Math.round(raw / 5) * 5 : Math.round(raw)
}

export function sliderFromStrength(strength: number): number {
  const s = Math.max(MIN_FOCUS_STRENGTH, Math.min(MAX_FOCUS_STRENGTH, strength || DEFAULT_FOCUS_STRENGTH))
  return Math.round((Math.log(s / MIN_FOCUS_STRENGTH) / Math.log(MAX_FOCUS_STRENGTH / MIN_FOCUS_STRENGTH)) * 100)
}

/** The mask canvas' size: the picture's own size, capped so a large photo
 * doesn't cost a huge canvas (the backend resizes the mask to its working
 * size anyway). */
export function maskCanvasSize(imageWidth: number, imageHeight: number, maxSide = 1024): { width: number; height: number } {
  const scale = Math.min(1, maxSide / Math.max(imageWidth, imageHeight, 1))
  return { width: Math.max(1, Math.round(imageWidth * scale)), height: Math.max(1, Math.round(imageHeight * scale)) }
}

/** Brush diameter in mask pixels, from the size slider (percent of the
 * picture's longest side). */
export function brushDiameter(sizePercent: number, maskWidth: number, maskHeight: number): number {
  return Math.max(1, (sizePercent / 100) * Math.max(maskWidth, maskHeight))
}

/** Where a pointer at (clientX, clientY) lands in the mask, for a canvas
 * drawn with object-fit: contain inside `rect` (its on-screen box, CSS
 * transforms included). Null when it's outside the picture. */
export function clientToMask(
  clientX: number,
  clientY: number,
  rect: { left: number; top: number; width: number; height: number },
  maskWidth: number,
  maskHeight: number,
): { x: number; y: number; scale: number } | null {
  const scale = Math.min(rect.width / maskWidth, rect.height / maskHeight)
  if (!(scale > 0)) return null
  const offX = (rect.width - maskWidth * scale) / 2
  const offY = (rect.height - maskHeight * scale) / 2
  const x = (clientX - rect.left - offX) / scale
  const y = (clientY - rect.top - offY) / scale
  return { x, y, scale }
}

/** Alpha above which a mask pixel counts as painted. The faint
 * anti-aliased fringe an eraser stroke leaves behind stays below it, so
 * "erased everything" really saves no mask — the same rule the "% marked"
 * readout uses. */
const PAINTED_ALPHA = 127

/** True if any pixel of this RGBA buffer is painted. */
export function maskHasPaint(rgba: Uint8ClampedArray): boolean {
  for (let i = 3; i < rgba.length; i += 4) if (rgba[i] > PAINTED_ALPHA) return true
  return false
}

/** Share (0–1) of the picture that is painted, for the "12% marked" readout. */
export function paintedShare(rgba: Uint8ClampedArray): number {
  let painted = 0
  const n = rgba.length / 4
  for (let i = 3; i < rgba.length; i += 4) if (rgba[i] > PAINTED_ALPHA) painted++
  return n ? painted / n : 0
}

/** Painted-alpha canvas data → the opaque greyscale the backend reads
 * (in place: grey = alpha, alpha = 255). */
export function alphaToGreyscale(rgba: Uint8ClampedArray): Uint8ClampedArray {
  for (let i = 0; i < rgba.length; i += 4) {
    const a = rgba[i + 3]
    rgba[i] = a
    rgba[i + 1] = a
    rgba[i + 2] = a
    rgba[i + 3] = 255
  }
  return rgba
}

/** The saved greyscale mask → painted-alpha canvas data (in place), for
 * loading a mask back (undo, a restored session). */
export function greyscaleToAlpha(rgba: Uint8ClampedArray): Uint8ClampedArray {
  for (let i = 0; i < rgba.length; i += 4) {
    const grey = Math.round(0.299 * rgba[i] + 0.587 * rgba[i + 1] + 0.114 * rgba[i + 2])
    rgba[i] = 255
    rgba[i + 1] = 255
    rgba[i + 2] = 255
    rgba[i + 3] = grey
  }
  return rgba
}
