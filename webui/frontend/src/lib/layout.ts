/** Size of a resizable pane: within [min, max], and min wins if the space is
 * too small for both. */
export function clampSize(size: number, min: number, max: number): number {
  if (!Number.isFinite(size)) return min
  return Math.max(min, Math.min(max, size))
}

export interface ZoomTransform {
  scale: number
  x: number
  y: number
}

export const IDENTITY_ZOOM: ZoomTransform = { scale: 1, x: 0, y: 0 }

/** Zooms by `factor` keeping the point (cx, cy) — relative to the viewport's
 * top-left — under the cursor. Back at 1× the view is re-centred. */
export function zoomAt(t: ZoomTransform, factor: number, cx: number, cy: number, min = 1, max = 16): ZoomTransform {
  const scale = clampSize(t.scale * factor, min, max)
  if (scale === 1) return IDENTITY_ZOOM
  const ratio = scale / t.scale
  return { scale, x: cx - (cx - t.x) * ratio, y: cy - (cy - t.y) * ratio }
}
