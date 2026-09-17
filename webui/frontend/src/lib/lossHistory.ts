export interface LossPoint {
  iteration: number
  loss: number
}

/** Appends a progress update, ignoring repeats of the same iteration. Once
 * past `max` points the older half is thinned out, so a long run keeps its
 * overall shape and full detail for the recent part. */
export function appendLossPoint(points: LossPoint[], iteration: number, loss: number, max = 240): LossPoint[] {
  if (!Number.isFinite(loss)) return points
  const last = points[points.length - 1]
  if (last && last.iteration === iteration) return points
  const next = last && iteration < last.iteration ? [{ iteration, loss }] : [...points, { iteration, loss }]
  if (next.length <= max) return next
  const half = Math.floor(next.length / 2)
  return [...next.slice(0, half).filter((_, i) => i % 2 === 0), ...next.slice(half)]
}

/** SVG path for a sparkline of the points inside a w×h box. */
export function sparklinePath(points: LossPoint[], width: number, height: number): string {
  if (points.length < 2) return ''
  const xs = points.map((p) => p.iteration)
  const ys = points.map((p) => p.loss)
  const [x0, x1] = [Math.min(...xs), Math.max(...xs)]
  const [y0, y1] = [Math.min(...ys), Math.max(...ys)]
  const sx = (x: number) => (x1 === x0 ? 0 : ((x - x0) / (x1 - x0)) * width)
  const sy = (y: number) => (y1 === y0 ? height / 2 : height - ((y - y0) / (y1 - y0)) * height)
  return points.map((p, i) => `${i ? 'L' : 'M'}${sx(p.iteration).toFixed(1)},${sy(p.loss).toFixed(1)}`).join(' ')
}
