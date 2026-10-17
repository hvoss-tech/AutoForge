import React, { useMemo, useRef, useState } from 'react'
import { curveLut, IDENTITY_CURVE, type CurvePoint, type Curves } from '../../lib/imageAdjust'

type Channel = keyof Curves
const CHANNELS: { id: Channel; label: string; color: string }[] = [
  { id: 'master', label: 'RGB', color: '#e5e7eb' },
  { id: 'r', label: 'R', color: '#ef4444' },
  { id: 'g', label: 'G', color: '#22c55e' },
  { id: 'b', label: 'B', color: '#3b82f6' },
]
const SIZE = 255

/** Tone curves: click to add a point, drag it, double-click (or drag it out
 * of the box) to remove it. One curve for all channels, or each on its own. */
export const CurveEditor: React.FC<{
  curves: Curves
  histogram: Uint32Array | null
  onBegin: () => void
  onChange: (curves: Curves) => void
}> = ({ curves, histogram, onBegin, onChange }) => {
  const [channel, setChannel] = useState<Channel>('master')
  const svgRef = useRef<SVGSVGElement>(null)
  const drag = useRef<{ index: number } | null>(null)
  const points = curves[channel]
  const color = CHANNELS.find((c) => c.id === channel)!.color

  const path = useMemo(() => {
    const lut = curveLut(points)
    let d = ''
    for (let x = 0; x < 256; x++) d += `${x === 0 ? 'M' : 'L'}${x},${(SIZE - lut[x]).toFixed(1)} `
    return d
  }, [points])

  const histPath = useMemo(() => {
    if (!histogram) return null
    const sorted = Array.from(histogram).sort((a, b) => a - b)
    const top = Math.max(1, sorted[Math.floor(sorted.length * 0.995)] || 1)
    let d = `M0,${SIZE}`
    for (let i = 0; i < 256; i++) d += ` L${i},${(SIZE - Math.min(1, histogram[i] / top) * SIZE * 0.6).toFixed(1)}`
    return `${d} L255,${SIZE} Z`
  }, [histogram])

  const toCurve = (e: React.PointerEvent | PointerEvent): CurvePoint => {
    const rect = svgRef.current!.getBoundingClientRect()
    const x = ((e.clientX - rect.left) / rect.width) * SIZE
    const y = SIZE - ((e.clientY - rect.top) / rect.height) * SIZE
    return [Math.round(x), Math.round(y)]
  }

  const set = (next: CurvePoint[]) => onChange({ ...curves, [channel]: next })

  const onPointerDown = (e: React.PointerEvent) => {
    if (e.button !== 0) return
    e.preventDefault()
    const [x, y] = toCurve(e)
    onBegin()
    // An existing point near the click, else a new one on the click.
    let index = points.findIndex((p) => Math.abs(p[0] - x) <= 8 && Math.abs(p[1] - y) <= 12)
    let next = points.map((p) => [...p] as CurvePoint)
    if (index < 0) {
      next.push([Math.max(0, Math.min(255, x)), Math.max(0, Math.min(255, y))])
      next.sort((a, b) => a[0] - b[0])
      index = next.findIndex((p) => p[0] === Math.max(0, Math.min(255, x)))
      set(next)
    }
    drag.current = { index }
    ;(e.currentTarget as Element).setPointerCapture(e.pointerId)
  }

  const onPointerMove = (e: React.PointerEvent) => {
    const d = drag.current
    if (!d) return
    const [x, y] = toCurve(e)
    const next = points.map((p) => [...p] as CurvePoint)
    const last = next.length - 1
    const ends = d.index === 0 || d.index === last
    // Dragged out of the box: removed (the two ends stay).
    if (!ends && (y < -20 || y > SIZE + 20)) {
      next.splice(d.index, 1)
      drag.current = null
      set(next)
      return
    }
    const lo = d.index === 0 ? 0 : next[d.index - 1][0] + 1
    const hi = d.index === last ? 255 : next[d.index + 1][0] - 1
    next[d.index] = [Math.max(lo, Math.min(hi, x)), Math.max(0, Math.min(255, y))]
    set(next)
  }

  const onDoubleClick = (e: React.MouseEvent) => {
    const rect = svgRef.current!.getBoundingClientRect()
    const x = ((e.clientX - rect.left) / rect.width) * SIZE
    const y = SIZE - ((e.clientY - rect.top) / rect.height) * SIZE
    const index = points.findIndex((p) => Math.abs(p[0] - x) <= 8 && Math.abs(p[1] - y) <= 12)
    if (index > 0 && index < points.length - 1) {
      onBegin()
      set(points.filter((_, i) => i !== index))
    }
  }

  return (
    <div className="space-y-1.5" data-testid="curve-editor">
      <div className="flex items-center gap-1">
        {CHANNELS.map((c) => (
          <button
            key={c.id}
            onClick={() => setChannel(c.id)}
            aria-pressed={channel === c.id}
            className={`px-2 py-0.5 rounded text-[11px] font-medium ${channel === c.id ? 'bg-gray-600 text-gray-100' : 'text-gray-400 hover:bg-gray-700'}`}
            data-testid={`curve-channel-${c.id}`}
          >
            <span style={{ color: c.id === 'master' ? undefined : c.color }}>{c.label}</span>
          </button>
        ))}
        <button
          onClick={() => {
            onBegin()
            set(IDENTITY_CURVE.map((p) => [...p] as CurvePoint))
          }}
          className="ml-auto text-[11px] text-gray-400 hover:text-gray-200"
          data-testid="curve-reset"
        >
          Reset
        </button>
      </div>
      <svg
        ref={svgRef}
        viewBox={`-4 -4 ${SIZE + 8} ${SIZE + 8}`}
        className="w-full aspect-square bg-gray-950 rounded border border-gray-700 cursor-crosshair touch-none select-none"
        onPointerDown={onPointerDown}
        onPointerMove={onPointerMove}
        onPointerUp={() => (drag.current = null)}
        onPointerCancel={() => (drag.current = null)}
        onDoubleClick={onDoubleClick}
        data-testid="curve-canvas"
      >
        {histPath && <path d={histPath} fill="#6b7280" fillOpacity={0.25} />}
        {[64, 128, 192].map((v) => (
          <g key={v} stroke="#374151" strokeWidth={0.75}>
            <line x1={v} y1={0} x2={v} y2={SIZE} />
            <line x1={0} y1={v} x2={SIZE} y2={v} />
          </g>
        ))}
        <line x1={0} y1={SIZE} x2={SIZE} y2={0} stroke="#4b5563" strokeWidth={0.75} strokeDasharray="3 3" />
        <path d={path} fill="none" stroke={color} strokeWidth={2} />
        {points.map((p, i) => (
          <circle key={i} cx={p[0]} cy={SIZE - p[1]} r={5} fill="#111827" stroke={color} strokeWidth={2} data-testid="curve-point" />
        ))}
      </svg>
      <p className="text-[11px] text-gray-500">Click to add a point, drag to shape the curve, double-click a point to remove it.</p>
    </div>
  )
}
