import React, { useEffect, useRef, useState } from 'react'
import { Loader2 } from 'lucide-react'
import {
  CLEARLY_DIFFERENT,
  comparisonSize,
  describeMeanDifference,
  diffHeatmap,
  heatGradientCss,
  type DiffStats,
} from '../lib/imageDiff'

function loadImage(src: string): Promise<HTMLImageElement> {
  return new Promise((resolve, reject) => {
    const img = new Image()
    img.onload = () => resolve(img)
    img.onerror = () => reject(new Error('image failed to load'))
    img.src = src
  })
}

function pixels(img: HTMLImageElement, width: number, height: number): Uint8ClampedArray {
  const c = document.createElement('canvas')
  c.width = width
  c.height = height
  const ctx = c.getContext('2d', { willReadFrequently: true })!
  ctx.drawImage(img, 0, 0, width, height)
  return ctx.getImageData(0, 0, width, height).data
}

/** The heatmap canvas for the zoom viewport, and the numbers for the legend. */
export function useDifferenceMap(originalSrc: string | null, resultSrc: string | null) {
  const canvasRef = useRef<HTMLCanvasElement | null>(null)
  const [state, setState] = useState<{ status: 'idle' | 'working' | 'ready' | 'error'; stats: DiffStats | null; version: number }>({
    status: 'idle',
    stats: null,
    version: 0,
  })

  useEffect(() => {
    if (!originalSrc || !resultSrc) {
      setState((s) => ({ ...s, status: 'idle', stats: null }))
      return
    }
    let cancelled = false
    setState((s) => ({ ...s, status: 'working' }))
    Promise.all([loadImage(originalSrc), loadImage(resultSrc)])
      .then(([orig, result]) => {
        if (cancelled) return
        // Compared at the result's size: that's the resolution the print
        // was solved at, and the original scales down to it cleanly.
        const { width, height } = comparisonSize(result.naturalWidth, result.naturalHeight)
        const { rgba, stats } = diffHeatmap(pixels(orig, width, height), pixels(result, width, height), width, height)
        const c = canvasRef.current ?? document.createElement('canvas')
        c.width = width
        c.height = height
        c.getContext('2d')!.putImageData(new ImageData(rgba, width, height), 0, 0)
        canvasRef.current = c
        setState((s) => ({ status: 'ready', stats, version: s.version + 1 }))
      })
      .catch(() => {
        if (!cancelled) setState((s) => ({ ...s, status: 'error', stats: null }))
      })
    return () => {
      cancelled = true
    }
  }, [originalSrc, resultSrc])

  return { canvas: canvasRef.current, ...state }
}

/** Draws the computed heatmap; sized like the other views' images. */
export const DifferenceImage: React.FC<{ canvas: HTMLCanvasElement | null; version: number }> = ({ canvas, version }) => {
  const ref = useRef<HTMLCanvasElement>(null)
  useEffect(() => {
    const el = ref.current
    if (!el || !canvas) return
    el.width = canvas.width
    el.height = canvas.height
    el.getContext('2d')!.drawImage(canvas, 0, 0)
  }, [canvas, version])
  return <canvas ref={ref} className="w-full h-full object-contain select-none [image-rendering:pixelated]" data-testid="difference-image" />
}

/** Scale and verdict under the heatmap. */
export const DifferenceLegend: React.FC<{ status: string; stats: DiffStats | null; onPaintFocus?: () => void }> = ({ status, stats, onPaintFocus }) => (
  <div className="flex flex-wrap items-center gap-x-4 gap-y-1 px-3 py-1.5 border-t border-gray-700 text-[11px] text-gray-400" data-testid="difference-legend">
    <div
      className="flex items-center gap-2"
      title={stats ? `Colors run from no difference to ΔE ${stats.scaleMax.toFixed(0)} (this picture's biggest differences) and above.` : undefined}
    >
      <span>Matches</span>
      <span className="w-24 h-2 rounded-full" style={{ background: heatGradientCss() }} aria-hidden />
      <span>Differs most</span>
    </div>
    {status === 'working' && (
      <span className="flex items-center gap-1">
        <Loader2 className="w-3 h-3 animate-spin" /> Comparing…
      </span>
    )}
    {status === 'error' && <span className="text-red-400">Couldn't compare the images.</span>}
    {status === 'ready' && stats && (
      <span
        className="text-gray-300"
        title={`Average color difference ΔE ${stats.mean.toFixed(1)}: below 2 is invisible, above ${CLEARLY_DIFFERENT} is plainly a different color.`}
        data-testid="difference-summary"
        data-mean={stats.mean.toFixed(2)}
      >
        <strong className="font-medium text-gray-100">{describeMeanDifference(stats.mean)}</strong>
        {' · '}
        {Math.round(stats.clearlyDifferent * 100)}% clearly different
      </span>
    )}
    {onPaintFocus && status === 'ready' && (
      <button onClick={onPaintFocus} className="ml-auto text-cyan-400 hover:text-cyan-300 hover:underline" data-testid="difference-paint-focus">
        Mark areas to improve…
      </button>
    )}
  </div>
)
