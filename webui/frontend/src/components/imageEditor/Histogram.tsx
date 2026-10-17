import React, { useMemo } from 'react'

/** A 256-bin histogram as filled areas (luminance, or one channel each). */
export const Histogram: React.FC<{
  bins: { r: Uint32Array; g: Uint32Array; b: Uint32Array; l: Uint32Array } | null
  height?: number
  channels?: 'rgb' | 'l'
  className?: string
}> = ({ bins, height = 56, channels = 'rgb', className = '' }) => {
  const paths = useMemo(() => {
    if (!bins) return null
    const series = channels === 'rgb' ? [bins.r, bins.g, bins.b] : [bins.l]
    // Scaled to the 99.5th percentile, so one huge spike (a flat background)
    // doesn't flatten everything else.
    const all = series.flatMap((s) => Array.from(s)).sort((a, b) => a - b)
    const top = Math.max(1, all[Math.floor(all.length * 0.995)] || 1)
    return series.map((s) => {
      let d = `M0,${height}`
      for (let i = 0; i < 256; i++) d += ` L${i},${(height - Math.min(1, s[i] / top) * height).toFixed(1)}`
      return `${d} L255,${height} Z`
    })
  }, [bins, channels, height])
  const colors = channels === 'rgb' ? ['#ef4444', '#22c55e', '#3b82f6'] : ['#9ca3af']
  return (
    <svg viewBox={`0 0 255 ${height}`} preserveAspectRatio="none" className={`w-full block ${className}`} style={{ height }} aria-hidden data-testid="editor-histogram">
      {paths?.map((d, i) => <path key={i} d={d} fill={colors[i]} fillOpacity={channels === 'rgb' ? 0.35 : 0.5} style={{ mixBlendMode: 'screen' }} />)}
    </svg>
  )
}
