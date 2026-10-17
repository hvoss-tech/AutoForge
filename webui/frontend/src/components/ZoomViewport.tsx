import React, { useEffect, useRef } from 'react'
import { IDENTITY_ZOOM, zoomAt, type ZoomTransform } from '../lib/layout'

/** Wheel to zoom (about the cursor), drag to pan, double-click to reset.
 * Children position themselves with `transformStyle(zoom)`; viewports given
 * the same zoom stay in lockstep, so both images of a comparison always show
 * the same spot. */
export const ZoomViewport: React.FC<{
  zoom: ZoomTransform
  setZoom: React.Dispatch<React.SetStateAction<ZoomTransform>>
  children: React.ReactNode
  testId?: string
  /** Left button paints (the focus-area brush): pan with the middle or
   * right button instead, and don't reset on double-click. */
  paintMode?: boolean
}> = ({ zoom, setZoom, children, testId, paintMode = false }) => {
  const ref = useRef<HTMLDivElement>(null)
  const pan = useRef<{ x: number; y: number; tx: number; ty: number } | null>(null)

  useEffect(() => {
    const el = ref.current
    if (!el) return
    // React's onWheel is passive, so it can't stop the page from scrolling.
    const onWheel = (e: WheelEvent) => {
      e.preventDefault()
      const rect = el.getBoundingClientRect()
      setZoom((t) => zoomAt(t, e.deltaY < 0 ? 1.2 : 1 / 1.2, e.clientX - rect.left, e.clientY - rect.top))
    }
    el.addEventListener('wheel', onWheel, { passive: false })
    return () => el.removeEventListener('wheel', onWheel)
  }, [setZoom])

  return (
    <div
      ref={ref}
      className={`relative w-full h-full overflow-hidden ${paintMode ? '' : zoom.scale > 1 ? 'cursor-grab active:cursor-grabbing' : 'cursor-zoom-in'}`}
      onContextMenu={paintMode ? (e) => e.preventDefault() : undefined}
      onPointerDown={(e) => {
        if (paintMode ? e.button !== 1 && e.button !== 2 : e.button !== 0) return
        if (zoom.scale === 1) return
        e.currentTarget.setPointerCapture(e.pointerId)
        pan.current = { x: e.clientX, y: e.clientY, tx: zoom.x, ty: zoom.y }
      }}
      onPointerMove={(e) => {
        const p = pan.current
        if (p) setZoom((t) => ({ ...t, x: p.tx + e.clientX - p.x, y: p.ty + e.clientY - p.y }))
      }}
      onPointerUp={() => {
        pan.current = null
      }}
      onDoubleClick={paintMode ? undefined : () => setZoom(IDENTITY_ZOOM)}
      data-testid={testId}
    >
      {children}
    </div>
  )
}

export const transformStyle = (t: ZoomTransform): React.CSSProperties => ({
  transform: `translate(${t.x}px, ${t.y}px) scale(${t.scale})`,
  transformOrigin: '0 0',
})
