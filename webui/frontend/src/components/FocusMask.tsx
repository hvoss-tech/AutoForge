import React, { useCallback, useEffect, useRef, useState } from 'react'
import { useAppStore } from '../store/appStore'
import { describeApiError } from '../lib/apiError'
import {
  alphaToGreyscale,
  brushDiameter,
  clientToMask,
  greyscaleToAlpha,
  maskCanvasSize,
  maskHasPaint,
  paintedShare,
} from '../lib/focusMask'

export type FocusTool = 'paint' | 'erase'

/** Wait after the last stroke before uploading, so a burst of strokes is
 * one upload (and one undo step), not one per stroke. */
const SAVE_DELAY = 700

export interface FocusMaskState {
  /** The painted mask: white strokes, alpha = how fully a pixel is painted. */
  canvas: HTMLCanvasElement | null
  width: number
  height: number
  /** Bumped on every change, to redraw the overlay. */
  version: number
  /** Share of the picture painted (0–1). */
  share: number
  saving: boolean
  /** Redraw after a change mid-stroke (cheap; nothing is counted or saved). */
  touch: () => void
  /** A stroke ended: recount the painted share and schedule the upload. */
  markChanged: () => void
  clear: () => void
}

/** The focus-area mask for the current picture: loads the saved one
 * (a restored session, undo/redo) and uploads edits as a greyscale PNG
 * the optimizer reads as its priority mask. */
export function useFocusMask(): FocusMaskState {
  const inputImage = useAppStore((s) => s.inputImage)
  const savedName = useAppStore((s) => s.settings.priority_mask || '')
  const setPriorityMask = useAppStore((s) => s.setPriorityMask)
  const pushToast = useAppStore((s) => s.pushToast)

  const canvasRef = useRef<HTMLCanvasElement | null>(null)
  const [size, setSize] = useState({ width: 0, height: 0 })
  const [version, setVersion] = useState(0)
  const [share, setShare] = useState(0)
  const [saving, setSaving] = useState(false)
  // The mask file the canvas currently shows (what we last saved or loaded).
  const shownNameRef = useRef<string | null>(null)
  const saveTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null)
  const saveSeqRef = useRef(0)

  const refreshShare = useCallback(() => {
    const c = canvasRef.current
    if (!c) return setShare(0)
    const ctx = c.getContext('2d', { willReadFrequently: true })
    if (!ctx) return
    setShare(paintedShare(ctx.getImageData(0, 0, c.width, c.height).data))
  }, [])

  // A canvas the size of the picture (capped), fresh for every picture.
  useEffect(() => {
    if (saveTimerRef.current) clearTimeout(saveTimerRef.current)
    saveTimerRef.current = null
    // An upload still in flight belongs to the previous picture.
    saveSeqRef.current++
    canvasRef.current = null
    shownNameRef.current = null
    setSize({ width: 0, height: 0 })
    setShare(0)
    if (!inputImage) return
    let cancelled = false
    const img = new Image()
    img.onload = () => {
      if (cancelled) return
      const { width, height } = maskCanvasSize(img.naturalWidth, img.naturalHeight)
      const c = document.createElement('canvas')
      c.width = width
      c.height = height
      canvasRef.current = c
      setSize({ width, height })
      setVersion((v) => v + 1)
    }
    img.src = inputImage
    return () => {
      cancelled = true
    }
  }, [inputImage])

  // Show the saved mask whenever it isn't the one on the canvas already —
  // on load, and when undo/redo switches to another one.
  useEffect(() => {
    const c = canvasRef.current
    if (!c || shownNameRef.current === savedName) return
    shownNameRef.current = savedName
    const ctx = c.getContext('2d', { willReadFrequently: true })
    if (!ctx) return
    ctx.clearRect(0, 0, c.width, c.height)
    if (!savedName) {
      setShare(0)
      setVersion((v) => v + 1)
      return
    }
    let cancelled = false
    const img = new Image()
    img.onload = () => {
      if (cancelled || shownNameRef.current !== savedName) return
      ctx.clearRect(0, 0, c.width, c.height)
      ctx.drawImage(img, 0, 0, c.width, c.height)
      const data = ctx.getImageData(0, 0, c.width, c.height)
      greyscaleToAlpha(data.data)
      ctx.putImageData(data, 0, 0)
      refreshShare()
      setVersion((v) => v + 1)
    }
    img.onerror = () => {
      if (!cancelled) pushToast('The saved focus areas could not be loaded. Paint them again, or clear them.', 'warning')
    }
    img.src = `/uploads/${encodeURIComponent(savedName)}`
    return () => {
      cancelled = true
    }
  }, [savedName, size, pushToast, refreshShare])

  const save = useCallback(async () => {
    const c = canvasRef.current
    if (!c) return
    const seq = ++saveSeqRef.current
    const ctx = c.getContext('2d', { willReadFrequently: true })
    if (!ctx) return
    const data = ctx.getImageData(0, 0, c.width, c.height)
    if (!maskHasPaint(data.data)) {
      shownNameRef.current = ''
      setPriorityMask('')
      return
    }
    // The backend reads greyscale and ignores alpha: flatten onto black.
    const out = document.createElement('canvas')
    out.width = c.width
    out.height = c.height
    out.getContext('2d')!.putImageData(new ImageData(alphaToGreyscale(data.data), c.width, c.height), 0, 0)
    const blob = await new Promise<Blob | null>((resolve) => out.toBlob(resolve, 'image/png'))
    if (!blob || seq !== saveSeqRef.current) return
    setSaving(true)
    try {
      const form = new FormData()
      form.append('file', blob, 'focus-areas.png')
      const response = await fetch('/api/images/upload', { method: 'POST', body: form })
      if (!response.ok) throw new Error(describeApiError(await response.json().catch(() => null), response.status))
      const { filename } = await response.json()
      // A later save (more strokes) supersedes this one.
      if (seq !== saveSeqRef.current || !filename) return
      shownNameRef.current = filename
      setPriorityMask(filename)
    } catch (e) {
      pushToast(`Couldn't save the focus areas: ${e instanceof Error ? e.message : String(e)}`)
    } finally {
      if (seq === saveSeqRef.current) setSaving(false)
    }
  }, [setPriorityMask, pushToast])

  const markChanged = useCallback(() => {
    setVersion((v) => v + 1)
    refreshShare()
    if (saveTimerRef.current) clearTimeout(saveTimerRef.current)
    saveTimerRef.current = setTimeout(() => {
      saveTimerRef.current = null
      save()
    }, SAVE_DELAY)
  }, [save, refreshShare])

  const touch = useCallback(() => setVersion((v) => v + 1), [])

  const clear = useCallback(() => {
    const c = canvasRef.current
    if (c) c.getContext('2d')?.clearRect(0, 0, c.width, c.height)
    if (saveTimerRef.current) clearTimeout(saveTimerRef.current)
    saveTimerRef.current = null
    saveSeqRef.current++
    shownNameRef.current = ''
    setShare(0)
    setVersion((v) => v + 1)
    setPriorityMask('')
  }, [setPriorityMask])

  // An edit still waiting for its upload when the panel goes away is saved now.
  useEffect(
    () => () => {
      if (saveTimerRef.current) {
        clearTimeout(saveTimerRef.current)
        save()
      }
    },
    [save],
  )

  return { canvas: canvasRef.current, width: size.width, height: size.height, version, share, saving, touch, markChanged, clear }
}

/** Where the brush is, in mask pixels, and how big a screen pixel is there. */
interface Cursor {
  x: number
  y: number
  scale: number
}

/** Draws the mask over the picture — the painted parts lit, the rest dimmed,
 * like a spotlight — and, while `painting`, turns the pointer into a brush.
 * Sits in the zoom viewport's transformed layer, with the same object-fit
 * as the picture, so it lines up at any zoom. Two canvases: the mask view
 * only redraws when the mask changes, the brush outline on every move. */
export const FocusMaskOverlay: React.FC<{
  mask: FocusMaskState
  painting: boolean
  tool: FocusTool
  brushPercent: number
}> = ({ mask, painting, tool, brushPercent }) => {
  const viewRef = useRef<HTMLCanvasElement>(null)
  const cursorRef = useRef<HTMLCanvasElement>(null)
  const tintRef = useRef<HTMLCanvasElement | null>(null)
  const strokeRef = useRef<{ x: number; y: number } | null>(null)
  const [cursor, setCursor] = useState<Cursor | null>(null)
  // Something was painted in the stroke under way (the painted share is
  // only recounted when a stroke ends).
  const [strokePainted, setStrokePainted] = useState(false)
  const { canvas, width, height, version, share } = mask
  const diameter = brushDiameter(brushPercent, width, height)
  const lit = share > 0 || strokePainted

  // Dim everything, cut the dimming away where painted, and tint the painted
  // parts so their edges stay visible on light pictures.
  useEffect(() => {
    const view = viewRef.current
    if (!view || !canvas) return
    const ctx = view.getContext('2d')
    if (!ctx) return
    ctx.clearRect(0, 0, view.width, view.height)
    if (!lit) return
    ctx.fillStyle = 'rgba(8, 12, 20, 0.62)'
    ctx.fillRect(0, 0, view.width, view.height)
    ctx.globalCompositeOperation = 'destination-out'
    ctx.drawImage(canvas, 0, 0)
    ctx.globalCompositeOperation = 'source-over'
    let tint = tintRef.current
    if (!tint || tint.width !== view.width || tint.height !== view.height) {
      tint = document.createElement('canvas')
      tint.width = view.width
      tint.height = view.height
      tintRef.current = tint
    }
    const tctx = tint.getContext('2d')!
    tctx.globalCompositeOperation = 'source-over'
    tctx.clearRect(0, 0, tint.width, tint.height)
    tctx.drawImage(canvas, 0, 0)
    tctx.globalCompositeOperation = 'source-in'
    tctx.fillStyle = 'rgb(34, 211, 238)'
    tctx.fillRect(0, 0, tint.width, tint.height)
    ctx.globalAlpha = 0.22
    ctx.drawImage(tint, 0, 0)
    ctx.globalAlpha = 1
  }, [canvas, version, lit])

  // The brush outline: dark halo under a light ring (dashed orange for the
  // eraser), a steady 1.5 screen pixels wide at any zoom.
  useEffect(() => {
    const el = cursorRef.current
    const ctx = el?.getContext('2d')
    if (!el || !ctx) return
    ctx.clearRect(0, 0, el.width, el.height)
    if (!painting || !cursor) return
    const px = 1 / cursor.scale
    ctx.beginPath()
    ctx.arc(cursor.x, cursor.y, diameter / 2, 0, Math.PI * 2)
    ctx.lineWidth = 3 * px
    ctx.strokeStyle = 'rgba(0, 0, 0, 0.7)'
    ctx.stroke()
    ctx.lineWidth = 1.5 * px
    ctx.strokeStyle = tool === 'erase' ? 'rgb(251, 146, 60)' : 'rgb(255, 255, 255)'
    if (tool === 'erase') ctx.setLineDash([4 * px, 3 * px])
    ctx.stroke()
    ctx.setLineDash([])
  }, [painting, cursor, diameter, tool])

  const toMask = (e: React.PointerEvent) => {
    const el = cursorRef.current
    if (!el) return null
    return clientToMask(e.clientX, e.clientY, el.getBoundingClientRect(), width, height)
  }

  const drawTo = (x: number, y: number) => {
    const ctx = canvas?.getContext('2d')
    if (!ctx) return
    ctx.globalCompositeOperation = tool === 'erase' ? 'destination-out' : 'source-over'
    ctx.strokeStyle = '#ffffff'
    ctx.fillStyle = '#ffffff'
    ctx.lineCap = 'round'
    ctx.lineJoin = 'round'
    ctx.lineWidth = diameter
    const from = strokeRef.current
    ctx.beginPath()
    if (from) {
      ctx.moveTo(from.x, from.y)
      ctx.lineTo(x, y)
      ctx.stroke()
    } else {
      ctx.arc(x, y, diameter / 2, 0, Math.PI * 2)
      ctx.fill()
    }
    ctx.globalCompositeOperation = 'source-over'
    strokeRef.current = { x, y }
    if (tool === 'paint') setStrokePainted(true)
    mask.touch()
  }

  const endStroke = () => {
    if (!strokeRef.current) return
    strokeRef.current = null
    setStrokePainted(false)
    mask.markChanged()
  }

  if (!canvas || !width) return null

  const layer = 'absolute inset-0 w-full h-full object-contain'
  return (
    <>
      <canvas ref={viewRef} width={width} height={height} className={`${layer} pointer-events-none`} data-testid="focus-mask-overlay" data-painted-share={share.toFixed(3)} />
      {painting && (
        <canvas
          ref={cursorRef}
          width={width}
          height={height}
          className={`${layer} cursor-crosshair`}
          style={{ touchAction: 'none' }}
          data-testid="focus-mask-canvas"
          onPointerDown={(e) => {
            if (e.button !== 0) return
            const p = toMask(e)
            if (!p) return
            e.stopPropagation()
            e.currentTarget.setPointerCapture(e.pointerId)
            strokeRef.current = null
            drawTo(p.x, p.y)
            setCursor(p)
          }}
          onPointerMove={(e) => {
            const p = toMask(e)
            if (!p) return
            setCursor(p)
            if (strokeRef.current && e.buttons & 1) drawTo(p.x, p.y)
          }}
          onPointerUp={endStroke}
          onPointerCancel={endStroke}
          onPointerLeave={() => {
            if (!strokeRef.current) setCursor(null)
          }}
        />
      )}
    </>
  )
}
