import React, { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { createPortal } from 'react-dom'
import { Minus, Plus } from 'lucide-react'
import { useAppStore } from '../store/appStore'
import type { Filament } from '../types'
import { filterActiveHandles } from '../lib/colorStack'
import { getPlanBands } from '../lib/printPlan'

interface HandleData {
  storeIndex: number
  value: number
  color: string
  label: string
  td: number
  filamentUuid: string
}

interface SegmentData {
  layerIndex: number
  color: string
  owningHandleIndex: number
  isEquivalent: boolean
}

function lerpColor(c1: string, c2: string, t: number): string {
  const r1 = parseInt(c1.slice(1, 3), 16)
  const g1 = parseInt(c1.slice(3, 5), 16)
  const b1 = parseInt(c1.slice(5, 7), 16)
  const r2 = parseInt(c2.slice(1, 3), 16)
  const g2 = parseInt(c2.slice(3, 5), 16)
  const b2 = parseInt(c2.slice(5, 7), 16)
  const r = Math.round(r1 + (r2 - r1) * t)
  const g = Math.round(g1 + (g2 - g1) * t)
  const b = Math.round(b1 + (b2 - b1) * t)
  return `#${r.toString(16).padStart(2, '0')}${g.toString(16).padStart(2, '0')}${b.toString(16).padStart(2, '0')}`
}

/** Black or white, whichever reads better on this background (the label
 * was always white, unreadable on beige/white filaments). */
function readableTextColor(hex: string): string {
  const n = parseInt(hex.replace('#', ''), 16)
  if (Number.isNaN(n)) return '#ffffff'
  const r = (n >> 16) & 255, g = (n >> 8) & 255, b = n & 255
  return 0.299 * r + 0.587 * g + 0.114 * b > 150 ? '#111111' : '#ffffff'
}

const MIN_LABEL_SPACING = 16
const MIN_ZOOM = 1
const MAX_ZOOM = 8

function hexColorsEqual(c1: string, c2: string): boolean {
  return c1.toLowerCase() === c2.toLowerCase()
}

export const ColorCore: React.FC = () => {
  const colorSliders = useAppStore((s) => s.colorSliders)
  const sliderLayerRange = useAppStore((s) => s.sliderLayerRange)
  const filaments = useAppStore((s) => s.filaments)
  const updateSlider = useAppStore((s) => s.updateSlider)
  const addActiveFilament = useAppStore((s) => s.addActiveFilament)
  const activeFilaments = useAppStore((s) => s.activeFilaments)
  const settings = useAppStore((s) => s.settings)
  const selectedBand = useAppStore((s) => s.selectedBand)
  const hoveredBand = useAppStore((s) => s.hoveredBand)
  const setSelectedBand = useAppStore((s) => s.setSelectedBand)
  const setHoveredBand = useAppStore((s) => s.setHoveredBand)
  const setBottomTab = useAppStore((s) => s.setBottomTab)
  // The scrolling viewport (data-testid="color-core"); `contentRef` is the
  // stack inside it, taller than the viewport when zoomed in.
  const containerRef = useRef<HTMLDivElement>(null)
  const contentRef = useRef<HTMLDivElement>(null)
  const trackRef = useRef<HTMLDivElement>(null)
  const handleRefs = useRef(new Map<number, HTMLDivElement>())
  const [focusedHandle, setFocusedHandle] = useState<number | null>(null)
  const [draggingHandle, setDraggingHandle] = useState<number | null>(null)
  // The stable identity of the band being dragged. `draggingHandle` is an
  // index into the layer-sorted `handles` array, which can reorder mid-drag
  // (dragging a band onto the same layer as a neighbor ties them, and the
  // stable sort then orders the pair by storeIndex) — tracking storeIndex
  // instead means the drag always follows the same band even if its index
  // in that array shifts underneath it.
  const draggingStoreIndexRef = useRef<number | null>(null)
  // The drag track's vertical scale, frozen for the duration of one drag —
  // see the effect below.
  const dragTrackTopLayerRef = useRef<number | null>(null)
  const [hoveredHandle, setHoveredHandle] = useState<number | null>(null)
  const [viewportHeight, setViewportHeight] = useState(300)
  const [zoom, setZoom] = useState(1)
  const containerHeight = viewportHeight * zoom

  useEffect(() => {
    if (!containerRef.current) return
    const updateHeight = () => {
      if (containerRef.current) {
        setViewportHeight(containerRef.current.clientHeight)
      }
    }
    updateHeight()
    const observer = new ResizeObserver(updateHeight)
    observer.observe(containerRef.current)
    return () => observer.disconnect()
  }, [])

  // Ctrl+wheel zooms the column (plain wheel scrolls it once zoomed in).
  useEffect(() => {
    const el = containerRef.current
    if (!el) return
    const onWheel = (e: WheelEvent) => {
      if (!e.ctrlKey && !e.metaKey) return
      e.preventDefault()
      setZoom((z) => Math.min(MAX_ZOOM, Math.max(MIN_ZOOM, e.deltaY < 0 ? z * 1.25 : z / 1.25)))
    }
    el.addEventListener('wheel', onWheel, { passive: false })
    return () => el.removeEventListener('wheel', onWheel)
  }, [])

  const getFilamentColor = useCallback(
    (uuid: string) => {
      if (!uuid) return '#555555'
      const f = filaments.find((f) => f.uuid === uuid)
      return f ? f.color : '#555555'
    },
    [filaments],
  )

  const handles: HandleData[] = useMemo(() => {
    const active = colorSliders
      .map((slider, idx) => ({ ...slider, storeIndex: idx }))
      .filter((s) => s.enabled && s.layer > 0)
      .sort((a, b) => a.layer - b.layer)

    return active.map((slider, idx) => ({
      storeIndex: slider.storeIndex,
      value: slider.layer,
      color: getFilamentColor(slider.filament_uuid),
      label: `L${slider.layer}`,
      td: slider.td,
      filamentUuid: slider.filament_uuid,
    }))
  }, [colorSliders, getFilamentColor])

  // Sliders are now allowed to share a layer (dragged on top of one
  // another). Only one handle can actually own a given layer for coloring
  // purposes — the one furthest right (highest storeIndex) wins; the rest
  // are dimmed in the handle list and excluded here so segment colors never
  // depend on more than one owner per layer.
  const activeHandles = useMemo(() => filterActiveHandles(handles), [handles])
  const overlapDisabled = useMemo(() => {
    const activeIndices = new Set(activeHandles.map((h) => h.storeIndex))
    return new Set(handles.filter((h) => !activeIndices.has(h.storeIndex)).map((h) => h.storeIndex))
  }, [handles, activeHandles])

  const segments: SegmentData[] = useMemo(() => {
    const handles = activeHandles
    if (handles.length === 0) return []

    const maxLayer = handles[handles.length - 1].value
    const result: SegmentData[] = []

    // Rendered top-to-bottom in DOM order, so this must walk from the
    // highest layer down to 1 — the handles column places the highest
    // layer at the top (`layerToY` mirrors a physical print: layer 1 at
    // the bottom), and the segment track has to agree with it. Iterating
    // ascending here put layer 1 at the top of the DOM instead, so a
    // handle dragged to a high layer highlighted a band near the *bottom*
    // of the track instead of near its own arrow.
    for (let layer = maxLayer; layer >= 1; layer--) {
      let owningHandleIdx = handles.length - 1
      for (let h = 0; h < handles.length; h++) {
        if (layer <= handles[h].value) {
          owningHandleIdx = h
          break
        }
      }

      let color: string
      if (owningHandleIdx === 0) {
        color = handles[0].color
      } else {
        const prevHandle = handles[owningHandleIdx - 1]
        const currHandle = handles[owningHandleIdx]
        const rangeStart = prevHandle.value
        const rangeEnd = currHandle.value
        const t = rangeEnd === rangeStart ? 0 : (layer - rangeStart) / (rangeEnd - rangeStart)
        color = lerpColor(prevHandle.color, currHandle.color, t)
      }

      let isEquivalent = false
      if (layer > 1) {
        const prevOwningHandleIdx = (() => {
          for (let h = 0; h < handles.length; h++) {
            if (layer - 1 <= handles[h].value) {
              return h
            }
          }
          return handles.length - 1
        })()

        let prevColor: string
        if (prevOwningHandleIdx === 0) {
          prevColor = handles[0].color
        } else {
          const prevH = handles[prevOwningHandleIdx - 1]
          const currH = handles[prevOwningHandleIdx]
          const rangeStart = prevH.value
          const rangeEnd = currH.value
          const t = rangeEnd === rangeStart ? 0 : (layer - 1 - rangeStart) / (rangeEnd - rangeStart)
          prevColor = lerpColor(prevH.color, currH.color, t)
        }

        if (hexColorsEqual(color, prevColor)) {
          isEquivalent = true
        }
      }

      result.push({
        layerIndex: layer,
        color,
        owningHandleIndex: owningHandleIdx,
        isEquivalent,
      })
    }

    return result
  }, [activeHandles])

  // While dragging, the track's vertical scale is frozen (dragTrackTopLayerRef)
  // to avoid the "runaway" feedback loop documented on that ref. Rendering
  // handle/segment positions off the *live* top layer instead — as before —
  // made the topmost handle visibly detach from the cursor mid-drag (its own
  // movement kept rescaling the track under it) and reflowed every other
  // handle even though their layers hadn't changed. Rendering off the same
  // frozen scale keeps the whole track visually stable for the drag's
  // duration; it re-syncs to the live top layer as soon as the drag ends.
  const displayMaxLayer =
    draggingHandle !== null && dragTrackTopLayerRef.current !== null
      ? dragTrackTopLayerRef.current
      : handles.length > 0
        ? handles[handles.length - 1].value
        : 0

  const layerToY = useCallback(
    (layer: number): number => {
      if (handles.length === 0 || displayMaxLayer === 0) return 0
      const segmentHeight = containerHeight / displayMaxLayer
      return segmentHeight * (displayMaxLayer - layer) + segmentHeight / 2
    },
    [handles, containerHeight, displayMaxLayer],
  )

  /** Pointing at a handle selects its band everywhere (row, strip, 3D). */
  const selectHandle = useCallback(
    (handleIndex: number) => {
      setFocusedHandle(handleIndex)
      const handle = handles[handleIndex]
      if (handle) {
        setSelectedBand(handle.storeIndex)
        setBottomTab('layers')
      }
    },
    [handles, setSelectedBand, setBottomTab],
  )

  const handleMouseDown = useCallback(
    (handleIndex: number) => (e: React.MouseEvent) => {
      e.preventDefault()
      handleRefs.current.get(handleIndex)?.focus({ preventScroll: true })
      selectHandle(handleIndex)
      draggingStoreIndexRef.current = handles[handleIndex]?.storeIndex ?? null
      setDraggingHandle(handleIndex)
    },
    [selectHandle, handles],
  )

  const handleTouchStart = useCallback(
    (handleIndex: number) => () => {
      selectHandle(handleIndex)
      draggingStoreIndexRef.current = handles[handleIndex]?.storeIndex ?? null
      setDraggingHandle(handleIndex)
    },
    [selectHandle, handles],
  )

  /** Arrow keys move a handle one layer (Shift: five), within the same
   * bounds as dragging it. */
  const handleKeyDown = useCallback(
    (handleIndex: number) => (e: React.KeyboardEvent) => {
      const step = e.key === 'ArrowUp' ? 1 : e.key === 'ArrowDown' ? -1 : e.key === 'PageUp' ? 5 : e.key === 'PageDown' ? -5 : 0
      if (!step) return
      e.preventDefault()
      const handle = handles[handleIndex]
      if (!handle) return
      const min = handleIndex > 0 ? handles[handleIndex - 1].value : 1
      const max = handleIndex < handles.length - 1 ? handles[handleIndex + 1].value : sliderLayerRange.max
      const next = Math.max(min, Math.min(max, handle.value + step * (e.shiftKey ? 5 : 1)))
      if (next !== handle.value) updateSlider(handle.storeIndex, { layer: next })
    },
    [handles, sliderLayerRange.max, updateSlider],
  )

  // Tooltip content per band: filament, layer range, TD.
  const bandInfo = useMemo(() => {
    const byUuid = new Map<string, Filament>()
    for (const f of [...filaments, ...activeFilaments]) byUuid.set(f.uuid, f)
    const info = new Map<number, { name: string; start: number; end: number; td: number }>()
    for (const b of getPlanBands(colorSliders, [], settings)) {
      const f = byUuid.get(b.filamentUuid)
      info.set(b.storeIndex, { name: f ? `${f.brand ? `${f.brand} · ` : ''}${f.name}` : 'Unassigned', start: b.startLayer, end: b.endLayer, td: b.td })
    }
    return info
  }, [colorSliders, filaments, activeFilaments, settings])

  // Keep the selected handle visible when zoomed in.
  useEffect(() => {
    if (selectedBand === null || zoom === 1 || draggingHandle !== null) return
    const idx = handles.findIndex((h) => h.storeIndex === selectedBand)
    handleRefs.current.get(idx)?.scrollIntoView({ block: 'nearest' })
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [selectedBand, zoom])

  useEffect(() => {
    if (draggingHandle === null) {
      // Reset so the next drag captures a fresh value at its own start.
      dragTrackTopLayerRef.current = null
      draggingStoreIndexRef.current = null
      return
    }

    const activeSliders = colorSliders
      .map((s, idx) => ({ ...s, storeIndex: idx }))
      .filter((s) => s.enabled && s.layer > 0)
      .sort((a, b) => a.layer - b.layer)

    // Re-derive the dragged band's position by its stable storeIndex, not
    // by trusting the `draggingHandle` array index — a tie with a neighbor
    // reorders `activeSliders` underneath an in-progress drag.
    const dragIdx = activeSliders.findIndex((s) => s.storeIndex === draggingStoreIndexRef.current)
    if (dragIdx === -1) return
    if (dragIdx !== draggingHandle) setDraggingHandle(dragIdx)

    const handle = activeSliders[dragIdx]
    if (!handle) return

    const prevHandle = dragIdx > 0 ? activeSliders[dragIdx - 1] : null
    const nextHandle = dragIdx < activeSliders.length - 1 ? activeSliders[dragIdx + 1] : null

    // Sliders may now be dragged onto the exact same layer as a neighbor
    // (that's how "move over one another" works) — bounds are inclusive of
    // the neighbor's own layer instead of stopping one short of it. This
    // also fixes the previous exclusive bounds going invalid (min > max)
    // whenever two neighbors were already only 1 layer apart.
    const minLayer = prevHandle ? prevHandle.layer : 1
    const maxLayer = nextHandle ? nextHandle.layer : sliderLayerRange.max

    // The vertical scale of the drag track is fixed once, at the start of
    // this drag — not recomputed from the live `colorSliders` on every one
    // of this effect's re-runs (which `updateSlider` below triggers, since
    // colorSliders is a dependency). For every handle but the topmost, that
    // recomputed value happens to equal the frozen one anyway (a neighbor's
    // layer doesn't move during this drag). For the topmost handle it does
    // NOT: `activeSliders[last]` *is* the handle being dragged, so its
    // "scale" changed on the previous update — shrinking the denominator as
    // the handle rose, which made the next mousemove event (same pixel
    // delta) compute a larger layer jump than the last one, and the handle
    // ran away toward the layer max within a few pixels of jitter.
    if (dragTrackTopLayerRef.current === null) {
      dragTrackTopLayerRef.current = activeSliders[activeSliders.length - 1].layer
    }
    const currentMaxLayer = dragTrackTopLayerRef.current

    const handleMove = (clientY: number) => {
      if (!contentRef.current) return
      const rect = contentRef.current.getBoundingClientRect()
      const relativeY = clientY - rect.top
      const segmentHeight = rect.height / currentMaxLayer
      const rawLayer = Math.round((rect.height - relativeY) / segmentHeight)
      const clampedLayer = Math.max(minLayer, Math.min(maxLayer, Math.max(1, rawLayer)))
      if (clampedLayer !== handle.layer) {
        updateSlider(handle.storeIndex, { layer: clampedLayer })
      }
    }

    const onMouseMove = (e: MouseEvent) => {
      handleMove(e.clientY)
    }

    const onTouchMove = (e: TouchEvent) => {
      if (e.touches.length > 0) {
        handleMove(e.touches[0].clientY)
      }
    }

    const onEnd = () => {
      setDraggingHandle(null)
    }

    window.addEventListener('mousemove', onMouseMove)
    window.addEventListener('mouseup', onEnd)
    window.addEventListener('touchmove', onTouchMove)
    window.addEventListener('touchend', onEnd)

    return () => {
      window.removeEventListener('mousemove', onMouseMove)
      window.removeEventListener('mouseup', onEnd)
      window.removeEventListener('touchmove', onTouchMove)
      window.removeEventListener('touchend', onEnd)
    }
  }, [draggingHandle, colorSliders, updateSlider, sliderLayerRange.max])

  const handleDrop = useCallback(
    (e: React.DragEvent) => {
      e.preventDefault()
      try {
        const filament: Filament = JSON.parse(e.dataTransfer.getData('application/json'))
        // Mirror addBand's own criterion for a reusable column: disabled AND
        // never positioned. A disabled column that still has a layer is a
        // deliberately hidden (eye-off) band — reusing it here clobbered its
        // filament/TD/layer and silently re-enabled it instead of adding a
        // new band.
        const firstDisabled = colorSliders.findIndex((s) => !s.enabled && s.layer === 0)
        if (!activeFilaments.some((f) => f.uuid === filament.uuid)) addActiveFilament(filament)
        if (firstDisabled >= 0) {
          const top = Math.max(0, ...colorSliders.filter((s) => s.enabled).map((s) => s.layer))
          const slot = colorSliders[firstDisabled]
          const td = slot && slot.filament_uuid === filament.uuid ? slot.td : filament.td
          updateSlider(firstDisabled, { filament_uuid: filament.uuid, enabled: true, td, layer: Math.min(sliderLayerRange.max, top + 5) || 5 })
        } else {
          // No free column (e.g. a fresh project): add a band instead of
          // overwriting the last one (or crashing on an empty stack).
          useAppStore.getState().addBand(filament)
        }
      } catch {
        // ignore
      }
    },
    [colorSliders, updateSlider, addActiveFilament, activeFilaments, sliderLayerRange.max],
  )

  const handleDragOver = useCallback((e: React.DragEvent) => {
    e.preventDefault()
    e.dataTransfer.dropEffect = 'copy'
  }, [])

  const maxLayer = displayMaxLayer

  // Hover shows the tooltip; so do keyboard focus and dragging.
  const tipIndex = draggingHandle ?? hoveredHandle ?? focusedHandle
  const tipHandle = tipIndex !== null ? handles[tipIndex] : undefined
  // A band hidden under a later one on the same layer owns no layers of its own.
  const tipInfo = tipHandle
    ? bandInfo.get(tipHandle.storeIndex) ?? {
        name: [...filaments, ...activeFilaments].find((f) => f.uuid === tipHandle.filamentUuid)?.name ?? 'Unassigned',
        start: tipHandle.value,
        end: tipHandle.value,
        td: tipHandle.td,
      }
    : undefined
  const tipRect = tipIndex !== null ? handleRefs.current.get(tipIndex)?.getBoundingClientRect() : undefined

  const zoomButton = 'p-0.5 rounded text-gray-400 hover:text-gray-100 hover:bg-gray-700 disabled:opacity-30 disabled:hover:bg-transparent'

  return (
    <div
      style={{
        border: '1px solid var(--cyan-accent)',
        borderRadius: 4,
        backgroundColor: 'var(--bg-panel)',
        height: '100%',
        width: 80,
        overflow: 'hidden',
        display: 'flex',
        flexDirection: 'column',
      }}
    >
      {handles.length > 0 && (
        <div className="flex items-center justify-between px-0.5 border-b border-gray-700 flex-shrink-0" title="Zoom the color column (Ctrl+scroll)">
          <button onClick={() => setZoom((z) => Math.max(MIN_ZOOM, z / 1.5))} disabled={zoom <= MIN_ZOOM} className={zoomButton} aria-label="Zoom out the color column" data-testid="color-core-zoom-out">
            <Minus className="w-3 h-3" />
          </button>
          <button onClick={() => setZoom(1)} className="text-[10px] text-gray-400 tabular-nums hover:text-gray-100" title="Reset zoom" data-testid="color-core-zoom-level">
            {zoom.toFixed(zoom < 10 && zoom % 1 ? 1 : 0)}×
          </button>
          <button onClick={() => setZoom((z) => Math.min(MAX_ZOOM, z * 1.5))} disabled={zoom >= MAX_ZOOM} className={zoomButton} aria-label="Zoom in the color column" data-testid="color-core-zoom-in">
            <Plus className="w-3 h-3" />
          </button>
        </div>
      )}
      <div
        ref={containerRef}
        className={`flex-1 relative ${zoom > 1 ? 'overflow-y-auto overflow-x-hidden' : 'overflow-hidden'}`}
        data-testid="color-core"
        onDrop={handleDrop}
        onDragOver={handleDragOver}
      >
        {handles.length > 0 ? (
          <div ref={contentRef} className="relative flex" style={{ height: containerHeight }}>
            {/* Handles (arrowheads on the left) */}
            <div className="relative w-14 flex-shrink-0 z-20" data-testid="color-core-handles">
              {/* Slider rail */}
              <div className="absolute right-1 top-0 bottom-0 w-0.5 bg-white/15" />
              {handles.map((handle, idx) => {
                const yPos = layerToY(handle.value)
                const isFocused = focusedHandle === idx
                const isLinked = handle.storeIndex === selectedBand || handle.storeIndex === hoveredBand
                const isOverlapDisabled = overlapDisabled.has(handle.storeIndex)
                // With many bands the layer numbers overlapped into an
                // unreadable pile; show a number only where there's room,
                // or for the handle being hovered/dragged.
                const neighbors = [handles[idx - 1], handles[idx + 1]].filter(Boolean).map((h) => Math.abs(layerToY(h.value) - yPos))
                const roomForLabel = neighbors.every((d) => d >= MIN_LABEL_SPACING)
                const showLabel = roomForLabel || isFocused || isLinked || hoveredHandle === idx || draggingHandle === idx
                const info = bandInfo.get(handle.storeIndex)

                return (
                  <div
                    key={handle.storeIndex}
                    ref={(el) => {
                      if (el) handleRefs.current.set(idx, el)
                      else handleRefs.current.delete(idx)
                    }}
                    role="slider"
                    tabIndex={0}
                    aria-label={`Band ${handle.storeIndex + 1}${info ? `, ${info.name}` : ''}: top layer`}
                    aria-valuenow={handle.value}
                    aria-valuemin={1}
                    aria-valuemax={sliderLayerRange.max}
                    className="absolute left-0 flex items-center cursor-grab active:cursor-grabbing select-none outline-none"
                    style={{ top: `${yPos}px`, transform: 'translateY(-50%)', opacity: isOverlapDisabled ? 0.45 : 1, zIndex: (showLabel && !roomForLabel) || isLinked ? 30 : undefined, touchAction: 'none' }}
                    onMouseDown={handleMouseDown(idx)}
                    onMouseEnter={() => {
                      setHoveredHandle(idx)
                      setHoveredBand(handle.storeIndex)
                    }}
                    onMouseLeave={() => {
                      setHoveredHandle((h) => (h === idx ? null : h))
                      setHoveredBand(null)
                    }}
                    onFocus={() => selectHandle(idx)}
                    onBlur={() => setFocusedHandle((h) => (h === idx ? null : h))}
                    onKeyDown={handleKeyDown(idx)}
                    onTouchStart={handleTouchStart(idx)}
                    data-testid={`color-core-handle-${idx}`}
                    data-layer={handle.value}
                    data-selected={handle.storeIndex === selectedBand || undefined}
                    data-overlap-disabled={isOverlapDisabled || undefined}
                  >
                    {/* The border and outline keep white and black handles
                        visible against both the dark panel and the track. */}
                    <div
                      className={`flex items-center pl-1 pr-0.5 py-0.5 rounded-l text-xs font-mono font-bold shadow transition-all border border-black/50 outline outline-1 outline-white/25 ${
                        isFocused ? 'ring-2 ring-white ring-offset-1 ring-offset-gray-800' : isLinked ? 'ring-2 ring-cyan-400' : ''
                      }`}
                      style={{ backgroundColor: handle.color }}
                    >
                      {showLabel && (
                        <span className="text-[10px] leading-none pr-0.5" style={{ color: readableTextColor(handle.color) }} data-testid={`color-core-label-${idx}`}>
                          {handle.value}
                        </span>
                      )}
                      <svg width="12" height="14" viewBox="0 0 12 14" className="flex-shrink-0">
                        <polygon points="12,7 0,0 0,14" fill={readableTextColor(handle.color)} />
                      </svg>
                    </div>
                  </div>
                )
              })}
            </div>

            {/* Segments (track) */}
            <div
              ref={trackRef}
              className="flex-1 relative overflow-hidden"
              data-testid="color-core-segments"
            >
              {segments.map((segment, idx) => {
                const segmentHeight = containerHeight / maxLayer
                return (
                  <div
                    key={segment.layerIndex}
                    className="w-full flex items-center justify-center relative"
                    style={{
                      backgroundColor: segment.color,
                      height: `${segmentHeight}px`,
                      minHeight: `${segmentHeight}px`,
                      boxShadow: 'inset 0 1px 0 rgba(0,0,0,0.25)',
                    }}
                    data-testid={`color-core-segment-${idx}`}
                    data-layer={segment.layerIndex}
                    data-color={segment.color}
                  >
                    {segment.isEquivalent && (
                      <div className="absolute right-0.5 text-[6px] text-white/60 font-mono leading-none" data-testid="color-core-equiv">
                        ||
                      </div>
                    )}
                  </div>
                )
              })}
            </div>

            {/* Connectors (SVG overlay on the right) */}
            <svg
              className="absolute right-0 top-0 h-full w-4 pointer-events-none z-10"
              data-testid="color-core-connectors"
            >
              {handles.map((handle, idx) => {
                const yPos = layerToY(handle.value)
                const nextHandle = handles[idx + 1]
                const nextYPos = nextHandle ? layerToY(nextHandle.value) : yPos

                return (
                  <g key={`connector-${idx}`}>
                    <line
                      x1="6"
                      y1={yPos}
                      x2="6"
                      y2={nextHandle ? nextYPos : yPos}
                      stroke={handle.color}
                      strokeWidth="1.5"
                      strokeDasharray="2,2"
                      opacity="0.6"
                    />
                    <line
                      x1="4"
                      y1={yPos}
                      x2="8"
                      y2={yPos}
                      stroke={handle.color}
                      strokeWidth="1.5"
                      opacity="0.8"
                    />
                    {nextHandle && (
                      <line
                        x1="4"
                        y1={nextYPos}
                        x2="8"
                        y2={nextYPos}
                        stroke={handle.color}
                        strokeWidth="1.5"
                        opacity="0.8"
                      />
                    )}
                  </g>
                )
              })}
            </svg>
          </div>
        ) : (
          <div className="h-full flex items-center justify-center text-center px-1.5 text-[11px] leading-snug text-gray-400" data-testid="color-core-empty">
            Drag a filament here to add a color layer
          </div>
        )}
      </div>

      {tipHandle && tipInfo && tipRect &&
        createPortal(
          <div
            className="fixed z-[70] pointer-events-none px-2 py-1 rounded border border-gray-700 bg-gray-900/95 shadow-lg text-[11px] text-gray-200 whitespace-nowrap"
            style={{ left: tipRect.right + 8, top: tipRect.top + tipRect.height / 2, transform: 'translateY(-50%)' }}
            data-testid="color-core-tooltip"
          >
            <div className="flex items-center gap-1.5 font-medium text-gray-100">
              <span className="w-2.5 h-2.5 rounded-full border border-gray-600" style={{ backgroundColor: tipHandle.color }} />
              {tipInfo.name}
            </div>
            <div className="text-gray-400">
              {tipInfo.start === tipInfo.end ? `Layer ${tipInfo.end}` : `Layers ${tipInfo.start}–${tipInfo.end}`} · TD {tipInfo.td}
              {overlapDisabled.has(tipHandle.storeIndex) ? ' · hidden by a later band on the same layer' : ' · drag or use ↑/↓'}
            </div>
          </div>,
          document.body,
        )}
    </div>
  )
}
