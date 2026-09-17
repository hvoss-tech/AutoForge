import React, { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { useAppStore } from '../store/appStore'
import type { Filament } from '../types'
import { filterActiveHandles } from '../lib/colorStack'

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

function hexColorsEqual(c1: string, c2: string): boolean {
  return c1.toLowerCase() === c2.toLowerCase()
}

export const ColorCore: React.FC = () => {
  const colorSliders = useAppStore((s) => s.colorSliders)
  const sliderLayerRange = useAppStore((s) => s.sliderLayerRange)
  const filaments = useAppStore((s) => s.filaments)
  const updateSlider = useAppStore((s) => s.updateSlider)
  const addActiveFilament = useAppStore((s) => s.addActiveFilament)
  const containerRef = useRef<HTMLDivElement>(null)
  const trackRef = useRef<HTMLDivElement>(null)
  const [focusedHandle, setFocusedHandle] = useState<number | null>(null)
  const [draggingHandle, setDraggingHandle] = useState<number | null>(null)
  const [containerHeight, setContainerHeight] = useState(300)

  useEffect(() => {
    if (!containerRef.current) return
    const updateHeight = () => {
      if (containerRef.current) {
        setContainerHeight(containerRef.current.clientHeight)
      }
    }
    updateHeight()
    const observer = new ResizeObserver(updateHeight)
    observer.observe(containerRef.current)
    return () => observer.disconnect()
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

  const layerToY = useCallback(
    (layer: number): number => {
      if (handles.length === 0) return 0
      const maxLayer = handles[handles.length - 1].value
      const segmentHeight = containerHeight / maxLayer
      return segmentHeight * (maxLayer - layer) + segmentHeight / 2
    },
    [handles, containerHeight],
  )

  const handleMouseDown = useCallback(
    (handleIndex: number) => (e: React.MouseEvent) => {
      e.preventDefault()
      setFocusedHandle(handleIndex)
      setDraggingHandle(handleIndex)
    },
    [],
  )

  const handleTouchStart = useCallback(
    (handleIndex: number) => (e: React.TouchEvent) => {
      setFocusedHandle(handleIndex)
      setDraggingHandle(handleIndex)
    },
    [],
  )

  useEffect(() => {
    if (draggingHandle === null) return

    const activeSliders = colorSliders
      .map((s, idx) => ({ ...s, storeIndex: idx }))
      .filter((s) => s.enabled && s.layer > 0)
      .sort((a, b) => a.layer - b.layer)

    const handle = activeSliders[draggingHandle]
    if (!handle) return

    const prevHandle = draggingHandle > 0 ? activeSliders[draggingHandle - 1] : null
    const nextHandle = draggingHandle < activeSliders.length - 1 ? activeSliders[draggingHandle + 1] : null

    // Sliders may now be dragged onto the exact same layer as a neighbor
    // (that's how "move over one another" works) — bounds are inclusive of
    // the neighbor's own layer instead of stopping one short of it. This
    // also fixes the previous exclusive bounds going invalid (min > max)
    // whenever two neighbors were already only 1 layer apart.
    const minLayer = prevHandle ? prevHandle.layer : 1
    const maxLayer = nextHandle ? nextHandle.layer : sliderLayerRange.max

    const handleMove = (clientY: number) => {
      if (!containerRef.current) return
      const rect = containerRef.current.getBoundingClientRect()
      const relativeY = clientY - rect.top
      const currentMaxLayer = activeSliders[activeSliders.length - 1].layer
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
        const firstDisabled = colorSliders.findIndex((s) => !s.enabled)
        const targetIndex = firstDisabled >= 0 ? firstDisabled : colorSliders.length - 1
        addActiveFilament(filament)
        updateSlider(targetIndex, { filament_uuid: filament.uuid, enabled: true, td: filament.td, layer: 10 })
      } catch {
        // ignore
      }
    },
    [colorSliders, updateSlider, addActiveFilament],
  )

  const handleDragOver = useCallback((e: React.DragEvent) => {
    e.preventDefault()
    e.dataTransfer.dropEffect = 'copy'
  }, [])

  const maxLayer = handles.length > 0 ? handles[handles.length - 1].value : 0

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
      <div
        ref={containerRef}
        className="flex-1 relative flex overflow-hidden"
        data-testid="color-core"
        onDrop={handleDrop}
        onDragOver={handleDragOver}
      >
        {handles.length > 0 ? (
          <>
            {/* Handles (arrowheads on the left) */}
            <div className="relative w-14 flex-shrink-0 z-20" data-testid="color-core-handles">
              {/* Slider rail */}
              <div className="absolute right-1 top-0 bottom-0 w-0.5 bg-white/15" />
              {handles.map((handle, idx) => {
                const yPos = layerToY(handle.value)
                const isFocused = focusedHandle === idx
                const isOverlapDisabled = overlapDisabled.has(handle.storeIndex)

                return (
                  <div
                    key={handle.storeIndex}
                    className="absolute left-0 flex items-center cursor-grab active:cursor-grabbing select-none"
                    style={{ top: `${yPos}px`, transform: 'translateY(-50%)', opacity: isOverlapDisabled ? 0.45 : 1 }}
                    onMouseDown={handleMouseDown(idx)}
                    onTouchStart={handleTouchStart(idx)}
                    data-testid={`color-core-handle-${idx}`}
                    data-layer={handle.value}
                    data-overlap-disabled={isOverlapDisabled || undefined}
                    title={isOverlapDisabled ? 'This filament is covered by another slider on the same layer' : undefined}
                  >
                    <div
                      className={`flex items-center pl-1 pr-0.5 py-0.5 rounded-l text-xs font-mono font-bold shadow transition-all ${
                        isFocused ? 'ring-2 ring-white ring-offset-1 ring-offset-gray-800' : ''
                      }`}
                      style={{ backgroundColor: handle.color }}
                    >
                      <span className="text-white text-[10px] leading-none">{handle.value}</span>
                      <svg width="12" height="14" viewBox="0 0 12 14" className="flex-shrink-0">
                        <polygon points="12,7 0,0 0,14" fill="white" />
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
          </>
        ) : (
          <div className="flex-1 flex items-center justify-center text-gray-500 text-xs" data-testid="color-core-empty">
            Drag filaments here to assign colors
          </div>
        )}
      </div>
    </div>
  )
}
