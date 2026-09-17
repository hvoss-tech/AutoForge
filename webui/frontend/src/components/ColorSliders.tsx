import React, { useEffect, useMemo, useRef } from 'react'
import { useDebouncedCallback } from '../hooks/useDebouncedCallback'
import { useAppStore } from '../store/appStore'
import type { Filament } from '../types'
import { getOverlapDisabledIndices } from '../lib/colorStack'

// Mirrors api/preview.py's INIT_JOB_SENTINEL — tells the backend "there's
// no completed job yet, render against the post-upload auto-preview state
// instead" when POSTing a slider edit.
const INIT_JOB_SENTINEL = '__init__'

export const ColorSliders: React.FC = () => {
  const colorSliders = useAppStore((s) => s.colorSliders)
  const sliderLayerRange = useAppStore((s) => s.sliderLayerRange)
  const updateSlider = useAppStore((s) => s.updateSlider)
  const addActiveFilament = useAppStore((s) => s.addActiveFilament)
  const filaments = useAppStore((s) => s.filaments)
  const currentJob = useAppStore((s) => s.currentJob)
  const initState = useAppStore((s) => s.initState)
  const activeFilaments = useAppStore((s) => s.activeFilaments)
  const lastRenderedSlidersRef = useRef(JSON.stringify(colorSliders))
  const [isRendering, setIsRendering] = React.useState(false)

  // Slider edits only make sense once there's a discretized solution to
  // recolor — either a completed job, or (now that api/init.py actually
  // runs the heightmap-init algorithm instead of a no-op stub) the
  // pre-run auto-preview: its height map is real, so a slider edit can be
  // rendered against it exactly like a completed job's, letting the user
  // assign colors manually before ever running the real optimizer.
  const hasResult = currentJob?.status === 'completed' || initState.status === 'ready'
  const jobId = currentJob?.status === 'completed' ? currentJob.job_id : INIT_JOB_SENTINEL

  const triggerPreviewRender = useDebouncedCallback(async (sliders: typeof colorSliders, filaments: Filament[], jobId: string | undefined) => {
    setIsRendering(true)
    try {
      await fetch('/api/preview/render-with-sliders', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ sliders, active_filaments: filaments, job_id: jobId }),
      })
    } catch {
      // Ignore — the WS-driven preview simply won't update this round.
    } finally {
      setIsRendering(false)
    }
  }, [], 150)

  useEffect(() => {
    if (!hasResult) return

    const currentKey = JSON.stringify(colorSliders)
    if (currentKey === lastRenderedSlidersRef.current) return

    const hasChanges = colorSliders.some((s, i) => {
      const prev = JSON.parse(lastRenderedSlidersRef.current)[i]
      return s.td !== prev.td || s.layer !== prev.layer || s.enabled !== prev.enabled || s.filament_uuid !== prev.filament_uuid
    })

    if (hasChanges) {
      triggerPreviewRender(colorSliders, activeFilaments, jobId)
    }

    lastRenderedSlidersRef.current = currentKey
  }, [colorSliders, hasResult, activeFilaments, jobId, triggerPreviewRender])

  // Sliders can now be dragged onto the same layer as one another; when that
  // happens only the right-most one actually governs that layer's material
  // (matches ColorCore), so its overlapped neighbors are dimmed here as a
  // silent "this one currently has no effect" cue.
  const overlapDisabled = useMemo(() => getOverlapDisabledIndices(colorSliders, filaments), [colorSliders, filaments])

  const findFilament = (uuid: string): Filament | undefined => {
    if (!uuid) return undefined
    return filaments.find((f) => f.uuid === uuid) ?? activeFilaments.find((f) => f.uuid === uuid)
  }

  const handleDrop = (e: React.DragEvent, index: number) => {
    e.preventDefault()
    try {
      const filament: Filament = JSON.parse(e.dataTransfer.getData('application/json'))
      // A filament dragged in from the library may not be active yet — a
      // slider referencing a filament_uuid outside active_filaments looks
      // fine here (this panel falls back to the full library for display)
      // but is invisible to the optimizer, which only ever sees
      // active_filaments, so the assignment silently wouldn't survive the
      // next run.
      addActiveFilament(filament)
      updateSlider(index, { filament_uuid: filament.uuid, enabled: true, td: filament.td })
    } catch {
      // ignore
    }
  }

  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault()
    e.dataTransfer.dropEffect = 'copy'
  }

  const handleTdChange = (index: number, td: number) => {
    updateSlider(index, { td })
  }

  const handleLayerChange = (index: number, layer: number) => {
    updateSlider(index, { layer })
  }

  const handleWheel = (index: number, currentLayer: number, enabled: boolean) => (e: React.WheelEvent) => {
    if (!enabled) return
    e.preventDefault()
    // Wheel up (deltaY < 0) raises the layer, matching the slider's own
    // "up = higher layer" direction (it's inverted top-to-bottom via
    // writingMode/direction below), so the two never feel backwards.
    const step = e.deltaY < 0 ? 1 : -1
    const next = Math.min(sliderLayerRange.max, Math.max(sliderLayerRange.min, currentLayer + step))
    if (next !== currentLayer) handleLayerChange(index, next)
  }

  return (
    <div style={{ backgroundColor: 'var(--bg-panel)', borderTop: '1px solid var(--border)' }} data-testid="color-sliders-panel">
      {/* Header */}
      <div style={{ padding: '4px 8px', borderBottom: '1px solid var(--border)', display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
        <h3 style={{ fontSize: 11, fontWeight: 600, color: 'var(--text-primary)' }}>Color Sliders</h3>
        <span style={{ fontSize: 10, color: 'var(--text-secondary)' }} data-testid="slider-render-status">
          {isRendering ? 'Rendering…' : `${colorSliders.length} column${colorSliders.length === 1 ? '' : 's'}`}
        </span>
      </div>

      {/* Sliders — however many the optimizer/pruner actually produced, not
          a fixed count, so this scrolls horizontally instead of clipping or
          padding out fake empty columns. */}
      <div style={{ display: 'flex', gap: 2, padding: 4, overflowX: 'auto', overflowY: 'hidden' }} data-testid="slider-columns">
        {colorSliders.map((slider, i) => {
          const filament = findFilament(slider.filament_uuid)
          const color = filament?.color ?? '#333333'
          const label = filament ? filament.name : slider.enabled ? 'Empty' : ''
          const isOverlapDisabled = overlapDisabled.has(i)
          return (
            <div
              key={i}
              style={{ width: 50, flexShrink: 0, display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 2, fontSize: 10, opacity: isOverlapDisabled ? 0.45 : 1 }}
              onDrop={(e) => handleDrop(e, i)}
              onDragOver={handleDragOver}
              data-testid={`slider-column-${i}`}
              data-overlap-disabled={isOverlapDisabled || undefined}
              title={isOverlapDisabled ? 'This filament is covered by another slider on the same layer' : undefined}
            >
              <div style={{ fontSize: 7, color: 'var(--text-secondary)', letterSpacing: 0.5 }}>TD</div>
              <input
                type="number"
                value={slider.td}
                onChange={(e) => handleTdChange(i, parseFloat(e.target.value) || 0)}
                style={{ backgroundColor: 'var(--bg-input)', border: '1px solid var(--border)', borderRadius: 2, width: 36, padding: '1px 2px', textAlign: 'center', fontSize: 9, color: 'var(--text-primary)' }}
                step={0.1}
                disabled={!slider.enabled}
                data-testid={`td-input-${i}`}
              />
              <input
                type="range"
                value={slider.layer}
                min={sliderLayerRange.min}
                max={sliderLayerRange.max}
                onChange={(e) => handleLayerChange(i, parseInt(e.target.value) || 0)}
                onWheel={handleWheel(i, slider.layer, slider.enabled)}
                disabled={!slider.enabled}
                className="af-vertical-slider"
                style={{ height: 70, width: 12, writingMode: 'vertical-lr', direction: 'rtl' } as React.CSSProperties}
                data-testid={`slider-${i}`}
              />
              <input
                type="number"
                value={slider.layer}
                min={sliderLayerRange.min}
                max={sliderLayerRange.max}
                onChange={(e) => handleLayerChange(i, parseInt(e.target.value) || 0)}
                disabled={!slider.enabled}
                style={{ backgroundColor: 'var(--bg-input)', border: '1px solid var(--border)', borderRadius: 2, width: 36, padding: '1px 2px', textAlign: 'center', fontSize: 9, color: 'var(--cyan-accent)' }}
                data-testid={`layer-input-${i}`}
              />
              <div style={{ fontSize: 8, color: 'var(--text-secondary)' }} data-testid={`depth-${i}`}>{slider.depth_mm?.toFixed(2)}</div>
              <div
                style={{ width: 12, height: 12, borderRadius: '50%', border: '1px solid var(--border)', backgroundColor: color, flexShrink: 0 }}
                data-testid={`color-indicator-${i}`}
                title={filament?.name}
              />
              <div
                style={{
                  fontSize: 8,
                  color: 'var(--text-secondary)',
                  width: '100%',
                  overflow: 'hidden',
                  textOverflow: 'ellipsis',
                  whiteSpace: 'nowrap',
                  textAlign: 'center',
                }}
                title={filament?.name}
                data-testid={`filament-label-${i}`}
              >
                {label}
              </div>
              <button
                onClick={() => updateSlider(i, { enabled: !slider.enabled })}
                data-testid={`toggle-${i}`}
                style={{ cursor: 'pointer', background: 'none', border: 'none', color: 'var(--cyan-accent)', fontSize: 10 }}
              >
                {slider.enabled ? '-' : '+'}
              </button>
            </div>
          )
        })}
      </div>
    </div>
  )
}
