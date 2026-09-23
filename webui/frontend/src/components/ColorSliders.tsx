import React, { useEffect, useMemo, useRef, useState } from 'react'
import { ArrowDownUp, Eye, EyeOff, GripVertical, ListPlus, Lock, Plus, RotateCcw, Trash2 } from 'lucide-react'
import { useAppStore } from '../store/appStore'
import { NumberInput } from './ui/number-input'
import { FilamentPicker } from './FilamentPicker'
import type { Filament } from '../types'
import { baseIsAuto, baseLayerCount, effectiveBaseColor, effectiveBaseFilamentUuid } from '../lib/baseColor'
import { getOverlapDisabledIndices } from '../lib/colorStack'
import { getPlanBands } from '../lib/printPlan'
import { isInPrintOrder } from '../lib/bandOps'

const UNASSIGNED_COLOR = '#555555'
const BAND_DRAG_TYPE = 'application/x-autoforge-band'
const GRID_COLUMNS = 'grid-cols-[1rem_2rem_minmax(9rem,14rem)_4.5rem_minmax(8rem,1fr)_4rem_4.5rem_4.5rem_6rem]'

/** Range input that scrolls the list instead of moving the band.
 *
 * The wheel used to nudge the value one layer per notch. That made the list
 * a minefield: scrolling through the bands moved whichever slider the
 * pointer happened to cross, silently rewriting the stack. Firefox changes a
 * range input's value on wheel natively too, so ignoring the event isn't
 * enough — it has to be cancelled (which needs a non-passive listener, hence
 * the manual registration rather than React's passive onWheel) and the
 * scrolling it would have caused performed on the list by hand.
 *
 * Spans the whole layer range on purpose: dragging a band past its
 * neighbours is how bands trade places. */
const LayerSlider: React.FC<{
  value: number
  min: number
  max: number
  disabled: boolean
  onChange: (layer: number) => void
  scrollTargetRef: React.RefObject<HTMLElement>
  testId: string
}> = ({ value, min, max, disabled, onChange, scrollTargetRef, testId }) => {
  const ref = useRef<HTMLInputElement>(null)

  useEffect(() => {
    const el = ref.current
    if (!el) return
    const onWheel = (e: WheelEvent) => {
      e.preventDefault()
      const list = scrollTargetRef.current
      if (!list) return
      // deltaMode 1 is lines, 2 is pages; normalise both to pixels so a
      // wheel notch moves the list by the same amount it would anywhere else.
      const factor = e.deltaMode === 1 ? 16 : e.deltaMode === 2 ? list.clientHeight : 1
      list.scrollTop += e.deltaY * factor
    }
    el.addEventListener('wheel', onWheel, { passive: false })
    return () => el.removeEventListener('wheel', onWheel)
  }, [scrollTargetRef])

  return (
    <input
      ref={ref}
      type="range"
      value={value}
      min={min}
      max={max}
      disabled={disabled}
      onChange={(e) => onChange(parseInt(e.target.value) || 0)}
      className="w-full accent-cyan-500 disabled:opacity-40"
      aria-label="Top layer of this band"
      data-testid={testId}
    />
  )
}

/** The base slab, as the bottom row of the color layers.
 *
 * It is a real part of the print — the first thing off the plate, and the
 * color every translucent layer above is seen against — but it is not one of
 * the optimizer's bands, so it never appeared here at all and its color was
 * only reachable through Settings. It is shown like any other band with one
 * difference: its thickness is fixed (that's the Base field in the status
 * bar, one value for the whole print), so only the filament can be changed.
 *
 * With `auto_background_color` on, the color shown is the one the pipeline
 * resolved (`resolvedBase`), not the one in settings — auto-selection picks
 * the active filament closest to the image's dominant color, and settings
 * still holds whatever was there before. */
const BaseBandRow: React.FC<{ findFilament: (uuid: string) => Filament | undefined }> = ({ findFilament }) => {
  const settings = useAppStore((s) => s.settings)
  const resolvedBase = useAppStore((s) => s.resolvedBase)
  const setBaseFilament = useAppStore((s) => s.setBaseFilament)

  const heightMm = settings.background_height || 0
  const layers = baseLayerCount(settings)
  const color = effectiveBaseColor(settings, resolvedBase)
  const filamentUuid = effectiveBaseFilamentUuid(resolvedBase)
  const filament = filamentUuid ? findFilament(filamentUuid) : undefined
  const auto = baseIsAuto(settings, resolvedBase)

  return (
    <div
      className={`grid ${GRID_COLUMNS} gap-x-2 items-center px-3 py-1 border-b border-gray-700 bg-gray-900/40 text-xs`}
      data-testid="base-band-row"
      data-base-color={color}
      title="The solid base the color layers are printed on. Its thickness is the Base setting in the status bar; only its filament can be changed here."
    >
      <span className="text-gray-500" title="The base is always the bottom of the stack">
        <Lock className="w-3.5 h-3.5" />
      </span>

      <span className="text-gray-400">Base</span>

      <div className="flex items-center gap-1.5 min-w-0">
        <FilamentPicker
          value={filamentUuid}
          fallbackColor={color}
          onChange={setBaseFilament}
          label="Filament for the base"
          testId="base-filament-select"
          swatchTestId="base-color-indicator"
        />
        <span className="sr-only" data-testid="base-filament-label">{filament ? filament.name : color}</span>
      </div>

      <span className="text-gray-300 tabular-nums" data-testid="base-band-range">
        {layers > 0 ? `base (${layers})` : 'base'}
      </span>

      {/* Where every other row has a draggable slider: a fixed bar, so the
          base reads as part of the same stack without inviting a drag. */}
      <div className="h-1.5 w-full rounded-full bg-gray-700 overflow-hidden" data-testid="base-fixed-track" aria-hidden>
        <div className="h-full w-full" style={{ backgroundColor: color }} />
      </div>
      <span className="text-[11px] text-gray-500 text-center" data-testid="base-fixed-note">Fixed</span>

      <span className="text-gray-300 tabular-nums" data-testid="base-height">{heightMm.toFixed(2)} mm</span>
      <span className="text-gray-500 text-center" title="The base is solid — no light passes through it">—</span>
      <span className="text-[11px] text-gray-500 text-right" data-testid="base-auto-note">
        {auto ? 'auto' : 'manual'}
      </span>
    </div>
  )
}

/** Proportional strip of the whole stack; click a band to jump to its row. */
const StackOverview: React.FC<{ onSelect: (storeIndex: number) => void }> = ({ onSelect }) => {
  const colorSliders = useAppStore((s) => s.colorSliders)
  const filaments = useAppStore((s) => s.filaments)
  const activeFilaments = useAppStore((s) => s.activeFilaments)
  const settings = useAppStore((s) => s.settings)
  const selectedBand = useAppStore((s) => s.selectedBand)
  const hoveredBand = useAppStore((s) => s.hoveredBand)
  const setHoveredBand = useAppStore((s) => s.setHoveredBand)
  const bands = useMemo(
    () => getPlanBands(colorSliders, [...activeFilaments, ...filaments], settings),
    [colorSliders, filaments, activeFilaments, settings],
  )
  const resolvedBase = useAppStore((s) => s.resolvedBase)
  const baseColor = effectiveBaseColor(settings, resolvedBase)
  // At least one, so the base is always a visible sliver of the strip.
  const baseLayers = Math.max(1, baseLayerCount(settings))
  const top = bands.length ? bands[bands.length - 1].endLayer : 0
  if (!top) return null
  return (
    <div className="flex items-center gap-1.5 text-[10px] text-gray-500">
      <span className="flex-shrink-0" title="The solid base, printed first">Base</span>
      <div className="flex-1 flex h-3 rounded overflow-hidden border border-gray-700" data-testid="stack-overview" title="The whole stack, first printed layer on the left">
        {/* The base leads the strip so the overview matches what comes off
            the plate, rather than starting at the first color layer. */}
        <div
          style={{ flexGrow: baseLayers, backgroundColor: baseColor }}
          className="h-full border-r border-gray-900/60"
          title={`Base: ${baseColor}`}
          data-testid="stack-overview-base"
        />
        {bands.map((b) => {
          const marked = b.storeIndex === selectedBand || b.storeIndex === hoveredBand
          return (
            <button
              key={b.storeIndex}
              onClick={() => onSelect(b.storeIndex)}
              onMouseEnter={() => setHoveredBand(b.storeIndex)}
              onMouseLeave={() => setHoveredBand(null)}
              style={{ flexGrow: b.endLayer - b.startLayer + 1, backgroundColor: b.filament?.color ?? UNASSIGNED_COLOR }}
              className={`h-full hover:brightness-125 ${marked ? 'relative z-10 outline outline-2 -outline-offset-2 outline-cyan-400' : ''}`}
              title={`${b.filament?.name ?? 'Unassigned'}: layers ${b.startLayer}–${b.endLayer}`}
              aria-label={`Band ${b.storeIndex + 1}`}
              data-selected={marked || undefined}
            />
          )
        })}
      </div>
      <span className="flex-shrink-0 tabular-nums" title="The top of the print">Top ({top})</span>
    </div>
  )
}

export const ColorSliders: React.FC = () => {
  const colorSliders = useAppStore((s) => s.colorSliders)
  const sliderLayerRange = useAppStore((s) => s.sliderLayerRange)
  const updateSlider = useAppStore((s) => s.updateSlider)
  const addActiveFilament = useAppStore((s) => s.addActiveFilament)
  const addBand = useAppStore((s) => s.addBand)
  const removeBand = useAppStore((s) => s.removeBand)
  const moveBand = useAppStore((s) => s.moveBand)
  const insertBandAbove = useAppStore((s) => s.insertBandAbove)
  const sortBandsByLayer = useAppStore((s) => s.sortBandsByLayer)
  const selectedBand = useAppStore((s) => s.selectedBand)
  const hoveredBand = useAppStore((s) => s.hoveredBand)
  const setSelectedBand = useAppStore((s) => s.setSelectedBand)
  const setHoveredBand = useAppStore((s) => s.setHoveredBand)
  const pushToast = useAppStore((s) => s.pushToast)
  const filaments = useAppStore((s) => s.filaments)
  const activeFilaments = useAppStore((s) => s.activeFilaments)
  const settings = useAppStore((s) => s.settings)
  const layerHeight = settings.layer_height || 0.04
  const [filter, setFilter] = useState<string>('all')
  const [dropTarget, setDropTarget] = useState<number | null>(null)
  const listRef = useRef<HTMLDivElement>(null)

  const knownFilaments = useMemo(() => {
    const byUuid = new Map<string, Filament>()
    for (const f of [...filaments, ...activeFilaments]) byUuid.set(f.uuid, f)
    return byUuid
  }, [filaments, activeFilaments])
  const findFilament = (uuid: string): Filament | undefined => (uuid ? knownFilaments.get(uuid) : undefined)

  // Only one slider can own a layer; the ones it covers are dimmed.
  const overlapDisabled = useMemo(() => getOverlapDisabledIndices(colorSliders, filaments), [colorSliders, filaments])

  // Band start = previous enabled band's top layer + 1, for the "layers a–b" readout.
  const bandStarts = useMemo(() => {
    const starts = new Map<number, number>()
    for (const b of getPlanBands(colorSliders, [], settings)) starts.set(b.storeIndex, b.startLayer)
    return starts
  }, [colorSliders, settings])

  const inPrintOrder = useMemo(() => isInPrintOrder(colorSliders), [colorSliders])

  const usedFilaments = useMemo(() => {
    const seen = new Map<string, Filament>()
    for (const s of colorSliders) {
      const f = findFilament(s.filament_uuid)
      if (f) seen.set(f.uuid, f)
    }
    return [...seen.values()]
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [colorSliders, knownFilaments])

  useEffect(() => {
    if (filter !== 'all' && filter !== 'unassigned' && !usedFilaments.some((f) => f.uuid === filter)) setFilter('all')
  }, [filter, usedFilaments])

  // Selecting a band anywhere (color core, overview strip) brings its row into view.
  useEffect(() => {
    if (selectedBand === null) return
    requestAnimationFrame(() => {
      listRef.current?.querySelector(`[data-testid="slider-column-${selectedBand}"]`)?.scrollIntoView({ block: 'nearest' })
    })
  }, [selectedBand])

  const assignFilament = (index: number, filament: Filament) => {
    // The optimizer only sees active filaments; a slider pointing at an
    // inactive one would silently not survive the next run.
    if (!activeFilaments.some((f) => f.uuid === filament.uuid)) addActiveFilament(filament)
    // Only reset TD to the filament's default when the band is actually
    // switching filaments — otherwise a manual TD override on a band gets
    // silently discarded any time this is called with the same filament.
    const current = colorSliders[index]
    const td = current && current.filament_uuid === filament.uuid ? current.td : filament.td
    updateSlider(index, { filament_uuid: filament.uuid, enabled: true, td })
  }

  const readDroppedFilament = (e: React.DragEvent): Filament | null => {
    e.preventDefault()
    try {
      return JSON.parse(e.dataTransfer.getData('application/json'))
    } catch {
      return null
    }
  }

  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault()
    e.dataTransfer.dropEffect = 'copy'
  }

  const selectFromOverview = (storeIndex: number) => {
    setFilter('all')
    setSelectedBand(storeIndex)
  }

  const handleInsert = (index: number) => {
    if (!insertBandAbove(index)) pushToast('No room for another band here: every layer up to the maximum is already used.', 'warning')
  }

  const visible = colorSliders
    .map((slider, i) => ({ slider, i }))
    .filter(({ slider }) => filter === 'all' || (filter === 'unassigned' ? !slider.filament_uuid : slider.filament_uuid === filter))

  if (colorSliders.length === 0) {
    return (
      <div
        className="h-full flex flex-col"
        onDrop={(e) => { const f = readDroppedFilament(e); if (f) { addActiveFilament(f); addBand(f) } }}
        onDragOver={handleDragOver}
        data-testid="color-sliders-panel"
      >
        {/* The base is part of every print, with or without color layers. */}
        <BaseBandRow findFilament={findFilament} />
        <div className="flex-1 flex flex-col items-center justify-center gap-2 text-center p-4">
          <p className="text-sm text-gray-300">No color layers yet</p>
          <p className="text-xs text-gray-400 max-w-md">
            Run the optimizer to fill these in automatically, or drag a filament here to start stacking colors by hand.
          </p>
          <button onClick={() => addBand()} className="mt-1 flex items-center gap-1 px-3 py-1.5 text-xs rounded bg-gray-700 hover:bg-gray-600 text-gray-100" data-testid="add-band-btn">
            <Plus className="w-3.5 h-3.5" /> Add a layer band
          </button>
        </div>
      </div>
    )
  }

  return (
    <div className="h-full flex flex-col" data-testid="color-sliders-panel">
      <div className="flex items-center gap-3 px-3 py-1.5">
        <div className="flex-1 min-w-0">
          <StackOverview onSelect={selectFromOverview} />
        </div>
        {!inPrintOrder && (
          <button
            onClick={sortBandsByLayer}
            className="flex items-center gap-1 px-2 py-1 text-xs rounded border border-amber-500/50 text-amber-500 hover:bg-amber-500/10"
            title="Bands were dragged past each other, so the rows no longer follow the print. Sort them by layer."
            data-testid="sort-bands-btn"
          >
            <ArrowDownUp className="w-3.5 h-3.5" /> Sort by layer
          </button>
        )}
        <label className="flex items-center gap-1.5 text-xs text-gray-400">
          Show
          <select
            value={filter}
            onChange={(e) => setFilter(e.target.value)}
            className="text-xs bg-gray-800 border border-gray-700 rounded px-1.5 py-0.5 text-gray-200"
            data-testid="band-filter"
          >
            <option value="all">All filaments</option>
            {usedFilaments.map((f) => (
              <option key={f.uuid} value={f.uuid}>{f.name}</option>
            ))}
            <option value="unassigned">Unassigned</option>
          </select>
        </label>
        <button onClick={() => addBand()} className="flex items-center gap-1 px-2 py-1 text-xs rounded bg-gray-700 hover:bg-gray-600 text-gray-100" data-testid="add-band-btn">
          <Plus className="w-3.5 h-3.5" /> Add band
        </button>
      </div>

      <div className={`grid ${GRID_COLUMNS} gap-x-2 px-3 pb-1 text-[11px] uppercase tracking-wide text-gray-500 border-b border-gray-800`}>
        <span />
        <span>Band</span>
        <span>Filament</span>
        <span>Layers</span>
        <span>Top layer</span>
        <span />
        <span>Height</span>
        <span title="Transmission distance: how see-through the filament is">TD</span>
        <span className="text-right">Actions</span>
      </div>

      <div ref={listRef} className="flex-1 overflow-y-auto" data-testid="slider-columns" onMouseLeave={() => setHoveredBand(null)}>
        {/* Bottom of the print, so it leads the list — the same order the
            rows below follow (first printed layer first). */}
        <BaseBandRow findFilament={findFilament} />
        {visible.map(({ slider, i }) => {
          const filament = findFilament(slider.filament_uuid)
          const isOverlapDisabled = overlapDisabled.has(i)
          const start = bandStarts.get(i)
          const isSelected = selectedBand === i
          const isHovered = hoveredBand === i
          const tdOverridden = !!filament && slider.td !== filament.td
          return (
            <div
              key={i}
              className={`relative grid ${GRID_COLUMNS} gap-x-2 items-center px-3 py-1 border-b border-gray-800 text-xs ${
                isSelected ? 'bg-cyan-500/15 shadow-[inset_2px_0_0_0_rgb(6,182,212)]' : isHovered ? 'bg-gray-800/80' : 'hover:bg-gray-800/60'
              } ${!slider.enabled || isOverlapDisabled ? 'opacity-50' : ''}`}
              onMouseEnter={() => setHoveredBand(i)}
              onClick={() => setSelectedBand(i)}
              onDragOver={(e) => {
                handleDragOver(e)
                if (e.dataTransfer.types.includes(BAND_DRAG_TYPE)) {
                  // Unplaced columns (layer 0) always stay at the end,
                  // unordered by layer — reordering a band onto one has
                  // nothing to do. Showing the same drop indicator here as
                  // everywhere else invited a drop that silently did
                  // nothing, with no sign it had been rejected.
                  if (slider.layer === 0) {
                    // No indicator here — a drop still fires the rejection
                    // toast below (in onDrop), but forcing dropEffect to
                    // 'none' would stop the browser from firing `drop` at
                    // all, silently swallowing the rejection along with it.
                    setDropTarget(null)
                    return
                  }
                  e.dataTransfer.dropEffect = 'move'
                  setDropTarget(i)
                }
              }}
              onDragLeave={() => setDropTarget((t) => (t === i ? null : t))}
              onDrop={(e) => {
                setDropTarget(null)
                const from = e.dataTransfer.getData(BAND_DRAG_TYPE)
                if (from !== '') {
                  e.preventDefault()
                  if (slider.layer === 0) {
                    pushToast('Drop onto a placed band to reorder — this column has no layer yet.', 'warning')
                    return
                  }
                  if (Number(from) !== i) moveBand(Number(from), i)
                  return
                }
                const f = readDroppedFilament(e)
                if (f) assignFilament(i, f)
              }}
              data-testid={`slider-column-${i}`}
              data-selected={isSelected || undefined}
              data-overlap-disabled={isOverlapDisabled || undefined}
              title={isOverlapDisabled ? 'Hidden by a later band on the same layer' : undefined}
            >
              {dropTarget === i && <div className="absolute inset-x-0 top-0 h-0.5 bg-cyan-500 pointer-events-none" data-testid="band-drop-indicator" />}
              <span
                draggable={slider.enabled && slider.layer > 0}
                onDragStart={(e) => {
                  e.dataTransfer.setData(BAND_DRAG_TYPE, String(i))
                  e.dataTransfer.effectAllowed = 'move'
                }}
                onDragEnd={() => setDropTarget(null)}
                className={`text-gray-500 ${slider.enabled && slider.layer > 0 ? 'cursor-grab hover:text-gray-200' : 'opacity-30'}`}
                title="Drag to move this band (with its thickness) to another place in the stack"
                aria-label={`Move band ${i + 1}`}
                data-testid={`band-grip-${i}`}
              >
                <GripVertical className="w-3.5 h-3.5" />
              </span>

              <span className="text-gray-400 tabular-nums">{i + 1}</span>

              <div className="flex items-center gap-1.5 min-w-0">
                <FilamentPicker
                  value={slider.filament_uuid}
                  onChange={(f) => assignFilament(i, f)}
                  label={`Filament for band ${i + 1}`}
                  testId={`filament-select-${i}`}
                  swatchTestId={`color-indicator-${i}`}
                />
                <span className="sr-only" data-testid={`filament-label-${i}`}>{filament ? filament.name : slider.enabled ? 'Empty' : ''}</span>
              </div>

              <span className="text-gray-300 tabular-nums" data-testid={`band-range-${i}`}>
                {slider.enabled && slider.layer > 0 && start !== undefined ? (start === slider.layer ? `${start}` : `${start}–${slider.layer}`) : '—'}
              </span>

              <LayerSlider
                value={slider.layer}
                min={sliderLayerRange.min}
                max={sliderLayerRange.max}
                disabled={!slider.enabled}
                onChange={(layer) => updateSlider(i, { layer })}
                scrollTargetRef={listRef}
                testId={`slider-${i}`}
              />
              <NumberInput
                value={slider.layer}
                min={sliderLayerRange.min}
                max={sliderLayerRange.max}
                integer
                onValueChange={(layer) => updateSlider(i, { layer })}
                disabled={!slider.enabled}
                className="w-full bg-gray-900 border border-gray-700 rounded px-1 py-0.5 text-center text-cyan-500 disabled:opacity-50"
                aria-label={`Top layer of band ${i + 1}`}
                data-testid={`layer-input-${i}`}
              />
              <span className="text-gray-300 tabular-nums" data-testid={`depth-${i}`}>{(slider.layer * layerHeight).toFixed(2)} mm</span>
              <div className="flex items-center gap-0.5">
                <NumberInput
                  value={slider.td}
                  onValueChange={(td) => updateSlider(i, { td })}
                  min={0}
                  step={0.1}
                  disabled={!slider.enabled}
                  className={`w-full min-w-0 bg-gray-900 border rounded px-1 py-0.5 text-center text-gray-100 disabled:opacity-50 ${tdOverridden ? 'border-amber-500/60' : 'border-gray-700'}`}
                  aria-label={`TD of band ${i + 1}`}
                  data-testid={`td-input-${i}`}
                />
                {tdOverridden && (
                  <button
                    onClick={(e) => {
                      e.stopPropagation()
                      updateSlider(i, { td: filament.td })
                    }}
                    className="p-0.5 rounded text-amber-500 hover:bg-gray-700"
                    title={`Changed for this band only — reset to the filament's TD (${filament.td})`}
                    aria-label={`Reset TD of band ${i + 1} to ${filament.td}`}
                    data-testid={`td-reset-${i}`}
                  >
                    <RotateCcw className="w-3 h-3" />
                  </button>
                )}
              </div>
              <div className="flex items-center justify-end gap-0.5">
                <button
                  onClick={(e) => {
                    e.stopPropagation()
                    handleInsert(i)
                  }}
                  disabled={!slider.enabled || slider.layer <= 0}
                  className="p-1 rounded text-gray-300 hover:text-gray-100 hover:bg-gray-700 disabled:opacity-30 disabled:hover:bg-transparent"
                  aria-label={`Insert a band above band ${i + 1}`}
                  title="Insert a band above this one"
                  data-testid={`insert-band-${i}`}
                >
                  <ListPlus className="w-3.5 h-3.5" />
                </button>
                <button
                  onClick={() => updateSlider(i, { enabled: !slider.enabled })}
                  className="p-1 rounded text-gray-300 hover:text-gray-100 hover:bg-gray-700"
                  aria-label={slider.enabled ? `Hide band ${i + 1}` : `Show band ${i + 1}`}
                  aria-pressed={slider.enabled}
                  title={slider.enabled ? 'Hide this band' : 'Show this band'}
                  data-testid={`toggle-${i}`}
                  data-enabled={slider.enabled}
                >
                  {slider.enabled ? <Eye className="w-3.5 h-3.5" /> : <EyeOff className="w-3.5 h-3.5" />}
                </button>
                <button
                  onClick={(e) => {
                    e.stopPropagation()
                    removeBand(i)
                  }}
                  className="p-1 rounded text-gray-400 hover:text-red-400 hover:bg-gray-700"
                  aria-label={`Delete band ${i + 1}`}
                  title="Delete this band"
                  data-testid={`remove-band-${i}`}
                >
                  <Trash2 className="w-3.5 h-3.5" />
                </button>
              </div>
            </div>
          )
        })}
        {visible.length === 0 && <p className="p-3 text-xs text-gray-400">No bands use this filament.</p>}
      </div>
    </div>
  )
}
