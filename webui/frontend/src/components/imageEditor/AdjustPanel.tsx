import React, { useState } from 'react'
import { ChevronDown, ChevronRight, Info } from 'lucide-react'
import {
  autoLevels,
  changedCount,
  COLOR_SLIDERS,
  curvesAreIdentity,
  DEFAULT_LEVELS,
  defaultAdjustments,
  DETAIL_SLIDERS,
  levelsAreIdentity,
  LIGHT_SLIDERS,
  SLIDERS,
  type Adjustments,
  type Levels,
  type SliderKey,
} from '../../lib/imageAdjust'
import { REDUCE_MAX_COLORS, REDUCE_MIN_COLORS, type Palette } from '../../lib/colorReduce'
import { CurveEditor } from './CurveEditor'
import { Histogram } from './Histogram'

type Bins = { r: Uint32Array; g: Uint32Array; b: Uint32Array; l: Uint32Array }

const formatValue = (key: SliderKey, v: number) => {
  const unit = SLIDERS[key].unit ?? ''
  const text = key === 'exposure' ? v.toFixed(2) : String(Math.round(v))
  return `${v > 0 && key !== 'soften' && key !== 'sharpen' ? '+' : ''}${text}${unit}`
}

/** One labelled slider; double-click the label (or the value) to reset. */
const Slider: React.FC<{
  label: string
  hint: string
  value: number
  min: number
  max: number
  step: number
  display: string
  neutral: number
  onBegin: () => void
  onChange: (v: number) => void
  testId: string
}> = ({ label, hint, value, min, max, step, display, neutral, onBegin, onChange, testId }) => (
  <label className="block" title={`${hint}\nDouble-click the name to reset.`}>
    <div className="flex items-center justify-between text-[11px]">
      <span
        className={`select-none ${value !== neutral ? 'text-gray-100' : 'text-gray-400'}`}
        onDoubleClick={(e) => {
          e.preventDefault()
          if (value === neutral) return
          onBegin()
          onChange(neutral)
        }}
      >
        {label}
      </span>
      <span className={`tabular-nums ${value !== neutral ? 'text-cyan-300' : 'text-gray-500'}`} data-testid={`${testId}-value`}>
        {display}
      </span>
    </div>
    <input
      type="range"
      min={min}
      max={max}
      step={step}
      value={value}
      onPointerDown={onBegin}
      onKeyDown={(e) => {
        if (e.key.startsWith('Arrow') || e.key === 'Home' || e.key === 'End' || e.key.startsWith('Page')) onBegin()
      }}
      onChange={(e) => onChange(Number(e.target.value))}
      className="w-full accent-cyan-500 h-4"
      aria-label={label}
      data-testid={testId}
    />
  </label>
)

const Section: React.FC<{
  id: string
  title: string
  changed: number
  open: boolean
  onToggle: () => void
  onReset?: () => void
  children: React.ReactNode
}> = ({ id, title, changed, open, onToggle, onReset, children }) => (
  <section className="border-t border-gray-800" data-testid={`editor-section-${id}`}>
    <div className="flex items-center">
      <button onClick={onToggle} aria-expanded={open} className="flex-1 flex items-center gap-1.5 py-2 text-xs font-medium text-gray-200 hover:text-white" data-testid={`editor-section-${id}-toggle`}>
        {open ? <ChevronDown className="w-3.5 h-3.5" /> : <ChevronRight className="w-3.5 h-3.5" />}
        {title}
        {changed > 0 && <span className="px-1.5 rounded-full bg-cyan-600/30 text-cyan-300 text-[10px] tabular-nums">{changed}</span>}
      </button>
      {onReset && changed > 0 && (
        <button onClick={onReset} className="text-[11px] text-gray-400 hover:text-gray-200" data-testid={`editor-section-${id}-reset`}>
          Reset
        </button>
      )}
    </div>
    {open && <div className="pb-3 space-y-2">{children}</div>}
  </section>
)

/** The adjustments of the selected layer, in sections. */
export const AdjustPanel: React.FC<{
  adjust: Adjustments
  histogram: Bins | null
  palette: Palette | null
  onBegin: () => void
  onChange: (adjust: Adjustments) => void
}> = ({ adjust, histogram, palette, onBegin, onChange }) => {
  const [open, setOpen] = useState<Record<string, boolean>>({ light: true, color: true })
  const toggle = (id: string) => setOpen((o) => ({ ...o, [id]: !o[id] }))
  const defaults = defaultAdjustments()

  const slider = (key: SliderKey) => {
    const s = SLIDERS[key]
    return (
      <Slider
        key={key}
        label={s.label}
        hint={s.hint}
        value={adjust[key]}
        min={s.min}
        max={s.max}
        step={s.step}
        display={formatValue(key, adjust[key])}
        neutral={0}
        onBegin={onBegin}
        onChange={(v) => onChange({ ...adjust, [key]: v })}
        testId={`adjust-${key}`}
      />
    )
  }

  const resetKeys = (keys: SliderKey[]) => {
    onBegin()
    const next = { ...adjust }
    for (const k of keys) next[k] = 0
    onChange(next)
  }

  const levels = adjust.levels
  const setLevels = (patch: Partial<Levels>) => onChange({ ...adjust, levels: { ...levels, ...patch } })
  const levelSlider = (key: keyof Levels, label: string, min: number, max: number, step: number, hint: string) => (
    <Slider
      label={label}
      hint={hint}
      value={levels[key]}
      min={min}
      max={max}
      step={step}
      display={key === 'gamma' ? levels.gamma.toFixed(2) : String(levels[key])}
      neutral={DEFAULT_LEVELS[key]}
      onBegin={onBegin}
      onChange={(v) => setLevels({ [key]: v })}
      testId={`levels-${key}`}
    />
  )

  return (
    <div data-testid="adjust-panel">
      <Section id="light" title="Light" changed={changedCount(adjust, LIGHT_SLIDERS)} open={!!open.light} onToggle={() => toggle('light')} onReset={() => resetKeys(LIGHT_SLIDERS)}>
        {LIGHT_SLIDERS.map(slider)}
      </Section>
      <Section id="color" title="Colour" changed={changedCount(adjust, COLOR_SLIDERS)} open={!!open.color} onToggle={() => toggle('color')} onReset={() => resetKeys(COLOR_SLIDERS)}>
        {COLOR_SLIDERS.map(slider)}
      </Section>
      <Section
        id="levels"
        title="Levels"
        changed={levelsAreIdentity(levels) ? 0 : 1}
        open={!!open.levels}
        onToggle={() => toggle('levels')}
        onReset={() => {
          onBegin()
          onChange({ ...adjust, levels: { ...DEFAULT_LEVELS } })
        }}
      >
        <div className="relative">
          <Histogram bins={histogram} channels="l" height={44} className="rounded bg-gray-950" />
          {/* Where the input black and white points cut the histogram. */}
          <div className="absolute inset-y-0 left-0 bg-black/50 pointer-events-none" style={{ width: `${(levels.inBlack / 255) * 100}%` }} />
          <div className="absolute inset-y-0 right-0 bg-black/50 pointer-events-none" style={{ width: `${((255 - levels.inWhite) / 255) * 100}%` }} />
        </div>
        <div className="flex justify-end">
          <button
            onClick={() => {
              if (!histogram) return
              onBegin()
              setLevels(autoLevels(histogram.l))
            }}
            className="text-[11px] text-cyan-400 hover:text-cyan-300"
            title="Stretch the picture's own darkest and brightest parts to full black and white"
            data-testid="levels-auto"
          >
            Auto
          </button>
        </div>
        {levelSlider('inBlack', 'Black point', 0, 254, 1, 'Everything this dark or darker becomes black.')}
        {levelSlider('gamma', 'Midtones', 0.1, 3, 0.01, 'Above 1 brightens the midtones, below 1 darkens them.')}
        {levelSlider('inWhite', 'White point', 1, 255, 1, 'Everything this bright or brighter becomes white.')}
        {levelSlider('outBlack', 'Darkest output', 0, 255, 1, 'Raise it to make black a dark grey (less contrast).')}
        {levelSlider('outWhite', 'Brightest output', 0, 255, 1, 'Lower it to make white a light grey.')}
      </Section>
      <Section
        id="curves"
        title="Curves"
        changed={curvesAreIdentity(adjust.curves) ? 0 : 1}
        open={!!open.curves}
        onToggle={() => toggle('curves')}
        onReset={() => {
          onBegin()
          onChange({ ...adjust, curves: defaults.curves })
        }}
      >
        <CurveEditor curves={adjust.curves} histogram={histogram?.l ?? null} onBegin={onBegin} onChange={(curves) => onChange({ ...adjust, curves })} />
      </Section>
      <Section id="detail" title="Detail" changed={changedCount(adjust, DETAIL_SLIDERS)} open={!!open.detail} onToggle={() => toggle('detail')} onReset={() => resetKeys(DETAIL_SLIDERS)}>
        {DETAIL_SLIDERS.map(slider)}
      </Section>
      <Section
        id="reduce"
        title="Fewer colours"
        changed={adjust.reduce.enabled ? 1 : 0}
        open={!!open.reduce}
        onToggle={() => toggle('reduce')}
      >
        <label className="flex items-center gap-2 text-xs text-gray-200">
          <input
            type="checkbox"
            checked={adjust.reduce.enabled}
            onChange={(e) => {
              onBegin()
              onChange({ ...adjust, reduce: { ...adjust.reduce, enabled: e.target.checked } })
            }}
            className="accent-cyan-500"
            data-testid="reduce-enabled"
          />
          Reduce to a few flat colours
        </label>
        {adjust.reduce.enabled && (
          <>
            <Slider
              label="Colours"
              hint="How many colours are left. Similar colours merge first; greys and colours are kept apart."
              value={adjust.reduce.colors}
              min={REDUCE_MIN_COLORS}
              max={REDUCE_MAX_COLORS}
              step={1}
              display={palette && palette.count !== adjust.reduce.colors ? `${palette.count} of ${adjust.reduce.colors}` : String(adjust.reduce.colors)}
              neutral={defaults.reduce.colors}
              onBegin={onBegin}
              onChange={(v) => onChange({ ...adjust, reduce: { ...adjust.reduce, colors: v } })}
              testId="reduce-colors"
            />
            {palette && palette.count > 0 && (
              <div className="flex flex-wrap gap-1" data-testid="reduce-palette" data-count={palette.count}>
                {Array.from({ length: palette.count }, (_, i) => {
                  const hex = `#${[0, 1, 2].map((c) => palette.rgb[i * 3 + c].toString(16).padStart(2, '0')).join('')}`
                  return (
                    <span
                      key={i}
                      className="w-5 h-5 rounded border border-gray-600"
                      style={{ background: hex }}
                      title={`${hex.toUpperCase()} · ${(palette.share[i] * 100).toFixed(1)}% of the area`}
                    />
                  )
                })}
              </div>
            )}
          </>
        )}
        <p className="flex gap-1.5 text-[11px] text-gray-400 leading-snug">
          <Info className="w-3.5 h-3.5 flex-shrink-0 mt-px text-gray-500" />
          <span>
            A look, not a quality boost: the optimizer picks its colours itself and usually matches a picture more closely
            without this. Use it for a flat, poster-like style, or on a selection, like a busy background.
          </span>
        </p>
      </Section>
    </div>
  )
}
